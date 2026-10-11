// The full GLM-5.3 resident loader on the synthetic fixture: every class
// lands byte-exact at worlds 1, 2 and 4 as the slice formulas of
// docs/glm53_plan.md §2 say (the fused q_a | kv_a rows, q_b and kv_b head
// blocks, o_proj packed columns, the packed row/column slices at group
// granularity, the replicated set verbatim, kv_b's bf16 bridge equal to
// the exact dequant rounded once, the draft's BF16 experts requantized
// within one code step); the byte formulas equal actual usage and the
// source-byte plan equals the bytes read; the globals; resident cache hits
// and the image round trip; a shape record that disagrees with the packed
// geometry is refused by name.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "glm_dsa_fixture.hpp"
#include "loaders/packq_quant.hpp"
#include "models/glm_dsa/binding.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::GlmDsaExpectedTensor;
using dgpp::GlmDsaLayerStream;
using dgpp::GlmDsaTextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  GlmDsaTextConfig cfg;
  std::string dir;
  std::vector<GlmDsaExpectedTensor> table;
  const GlmDsaExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const {
    return glmdsafx::fixture_bytes(cfg, expected(name), table);
  }
};

// `nf4i8`: the NF4I8 contract's fixture (codebook routed experts under
// `weight_indices`, bf16 scales per 128, a 512-wide inter so a world-4
// slice holds a whole group).
// `m346` (2026-10-09): the Mixed346 contract's fixture — every routed
// expert of a packed layer in its own form (3-, 4- or 6-bit gate/up and
// down, or the baseline int4 g64 triple), bf16 scales per 128.
Fixture write_fixture(bool nf4i8 = false, bool m346 = false) {
  Fixture fx;
  fx.cfg = glmdsafx::tiny_config(5, 1, true, nf4i8, 8, m346);
  fx.dir = (fs::current_path() / (m346 ? "glm_dsa_loader_fixture_m346" : nf4i8 ? "glm_dsa_loader_fixture_nf4i8" : "glm_dsa_loader_fixture")).string();
  glmdsafx::write_fixture(fx.cfg, fx.dir, nf4i8, 8, m346);
  fx.table = dgpp::glm_dsa_expected_text_tensors(fx.cfg);
  return fx;
}
// The three contracts' fixtures: {nf4i8, m346}.
constexpr std::pair<bool, bool> kFixtureKinds[] = {{false, false}, {true, false}, {false, true}};

std::vector<uint8_t> device_bytes(const void* dev, size_t n) {
  std::vector<uint8_t> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}
std::vector<uint8_t> host_rows(const std::vector<uint8_t>& t, size_t row_bytes, int64_t r0, int64_t rows) {
  return std::vector<uint8_t>(t.begin() + r0 * row_bytes, t.begin() + (r0 + rows) * row_bytes);
}
std::vector<uint8_t> host_cols(const std::vector<uint8_t>& t, int64_t rows, size_t full, size_t c0, size_t w) {
  std::vector<uint8_t> out;
  out.reserve(rows * w);
  for (int64_t r = 0; r < rows; ++r)
    out.insert(out.end(), t.begin() + r * full + c0, t.begin() + r * full + c0 + w);
  return out;
}
std::vector<uint8_t> concat(std::vector<uint8_t> a, const std::vector<uint8_t>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
void expect_device_equals(const void* dev, const std::vector<uint8_t>& want, const std::string& what) {
  const std::vector<uint8_t> got = device_bytes(dev, want.size());
  require(got == want, what + ": resident bytes differ from the checkpoint slice");
}

// A packed view against the checkpoint's row / column slice of `base`:
// the triple's form (the scale record's scale_fmt) names the words tensor
// and the group.
void check_packed(const Fixture& fx, const dgpp::GlmPackedMatrix& m, const std::string& base, bool rows_slice,
                  int64_t start, int64_t count, const std::string& tag) {
  const auto& es = fx.expected(base + ".weight_scale");
  const int fmt = es.scale_fmt;
  const int64_t g = dgpp::packed_scale_group(fmt);
  const std::string words = base + dgpp::glm_dsa_packed_words_suffix(fmt);
  const auto& ep = fx.expected(words);
  const int bits = ep.bits;
  const int64_t N = ep.shape[0], K = ep.shape[1] * 32 / bits;
  const size_t wpr = static_cast<size_t>(K * bits / 32) * 4, spr = static_cast<size_t>(K / g) * 2;
  require(m.bits == bits, tag + " width");
  require(m.scale_fmt == fmt, tag + " scale format");
  if (rows_slice) {
    require(m.rows == count && m.cols == K, tag + " geometry");
    expect_device_equals(m.packed, host_rows(fx.bytes(words), wpr, start, count), tag + " word rows");
    expect_device_equals(m.scales, host_rows(fx.bytes(base + ".weight_scale"), spr, start, count), tag + " scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + " geometry");
    expect_device_equals(m.packed, host_cols(fx.bytes(words), N, wpr, start * bits / 8, count * bits / 8), tag + " word cols");
    expect_device_equals(m.scales, host_cols(fx.bytes(base + ".weight_scale"), N, spr, start / g * 2, count / g * 2), tag + " scale cols");
  }
}

// A requantized draft matrix against its BF16 source slice: every decoded
// value within half a code step of the source (one group scale).
void check_requant(const Fixture& fx, const dgpp::GlmPackedMatrix& m, const std::string& base, bool rows_slice,
                   int64_t start, int64_t count, int bits, const std::string& tag) {
  const auto& e = fx.expected(base + ".weight");
  const int64_t N = e.shape[0], K = e.shape[1];
  const std::vector<uint8_t> src = fx.bytes(base + ".weight");
  const int64_t rows = rows_slice ? count : N, cols = rows_slice ? K : count;
  require(m.rows == rows && m.cols == cols && m.bits == bits, tag + " geometry");
  const std::vector<uint8_t> words = device_bytes(m.packed, static_cast<size_t>(rows * cols * bits / 8));
  const std::vector<uint8_t> scales = device_bytes(m.scales, static_cast<size_t>(rows * cols / 64 * 2));
  const uint32_t* w = reinterpret_cast<const uint32_t*>(words.data());
  const uint16_t* s = reinterpret_cast<const uint16_t*>(scales.data());
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t sr = rows_slice ? start + r : r, sc = rows_slice ? c : start + c;
      uint16_t bits16;
      std::memcpy(&bits16, src.data() + (sr * K + sc) * 2, 2);
      const float want = dgpp::bf16_bits_to_float(bits16);
      const float got = dgpp::packq_decode(w, s, cols, bits, r, c);
      // RTN puts every element within half a step; the group's maximum may
      // land on the clamped code when the bf16 scale rounded down, which
      // adds q_max / 256 steps (int4 ~0.03, int8 ~0.5).
      const float step = dgpp::bf16_bits_to_float(s[r * (cols / 64) + c / 64]);
      const float q_max = static_cast<float>((1 << (bits - 1)) - 1);
      require(std::fabs(got - want) <= (0.5f + q_max / 256.0f + 0.02f) * step + 1e-6f,
              tag + " requant element off");
    }
}

// kv_b's bf16 bridge: every element the exact dequant rounded once.
void check_kv_b_bridge(const Fixture& fx, const uint16_t* dev, const std::string& base, int64_t row_start,
                       int64_t rows, const std::string& tag) {
  const auto& ep = fx.expected(base + ".weight_packed");
  const int bits = ep.bits;
  const int64_t K = ep.shape[1] * 32 / bits;
  const std::vector<uint8_t> words = fx.bytes(base + ".weight_packed");
  const std::vector<uint8_t> scales = fx.bytes(base + ".weight_scale");
  const uint32_t* w = reinterpret_cast<const uint32_t*>(words.data());
  const uint16_t* s = reinterpret_cast<const uint16_t*>(scales.data());
  const std::vector<uint8_t> got = device_bytes(dev, static_cast<size_t>(rows * K) * 2);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < K; ++c) {
      uint16_t g;
      std::memcpy(&g, got.data() + (r * K + c) * 2, 2);
      const uint16_t want = dgpp::float_to_bf16_bits(dgpp::packq_decode(w, s, K, bits, row_start + r, c));
      require(g == want, tag + " kv_b bridge element");
    }
}

void check_layer(const Fixture& fx, const GlmDsaLayerStream& s, const dgpp::GlmDsaLayerResident& r) {
  const GlmDsaTextConfig& cfg = fx.cfg;
  const dgpp::GlmDsaLocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::glm_dsa_layer_prefix(cfg, r.layer);
  const size_t H2 = static_cast<size_t>(cfg.hidden_size) * 2;
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  const bool draft = r.layer == cfg.mtp_layer();
  const bool packed = cfg.attention_bits_of(r.layer) != 0;
  expect_device_equals(r.input_norm, fx.bytes(p + "input_layernorm.weight"), tag + "input norm");
  expect_device_equals(r.post_norm, fx.bytes(p + "post_attention_layernorm.weight"), tag + "post norm");
  const std::string ap = p + "self_attn.";
  const int64_t nope = cfg.qk_nope_head_dim, rope = cfg.qk_rope_head_dim, v = cfg.v_head_dim;
  const int64_t ql = cfg.q_lora_rank;
  expect_device_equals(r.attn.q_aln, fx.bytes(ap + "q_a_layernorm.weight"), tag + "q_a norm");
  expect_device_equals(r.attn.kv_aln, fx.bytes(ap + "kv_a_layernorm.weight"), tag + "kv_a norm");
  require(r.attn.local_heads == g.local_heads && r.attn.head_begin == g.head_begin, tag + "head geometry");
  require(r.attn.packed() == packed, tag + "attention format");
  const int64_t qb0 = g.head_begin * (nope + rope), qbn = g.local_heads * (nope + rope);
  const int64_t kb0 = g.head_begin * (nope + v), kbn = g.local_heads * (nope + v);
  const int64_t o0 = g.head_begin * v, on = g.local_heads * v;
  if (packed) {
    // The fused rows: q_a's then kv_a's, one matrix.
    const int64_t rows_a = ql, rows_b = cfg.kv_lora_rank + rope;
    require(r.attn.qkv_a_packed.rows == rows_a + rows_b && r.attn.qkv_a_packed.cols == cfg.hidden_size, tag + "fused geometry");
    expect_device_equals(r.attn.qkv_a_packed.packed,
                         concat(fx.bytes(ap + "q_a_proj.weight_packed"), fx.bytes(ap + "kv_a_proj_with_mqa.weight_packed")),
                         tag + "fused words");
    expect_device_equals(r.attn.qkv_a_packed.scales,
                         concat(fx.bytes(ap + "q_a_proj.weight_scale"), fx.bytes(ap + "kv_a_proj_with_mqa.weight_scale")),
                         tag + "fused scales");
    check_packed(fx, r.attn.q_b_packed, ap + "q_b_proj", true, qb0, qbn, tag + "q_b");
    check_kv_b_bridge(fx, r.attn.kv_b, ap + "kv_b_proj", kb0, kbn, tag);
    check_packed(fx, r.attn.o_proj_packed, ap + "o_proj", false, o0, on, tag + "o_proj");
    require(r.attn.qkv_a == nullptr && r.attn.q_b == nullptr && r.attn.o_proj == nullptr, tag + "no bf16 attention");
  } else {
    expect_device_equals(r.attn.qkv_a, concat(fx.bytes(ap + "q_a_proj.weight"), fx.bytes(ap + "kv_a_proj_with_mqa.weight")), tag + "fused bf16");
    expect_device_equals(r.attn.q_b, host_rows(fx.bytes(ap + "q_b_proj.weight"), ql * 2, qb0, qbn), tag + "q_b rows");
    expect_device_equals(r.attn.kv_b, host_rows(fx.bytes(ap + "kv_b_proj.weight"), cfg.kv_lora_rank * 2, kb0, kbn), tag + "kv_b rows");
    const int64_t O = static_cast<int64_t>(cfg.num_attention_heads) * v;
    expect_device_equals(r.attn.o_proj, host_cols(fx.bytes(ap + "o_proj.weight"), cfg.hidden_size, O * 2, o0 * 2, on * 2), tag + "o cols");
    require(r.attn.qkv_a_packed.packed == nullptr && r.attn.o_proj_packed.packed == nullptr, tag + "no packed attention");
  }
  if (cfg.owns_indexer(r.layer)) {
    require(r.attn.owns_indexer(), tag + "indexer present");
    expect_device_equals(r.attn.wq_b, fx.bytes(ap + "indexer.wq_b.weight"), tag + "wq_b");
    expect_device_equals(r.attn.wk, fx.bytes(ap + "indexer.wk.weight"), tag + "wk");
    expect_device_equals(r.attn.wp, fx.bytes(ap + "indexer.weights_proj.weight"), tag + "weights_proj");
    expect_device_equals(r.attn.k_norm_w, fx.bytes(ap + "indexer.k_norm.weight"), tag + "k_norm w");
    expect_device_equals(r.attn.k_norm_b, fx.bytes(ap + "indexer.k_norm.bias"), tag + "k_norm b");
  } else {
    require(!r.attn.owns_indexer(), tag + "no indexer on a shared layer");
  }
  if (!r.moe) {
    const int64_t I = g.local_dense_inter;
    require(r.dense.local_inter == I, tag + "dense inter");
    expect_device_equals(r.dense.gate, host_rows(fx.bytes(p + "mlp.gate_proj.weight"), H2, rank * I, I), tag + "dense gate");
    expect_device_equals(r.dense.up, host_rows(fx.bytes(p + "mlp.up_proj.weight"), H2, rank * I, I), tag + "dense up");
    expect_device_equals(r.dense.down, host_cols(fx.bytes(p + "mlp.down_proj.weight"), cfg.hidden_size, cfg.intermediate_size * 2, rank * I * 2, I * 2), tag + "dense down");
  } else {
    const std::string mp = p + "mlp.";
    expect_device_equals(r.moe_w.router, fx.bytes(mp + "gate.weight"), tag + "router");
    expect_device_equals(r.moe_w.router_bias, fx.bytes(mp + "gate.e_score_correction_bias"), tag + "router bias");
    const int64_t I = g.local_inter, S = g.local_shared_inter;
    require(r.moe_w.local_inter == I && r.moe_w.local_shared_inter == S, tag + "moe inter");
    require(static_cast<int>(r.moe_w.experts.size()) == cfg.n_routed_experts * 3, tag + "expert count");
    for (int e = 0; e < cfg.n_routed_experts; ++e) {
      const std::string ep = mp + "experts." + std::to_string(e) + ".";
      if (draft) {
        check_requant(fx, r.moe_w.expert(e, 0), ep + "gate_proj", true, rank * I, I, cfg.expert_bits, tag + "draft expert gate");
        check_requant(fx, r.moe_w.expert(e, 1), ep + "up_proj", true, rank * I, I, cfg.expert_bits, tag + "draft expert up");
        check_requant(fx, r.moe_w.expert(e, 2), ep + "down_proj", false, rank * I, I, cfg.expert_bits, tag + "draft expert down");
      } else {
        check_packed(fx, r.moe_w.expert(e, 0), ep + "gate_proj", true, rank * I, I, tag + "expert gate");
        check_packed(fx, r.moe_w.expert(e, 1), ep + "up_proj", true, rank * I, I, tag + "expert up");
        check_packed(fx, r.moe_w.expert(e, 2), ep + "down_proj", false, rank * I, I, tag + "expert down");
      }
    }
    const std::string sp = mp + "shared_experts.";
    if (draft) {
      check_requant(fx, r.moe_w.shared[0], sp + "gate_proj", true, rank * S, S, 8, tag + "draft shared gate");
      check_requant(fx, r.moe_w.shared[1], sp + "up_proj", true, rank * S, S, 8, tag + "draft shared up");
      check_requant(fx, r.moe_w.shared[2], sp + "down_proj", false, rank * S, S, 8, tag + "draft shared down");
    } else {
      check_packed(fx, r.moe_w.shared[0], sp + "gate_proj", true, rank * S, S, tag + "shared gate");
      check_packed(fx, r.moe_w.shared[1], sp + "up_proj", true, rank * S, S, tag + "shared up");
      check_packed(fx, r.moe_w.shared[2], sp + "down_proj", false, rank * S, S, tag + "shared down");
    }
  }
  if (draft) {
    expect_device_equals(r.enorm, fx.bytes(p + "enorm.weight"), tag + "enorm");
    expect_device_equals(r.hnorm, fx.bytes(p + "hnorm.weight"), tag + "hnorm");
    expect_device_equals(r.eh_proj, fx.bytes(p + "eh_proj.weight"), tag + "eh_proj");
    expect_device_equals(r.shared_head_norm, fx.bytes(p + "shared_head.norm.weight"), tag + "shared head norm");
  } else {
    require(r.enorm == nullptr && r.eh_proj == nullptr, tag + "no draft head on a main layer");
  }
  require(r.bytes == GlmDsaLayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

DGPP_TEST(glm_dsa_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  for (const auto& [nf4i8, m346] : kFixtureKinds) {
  const Fixture fx = write_fixture(nf4i8, m346);
  const int layers = fx.cfg.num_hidden_layers + 1;
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      GlmDsaLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::GlmDsaResidency::Streaming,
                          dgpp::GlmDsaHeadSharding::Full, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before == GlmDsaLayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
    }
  }
  }
}

DGPP_TEST(glm_dsa_loader_globals_slices) {
  const Fixture fx = write_fixture();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::GlmDsaHeadSharding::VocabSharded : dgpp::GlmDsaHeadSharding::Full;
      GlmDsaLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::GlmDsaResidency::Streaming, head);
      const auto& g = s.load_globals();
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      expect_device_equals(g.embed, fx.bytes("model.embed_tokens.weight"), "embed");
      expect_device_equals(g.final_norm, fx.bytes("model.norm.weight"), "final norm");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("lm_head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == GlmDsaLayerStream::lm_vocab_count(fx.cfg, rank, world, head), "lm head count");
      require(g.embed_vocab_begin == 0 && g.embed_vocab_count == fx.cfg.vocab_size, "the embedding is whole by default");
      require(g.bytes == GlmDsaLayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula");
    }
  }
  // engine.embed_sharding = vocab: each rank holds the lm head's rows of
  // the embedding (world 1 keeps the whole table); the bytes formula follows.
  GlmDsaLayerStream::set_embed_vocab_sharded(true);
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::GlmDsaHeadSharding::VocabSharded : dgpp::GlmDsaHeadSharding::Full;
      GlmDsaLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::GlmDsaResidency::Streaming, head);
      const auto& g = s.load_globals();
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      if (world == 1) {
        require(g.embed_vocab_begin == 0 && g.embed_vocab_count == fx.cfg.vocab_size, "world 1 keeps the whole table");
      } else {
        require(g.embed_vocab_begin == g.lm_vocab_begin && g.embed_vocab_count == g.lm_vocab_count,
                "the embedding slice is the lm head's");
      }
      expect_device_equals(g.embed, host_rows(fx.bytes("model.embed_tokens.weight"), H2, g.embed_vocab_begin, g.embed_vocab_count),
                           "embed slice");
      require(g.bytes == GlmDsaLayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula (sharded)");
    }
  }
  GlmDsaLayerStream::set_embed_vocab_sharded(false);
}

DGPP_TEST(glm_dsa_loader_resident_mode_and_image_round_trip) {
  for (const auto& [nf4i8, m346] : kFixtureKinds) {
  const Fixture fx = write_fixture(nf4i8, m346);
  const fs::path cache = fs::current_path() / (m346 ? "glm_dsa_loader_image_cache_m346" : nf4i8 ? "glm_dsa_loader_image_cache_nf4i8" : "glm_dsa_loader_image_cache");
  fs::remove_all(cache);
  const std::string saved = GlmDsaLayerStream::resident_image_dir();
  GlmDsaLayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.num_hidden_layers + 1;
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    GlmDsaLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::GlmDsaResidency::Resident, dgpp::GlmDsaHeadSharding::VocabSharded,
                        /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      check_layer(fx, s, r);
      const auto span = s.resident_layer_span(l);
      require(span.first != nullptr && span.second == r.bytes, "resident span");
      built[static_cast<size_t>(l)] = device_bytes(span.first, span.second);
    }
    (void)s.load_globals();
    require(s.image_layers_captured() == layers, "every layer captured");
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) (void)s.load_layer(l);
    require(s.source_bytes_read() == before, "cache hits read no storage");
    require(GlmDsaLayerStream::resident_bytes(fx.cfg, 1, 2, dgpp::GlmDsaHeadSharding::VocabSharded, true) ==
                [&] { size_t t = s.load_globals().bytes; for (int l = 0; l < layers; ++l) t += s.load_layer(l).bytes; return t; }(),
            "resident bytes formula");
    s.release_sources();
    require(s.sources_released(), "sources released");
    require(s.staging_released(), "the pinned staging mirror goes with the sources");
  }
  {
    GlmDsaLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::GlmDsaResidency::Resident, dgpp::GlmDsaHeadSharding::VocabSharded,
                        /*resident_mtp=*/true);
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)], "restored layer bitwise the built one");
      check_layer(fx, s, r);
    }
    require(s.image_layers_restored() == layers && s.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
    const dgpp::GlmDsaReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
  }
  {
    // The model's order — globals, every layer, then the release before
    // any cache exists: the mappings and the staging mirror go together,
    // and the materialized layers and globals stay readable afterwards.
    GlmDsaLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::GlmDsaResidency::Resident, dgpp::GlmDsaHeadSharding::VocabSharded,
                        /*resident_mtp=*/true);
    (void)s.load_globals();
    for (int l = 0; l < layers; ++l) (void)s.load_layer(l);
    require(!s.sources_released() && !s.staging_released(), "nothing released before the model asks");
    s.release_sources();
    require(s.sources_released() && s.staging_released(), "the mappings and the staging mirror go together");
    require(s.load_layer(0).layer == 0 && s.load_globals().bytes > 0, "resident layers and globals stay readable");
  }
  GlmDsaLayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
  }
}

DGPP_TEST(glm_dsa_loader_refuses_a_shape_record_that_disagrees) {
  const Fixture fx = write_fixture();
  // Corrupt layer 2's o_proj shape record (its K field).
  const fs::path shard = fs::path(fx.dir) / "model.safetensors";
  const std::string name = "model.layers.2.self_attn.o_proj.weight_shape";
  std::FILE* f = std::fopen(shard.c_str(), "r+b");
  require(f != nullptr, "open shard");
  uint64_t hlen = 0;
  require(std::fread(&hlen, 8, 1, f) == 1, "header length");
  std::string header(hlen, '\0');
  require(std::fread(header.data(), 1, hlen, f) == hlen, "header");
  const size_t at = header.find("\"" + name + "\"");
  require(at != std::string::npos, "tensor in header");
  const size_t off_at = header.find("\"data_offsets\":[", at);
  const uint64_t begin = std::stoull(header.substr(off_at + 16));
  const int64_t bad = 4096;
  std::fseek(f, static_cast<long>(8 + hlen + begin + 8), SEEK_SET);
  require(std::fwrite(&bad, 8, 1, f) == 1, "write K");
  std::fclose(f);
  GlmDsaLayerStream s(fx.cfg, fx.dir, 0, 1);
  bool refused = false;
  try {
    (void)s.load_layer(2);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("disagrees with the packed geometry") != std::string::npos;
  }
  require(refused, "a shape record off the packed geometry is refused by name");
  (void)s.load_layer(1);  // an untouched layer still loads
}

// The codebook triple's host decode: every element of a resident NF4I8
// expert slice is codebook[index] x bf16(scale) with the index the low
// nibble first and one scale per 128 columns — the exact value the
// kernels compute (quant_matrix.hpp), read back through packq_decode.
DGPP_TEST(glm_dsa_loader_nf4i8_experts_decode_through_the_codebook) {
  const Fixture fx = write_fixture(true);
  GlmDsaLayerStream s(fx.cfg, fx.dir, 1, 2);
  const auto& r = s.load_layer(2);
  const dgpp::GlmPackedMatrix& m = r.moe_w.expert(3, 2);  // the down projection: a column slice
  require(m.scale_fmt == dgpp::kPackedScaleBf16G128Nf4i8 && m.bits == 4 && m.cols == 256 && m.rows == 256,
          "a world-2 down slice: [hidden, inter/2] in the codebook format");
  const std::vector<uint8_t> words = device_bytes(m.packed, static_cast<size_t>(m.rows * m.cols / 2));
  const std::vector<uint8_t> scales = device_bytes(m.scales, static_cast<size_t>(m.rows * m.cols / 128 * 2));
  const uint32_t* w = reinterpret_cast<const uint32_t*>(words.data());
  const uint16_t* sc = reinterpret_cast<const uint16_t*>(scales.data());
  // Against the checkpoint's own bytes (columns [256, 512) of expert 3's down).
  const std::string base = "model.layers.2.mlp.experts.3.down_proj";
  const std::vector<uint8_t> src_w = fx.bytes(base + ".weight_indices");
  const std::vector<uint8_t> src_s = fx.bytes(base + ".weight_scale");
  const uint32_t* sw = reinterpret_cast<const uint32_t*>(src_w.data());
  const uint16_t* ss = reinterpret_cast<const uint16_t*>(src_s.data());
  int levels_seen[16] = {};
  for (int64_t row = 0; row < m.rows; ++row)
    for (int64_t c = 0; c < m.cols; ++c) {
      const uint32_t word = w[row * (m.cols / 8) + c / 8];
      const unsigned idx = (word >> (4 * (c % 8))) & 15u;
      ++levels_seen[idx];
      const float want = static_cast<float>(dgpp::kNf4i8Codebook[idx]) *
                         dgpp::bf16_bits_to_float(sc[row * (m.cols / 128) + c / 128]);
      const float got = dgpp::packq_decode(w, sc, m.cols, 4, row, c, dgpp::kPackedScaleBf16G128Nf4i8);
      require(got == want, "codebook decode");
      const float checkpoint = dgpp::packq_decode(sw, ss, 512, 4, row, 256 + c, dgpp::kPackedScaleBf16G128Nf4i8);
      require(got == checkpoint, "the slice decodes to the checkpoint's columns");
    }
  for (int l = 0; l < 16; ++l) require(levels_seen[l] > 0, "every codebook level occurs in the random fixture");
}

int main() { return dgpp::test::run_all(); }
