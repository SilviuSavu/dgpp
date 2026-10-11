#include "models/glm_dsa/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<GlmDsaExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         GlmDsaWeightClass cls, int layer, int expert = -1,
         GlmDsaTensorRole role = GlmDsaTensorRole::Plain, int bits = 0, int scale_fmt = 0) {
  out.push_back(GlmDsaExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role, bits,
                                     scale_fmt});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              GlmDsaWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// One [rows, cols] matrix at `bits` (0 = a BF16 `.weight`): the packed
// codes (`weight_packed`, or `weight_indices` under the codebook format),
// the per-group bf16 scales and the shape record.
void add_matrix(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
                GlmDsaWeightClass cls, int layer, int expert, int bits, int scale_fmt,
                GlmDsaTensorRole plain_role = GlmDsaTensorRole::Plain) {
  if (bits == 0) {
    add(out, base + ".weight", DType::BF16, {rows, cols}, cls, layer, expert, plain_role);
    return;
  }
  if (!packed_scale_fmt_known(scale_fmt) || scale_fmt == kPackedScaleF16G128)
    throw std::invalid_argument("glm_dsa binding: the packed scale format must be bf16 per 64, the NF4I8 codebook or Mixed346 on " + base);
  if (!packed_bits_allowed(scale_fmt, bits))
    throw std::invalid_argument("glm_dsa binding: packed width " + std::to_string(bits) + " is not the scale format's on " + base);
  const int group = packed_scale_group(scale_fmt);
  if (cols % group != 0 || (cols * bits) % 32 != 0)
    throw std::invalid_argument("glm_dsa binding: packed K must be a multiple of the group on " + base);
  add(out, base + glm_dsa_packed_words_suffix(scale_fmt), DType::I32, {rows, cols * bits / 32}, cls, layer,
      expert, GlmDsaTensorRole::IntPacked, bits, scale_fmt);
  add(out, base + ".weight_scale", DType::BF16, {rows, cols / group}, cls, layer, expert,
      GlmDsaTensorRole::IntScale, bits, scale_fmt);
  add(out, base + ".weight_shape", DType::I64, {2}, cls, layer, expert, GlmDsaTensorRole::IntShape, bits,
      scale_fmt);
}

void expect_attention(TensorList& out, const std::string& p, const GlmDsaTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t heads = cfg.num_attention_heads;
  const int64_t ql = cfg.q_lora_rank, kvl = cfg.kv_lora_rank;
  const int64_t nope = cfg.qk_nope_head_dim, rope = cfg.qk_rope_head_dim, v = cfg.v_head_dim;
  const GlmDsaWeightClass c = GlmDsaWeightClass::Attention;
  const int bits = cfg.attention_bits_of(layer);
  // The attention triples are the compressed-tensors form (bf16 per 64).
  const int sf = kPackedScaleBf16G64;
  add_bf16(out, p + "q_a_layernorm.weight", {ql}, GlmDsaWeightClass::LayerNorm, layer);
  add_bf16(out, p + "kv_a_layernorm.weight", {kvl}, GlmDsaWeightClass::LayerNorm, layer);
  add_matrix(out, p + "q_a_proj", ql, H, c, layer, -1, bits, sf);
  add_matrix(out, p + "q_b_proj", heads * (nope + rope), ql, c, layer, -1, bits, sf);
  add_matrix(out, p + "kv_a_proj_with_mqa", kvl + rope, H, c, layer, -1, bits, sf);
  add_matrix(out, p + "kv_b_proj", heads * (nope + v), kvl, c, layer, -1, bits, sf);
  add_matrix(out, p + "o_proj", H, heads * v, c, layer, -1, bits, sf);
  if (cfg.owns_indexer(layer)) {
    const std::string ip = p + "indexer.";
    const int64_t ih = cfg.index_n_heads, id = cfg.index_head_dim;
    const GlmDsaWeightClass ic = GlmDsaWeightClass::Indexer;
    add_bf16(out, ip + "wq_b.weight", {ih * id, ql}, ic, layer);
    add_bf16(out, ip + "wk.weight", {id, H}, ic, layer);
    add_bf16(out, ip + "weights_proj.weight", {ih, H}, ic, layer);
    add_bf16(out, ip + "k_norm.weight", {id}, ic, layer);
    add_bf16(out, ip + "k_norm.bias", {id}, ic, layer);
  }
}

void expect_dense_mlp(TensorList& out, const std::string& p, const GlmDsaTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  const GlmDsaWeightClass c = GlmDsaWeightClass::DenseMlp;
  add_bf16(out, p + "gate_proj.weight", {I, H}, c, layer);
  add_bf16(out, p + "up_proj.weight", {I, H}, c, layer);
  add_bf16(out, p + "down_proj.weight", {H, I}, c, layer);
}

void expect_moe(TensorList& out, const std::string& p, const GlmDsaTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t S = cfg.shared_expert_inter();
  const bool draft = layer == cfg.mtp_layer();
  add_bf16(out, p + "gate.weight", {cfg.n_routed_experts, H}, GlmDsaWeightClass::Router, layer);
  add(out, p + "gate.e_score_correction_bias", DType::F32, {cfg.n_routed_experts},
      GlmDsaWeightClass::Router, layer);
  const GlmDsaTensorRole plain = draft ? GlmDsaTensorRole::Bf16Expert : GlmDsaTensorRole::Plain;
  // The Mixed346 contract names every expert's form on its own (the
  // recipe); the other contracts one form a layer.
  for (int e = 0; e < cfg.n_routed_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    const int esf = draft ? kPackedScaleBf16G64 : cfg.expert_scale_fmt_of(layer, e);
    const int gb = draft ? 0 : cfg.expert_proj_bits_of(layer, e, 0);
    const int ub = draft ? 0 : cfg.expert_proj_bits_of(layer, e, 1);
    const int db = draft ? 0 : cfg.expert_proj_bits_of(layer, e, 2);
    add_matrix(out, ep + "gate_proj", I, H, GlmDsaWeightClass::RoutedExpert, layer, e, gb, esf, plain);
    add_matrix(out, ep + "up_proj", I, H, GlmDsaWeightClass::RoutedExpert, layer, e, ub, esf, plain);
    add_matrix(out, ep + "down_proj", H, I, GlmDsaWeightClass::RoutedExpert, layer, e, db, esf, plain);
  }
  const std::string sp = p + "shared_experts.";
  const int sb = draft ? 0 : cfg.shared_bits_of(layer);
  const int ssf = kPackedScaleBf16G64;
  add_matrix(out, sp + "gate_proj", S, H, GlmDsaWeightClass::SharedExpert, layer, -1, sb, ssf, plain);
  add_matrix(out, sp + "up_proj", S, H, GlmDsaWeightClass::SharedExpert, layer, -1, sb, ssf, plain);
  add_matrix(out, sp + "down_proj", H, S, GlmDsaWeightClass::SharedExpert, layer, -1, sb, ssf, plain);
}

int max_layer(const GlmDsaTextConfig& cfg) {
  return cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
}

}  // namespace

std::string glm_dsa_layer_prefix(const GlmDsaTextConfig&, int layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

const char* glm_dsa_packed_words_suffix(int scale_fmt) {
  return packed_codebook(scale_fmt) ? ".weight_indices" : ".weight_packed";
}

std::vector<GlmDsaExpectedTensor> glm_dsa_expected_layer_tensors(const GlmDsaTextConfig& cfg, int layer) {
  if (layer < 0 || layer >= max_layer(cfg))
    throw std::invalid_argument("glm_dsa_expected_layer_tensors: layer out of range");
  const bool is_mtp = layer == cfg.mtp_layer();
  const std::string p = glm_dsa_layer_prefix(cfg, layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  if (is_mtp) {
    // The draft head's own tensors (vLLM deepseek_mtp.py: enorm, hnorm,
    // eh_proj, shared_head.norm); the embedding and the head are shared
    // with the main model — this checkpoint carries no copies.
    add_bf16(out, p + "enorm.weight", {H}, GlmDsaWeightClass::MtpHead, layer);
    add_bf16(out, p + "hnorm.weight", {H}, GlmDsaWeightClass::MtpHead, layer);
    add_bf16(out, p + "eh_proj.weight", {H, 2 * H}, GlmDsaWeightClass::MtpHead, layer);
    add_bf16(out, p + "shared_head.norm.weight", {H}, GlmDsaWeightClass::MtpHead, layer);
  }
  add_bf16(out, p + "input_layernorm.weight", {H}, GlmDsaWeightClass::LayerNorm, layer);
  add_bf16(out, p + "post_attention_layernorm.weight", {H}, GlmDsaWeightClass::LayerNorm, layer);
  expect_attention(out, p + "self_attn.", cfg, layer);
  if (cfg.is_moe_layer(layer))
    expect_moe(out, p + "mlp.", cfg, layer);
  else
    expect_dense_mlp(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<GlmDsaExpectedTensor> glm_dsa_expected_global_tensors(const GlmDsaTextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.embed_tokens.weight", {cfg.vocab_size, H}, GlmDsaWeightClass::Embed, -1);
  add_bf16(out, "model.norm.weight", {H}, GlmDsaWeightClass::FinalNorm, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, GlmDsaWeightClass::LmHead, -1);
  return out;
}

std::vector<GlmDsaExpectedTensor> glm_dsa_expected_text_tensors(const GlmDsaTextConfig& cfg) {
  TensorList out = glm_dsa_expected_global_tensors(cfg);
  for (int l = 0; l < max_layer(cfg); ++l) {
    TensorList layer = glm_dsa_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

GlmDsaBindReport glm_dsa_validate_text_binding(
    const GlmDsaTextConfig& cfg, const std::unordered_map<std::string, GlmDsaTensorDesc>& present,
    size_t max_errors) {
  GlmDsaBindReport rep;
  const auto expected = glm_dsa_expected_text_tensors(cfg);
  rep.expected = expected.size();
  auto push_error = [&](std::string msg) {
    if (rep.errors.size() < max_errors) rep.errors.push_back(std::move(msg));
  };
  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(s[i]);
    }
    return out + "]";
  };
  std::unordered_map<std::string, int8_t> consumed;
  consumed.reserve(present.size());
  for (const auto& e : expected) {
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++rep.missing;
      push_error(std::format("missing tensor '{}'", e.name));
      continue;
    }
    consumed.emplace(e.name, 1);
    if (it->second.dtype != e.dtype) {
      ++rep.dtype_mismatch;
      push_error(std::format("'{}' dtype {} != expected {}", e.name, dtype_name(it->second.dtype),
                             dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name, shape_str(it->second.shape),
                             shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    if (e.role == GlmDsaTensorRole::IntPacked) {
      if (e.bits == 4) ++rep.packed_int4_matrices;
      else if (e.bits == 8) ++rep.packed_int8_matrices;
      else ++rep.packed_mixed_matrices;
      if (packed_codebook(e.scale_fmt)) ++rep.codebook_matrices;
    }
    if (e.role == GlmDsaTensorRole::Bf16Expert) ++rep.bf16_expert_matrices;
  }
  // The layer count the config's stack ends at: a tensor of a main layer
  // past it belongs to a truncated diagnostic stack (the checkpoint's
  // other layers are simply not walked), not to a binding error.
  const int stack_end = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  const auto beyond_stack = [&](const std::string& name) {
    constexpr std::string_view prefix = "model.layers.";
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    size_t i = prefix.size();
    int layer = 0;
    bool digits = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
      layer = layer * 10 + (name[i] - '0');
      ++i;
      digits = true;
    }
    return digits && i < name.size() && name[i] == '.' && layer >= stack_end;
  };
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (beyond_stack(name)) {
      ++rep.beyond_stack;
      continue;
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void glm_dsa_tp_validate_geometry(const GlmDsaTextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("glm_dsa tp geometry: " + what);
  };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.vocab_size % world != 0) fail("vocab_size must divide by world (the head is vocab-sharded)");
  const int g = cfg.packed_group_size;
  // o_proj's input is the local heads' value rows: a packed column slice.
  const int local_heads = cfg.num_attention_heads / world;
  if ((local_heads * cfg.v_head_dim) % g != 0)
    fail("the o_proj input slice must be a multiple of the packed group");
  for (const int inter : {cfg.intermediate_size, cfg.moe_intermediate_size, cfg.shared_expert_inter()}) {
    if (inter % world != 0) fail("an intermediate size must divide by world");
    if ((inter / world) % g != 0)
      fail("an intermediate slice must be a multiple of " + std::to_string(g) +
           " (the packed group and the down projection's column slice)");
  }
  // The routed experts' own group (128 under the codebook format): the
  // down projection's column slice must hold whole scale groups, which
  // also keeps every 32-wide rotation block on one rank.
  const int eg = cfg.expert_group_size();
  if ((cfg.moe_intermediate_size / world) % eg != 0)
    fail("the routed expert slice must be a multiple of " + std::to_string(eg) +
         " (the routed experts' scale group)");
}

}  // namespace dgpp
