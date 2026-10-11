// The full GLM-5.3 world-1 forward (docs/glm53_plan.md G5, 2026-09-12).
//
// Modes:
//   --write-fixture DIR          the tiny synthetic checkpoint (tests/cuda/glm_dsa_fixture.hpp)
//   --smoke DIR                  GlmDsaModel over the fixture: finite outputs, bitwise repeat,
//                                the per-token selections' shape
//   --checkpoint-dir DIR
//   --dump-file FILE             GlmDsaModel against tools/glm_dsa_reference_dump.py's pure
//                                double reference: every layer's residual, the final read,
//                                the per-token top-k logits, the routing decisions, every
//                                indexed layer's selection (flips certified against the
//                                reference's boundary margins), the draft block's rows
//
// The ctest chain: glm_dsa_forward_fixture -> glm_dsa_forward_smoke, and
// glm_dsa_forward_fixture -> glm_dsa_forward_generate (python) -> glm_dsa_forward_test.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/model.hpp"
#include "glm_dsa_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::GlmDsaModel;
using dgpp::GlmDsaTextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<int64_t> smoke_tokens(const GlmDsaTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

// ---- the reference dump ------------------------------------------------------

struct Dump {
  struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
  };
  std::vector<uint8_t> bytes;
  std::map<std::string, Tensor> tensors;
  int hidden = 0, vocab = 0, num_layers = 0, top_k = 0, moe_layers = 0, index_layers = 0, max_selected = 0;
  int64_t token_count = 0;

  static Dump load(const std::string& path) {
    Dump d;
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "dump: cannot open " + path);
    d.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    require(d.bytes.size() >= 16 && std::memcmp(d.bytes.data(), "DGPPGDSD", 8) == 0, "dump: bad magic");
    uint32_t version = 0, header_len = 0;
    std::memcpy(&version, d.bytes.data() + 8, 4);
    std::memcpy(&header_len, d.bytes.data() + 12, 4);
    require(version == 1, "dump: unsupported version");
    require(16 + static_cast<size_t>(header_len) <= d.bytes.size(), "dump: header exceeds file");
    const std::string_view header(reinterpret_cast<const char*>(d.bytes.data() + 16), header_len);
    dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(header);
    const dgpp::minijson::Value& root = parsed.root;
    const dgpp::minijson::Value& cfg = root.at("config");
    d.hidden = static_cast<int>(cfg.at("hidden").as_int());
    d.vocab = static_cast<int>(cfg.at("vocab").as_int());
    d.num_layers = static_cast<int>(cfg.at("num_layers").as_int());
    d.top_k = static_cast<int>(cfg.at("top_k").as_int());
    d.moe_layers = static_cast<int>(cfg.at("moe_layers").as_int());
    d.index_layers = static_cast<int>(cfg.at("index_layers").as_int());
    d.max_selected = static_cast<int>(cfg.at("max_selected").as_int());
    d.token_count = cfg.at("tokens").as_int();
    const size_t payload_base = 16 + header_len;
    for (const auto& m : root.at("tensors").members()) {
      Tensor t;
      t.dtype = std::string(m.value.at("dtype").as_string());
      for (const auto& dim : m.value.at("shape").items()) t.shape.push_back(dim.as_int());
      const uint64_t offset = m.value.at("offset").as_int();
      t.nbytes = static_cast<size_t>(m.value.at("nbytes").as_int());
      require(payload_base + offset + t.nbytes <= d.bytes.size(), "dump: tensor exceeds payload");
      t.data = d.bytes.data() + payload_base + offset;
      d.tensors[m.key] = t;
    }
    return d;
  }
  const Tensor& tensor(const std::string& name) const {
    auto it = tensors.find(name);
    require(it != tensors.end(), "dump: missing tensor " + name);
    return it->second;
  }
};

struct Stats {
  double l2 = 0, max_ulps = 0;
  long soft = 0, hard = 0, total = 0;
};

// bf16 ulps with a cancellation floor at 2 % of the reference's RMS, the
// soft/hard counts and the relative l2.
Stats compare_bf16(const uint16_t* got, const uint16_t* want, size_t n, int soft, int hard) {
  Stats s;
  s.total = static_cast<long>(n);
  double rms = 0;
  for (size_t i = 0; i < n; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
  rms = std::sqrt(rms / std::max<size_t>(n, 1));
  const double floor_abs = 0.02 * rms;
  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < n; ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    if (std::fabs(g - w) <= floor_abs) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    sum_d2 += (g - w) * (g - w);
    sum_o2 += w * w;
    if (u > soft) ++s.soft;
    if (u > hard) ++s.hard;
  }
  s.l2 = std::sqrt(sum_d2) / std::sqrt(sum_o2 + 1e-30);
  return s;
}

// The per-token selections' shape at kpool 1: min(p + 1, topk) ascending
// token ids in [0, p], -1 padded.
void check_selection_shape(const std::vector<int32_t>& sel, int T, int ms, int topk, const char* what) {
  require(sel.size() == static_cast<size_t>(T) * ms, std::string(what) + ": selection rows");
  for (int t = 0; t < T; ++t) {
    const int n = std::min(t + 1, topk);
    for (int i = 0; i < ms; ++i) {
      const int32_t v = sel[static_cast<size_t>(t) * ms + i];
      if (i >= n) require(v == -1, std::string(what) + ": padding past the selection");
      else require(v >= 0 && v <= t && (i == 0 || v > sel[static_cast<size_t>(t) * ms + i - 1]),
                   std::string(what) + ": selection of row " + std::to_string(t));
    }
  }
}

int run_smoke(const std::string& dir) {
  const GlmDsaTextConfig cfg = GlmDsaTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 72;
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  GlmDsaModel model(cfg, dir, T, 256);
  const GlmDsaModel::Outputs out = model.forward(tokens, true);
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "smoke: layer captures");
  require(out.dsa_selections.size() == static_cast<size_t>(cfg.num_indexer_layers()), "smoke: selection captures");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double rms = 0, mx = 0;
    for (uint16_t v : out.layer_states[l]) {
      const float f = bf16_bits_to_float(v);
      require(std::isfinite(f), "smoke: non-finite hidden state at layer " + std::to_string(l));
      rms += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
    std::printf("[ .. ] layer %zu: h rms %.4g max %.4g\n", l, rms, mx);
  }
  const int ms = model.dsa_cfg().index_topk;  // max_selected at kpool 1
  for (size_t l = 0; l < out.dsa_selections.size(); ++l)
    check_selection_shape(out.dsa_selections[l], T, ms, cfg.index_topk, "smoke");
  for (float v : out.logits) require(std::isfinite(v), "smoke: non-finite logit");
  const GlmDsaModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "smoke: forward is not deterministic across calls");
  const auto top = GlmDsaModel::topk(out.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke: %d tokens, %d layers (%d indexed), top-1 of the last row %d (%.4g); deterministic\n",
              T, cfg.num_hidden_layers, cfg.num_indexer_layers(), top.back()[0].first, top.back()[0].second);
  std::printf("[ OK ] glm_dsa_forward_smoke\n");
  return 0;
}

int run_dump_parity(const std::string& dir, const std::string& dump_path) {
  const Dump dump = Dump::load(dump_path);
  const GlmDsaTextConfig cfg = GlmDsaTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  require(cfg.hidden_size == dump.hidden && cfg.vocab_size == dump.vocab &&
              cfg.num_hidden_layers == dump.num_layers && cfg.num_moe_layers() == dump.moe_layers &&
              cfg.num_indexer_layers() == dump.index_layers && cfg.index_topk == dump.max_selected,
          "dump config disagrees with the checkpoint config");
  const Dump::Tensor& tok = dump.tensor("tokens");
  require(tok.dtype == "I64", "dump: tokens dtype");
  const int T = static_cast<int>(dump.token_count);
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  std::memcpy(tokens.data(), tok.data, static_cast<size_t>(T) * 8);

  GlmDsaModel model(cfg, dir, T, std::max(256, T));
  const GlmDsaModel::Outputs out = model.forward(tokens, true);
  const GlmDsaModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "forward is not deterministic across calls");

  const int H = cfg.hidden_size, W = H;
  bool ok = true;
  // ---- the selections: flips certified as near ties -----------------------------
  // A selection that differs from the reference's is a wiring bug unless the
  // reference's boundary (the select_k-th vs the next logit) was a near tie
  // (the fp32 kernels vs the double oracle, the fp8 codes a Hadamard
  // rounding apart). Flipped rows and every later row of that layer are
  // excluded from the residual budgets below — a different selection is a
  // different (legitimate) attention output.
  const int ms = dump.max_selected;
  const Dump::Tensor& dsel = dump.tensor("dsa_selections");
  const Dump::Tensor& dmarg = dump.tensor("dsa_margins");
  require(dsel.dtype == "I32" && dsel.shape.size() == 3 && dsel.shape[0] == dump.index_layers &&
              dsel.shape[1] == T && dsel.shape[2] == ms,
          "dump: dsa_selections shape");
  require(dmarg.dtype == "F32" && dmarg.shape[0] == dump.index_layers && dmarg.shape[1] == T,
          "dump: dsa_margins shape");
  require(out.dsa_selections.size() == static_cast<size_t>(dump.index_layers), "captured selections");
  std::vector<bool> excluded(static_cast<size_t>(T), false);  // rows whose selection flipped, per layer accumulate
  int flips = 0, uncertified = 0;
  {
    const int32_t* ref = reinterpret_cast<const int32_t*>(dsel.data);
    const float* marg = reinterpret_cast<const float*>(dmarg.data);
    for (int l = 0; l < dump.index_layers; ++l) {
      check_selection_shape(out.dsa_selections[static_cast<size_t>(l)], T, ms, cfg.index_topk, "parity");
      for (int t = 0; t < T; ++t) {
        const int32_t* g = out.dsa_selections[static_cast<size_t>(l)].data() + static_cast<size_t>(t) * ms;
        const int32_t* r = ref + (static_cast<size_t>(l) * T + t) * ms;
        if (std::memcmp(g, r, static_cast<size_t>(ms) * 4) == 0) continue;
        ++flips;
        const float m = marg[static_cast<size_t>(l) * T + t];
        std::set<int32_t> gs, rs;
        for (int i = 0; i < ms; ++i) {
          if (g[i] >= 0) gs.insert(g[i]);
          if (r[i] >= 0) rs.insert(r[i]);
        }
        std::string only_g, only_r;
        for (int32_t v : gs) if (!rs.count(v)) only_g += " " + std::to_string(v);
        for (int32_t v : rs) if (!gs.count(v)) only_r += " " + std::to_string(v);
        // Certified when the reference's boundary gap is within the noise
        // on the logits: the first indexed layer sees pristine inputs, so
        // only the fp32-vs-double chain and a Hadamard rounding separate the
        // two (2e-3 of the row's logit scale); a deeper layer's inputs have
        // drifted ~1 % by then, and the fp8 quantization of q and k turns
        // that into code flips worth several percent of a logit (the
        // fixture's random weights leave the boundary gaps of the same
        // order, so flips are common there and mean nothing by themselves;
        // the strict first layer is the pin).
        const double budget = l == 0 ? 2e-3 : 1e-1;
        const bool certified = m >= 0 && m <= budget;
        std::printf("[ .. ]   selection flip: indexed layer %d token %d (reference boundary margin %.3g%s): "
                    "engine-only%s, reference-only%s\n", l, t, m, certified ? "" : ", UNCERTIFIED",
                    only_g.c_str(), only_r.c_str());
        if (!certified) ++uncertified;
        excluded[static_cast<size_t>(t)] = true;
      }
    }
    std::printf("[ .. ] selections: %d flips of %d rows, %d uncertified\n", flips, dump.index_layers * T, uncertified);
    if (uncertified > 0) ok = false;
    // The first indexed layer must select exactly as the reference (its
    // rows are the pristine pin); the deeper layers may flip at most half
    // of their sparse rows.
    {
      int first = 0, deeper = 0;
      for (int t = 0; t < T; ++t) {
        const int32_t* g = out.dsa_selections[0].data() + static_cast<size_t>(t) * ms;
        if (std::memcmp(g, ref + static_cast<size_t>(t) * ms, static_cast<size_t>(ms) * 4) != 0) ++first;
      }
      deeper = flips - first;
      const int sparse_rows = std::max(0, T - cfg.index_topk) * std::max(0, dump.index_layers - 1);
      std::printf("[ .. ] selections: first indexed layer %d flips (must be 0); deeper layers %d of %d sparse rows\n",
                  first, deeper, sparse_rows);
      if (first != 0 || deeper > sparse_rows / 2) ok = false;
    }
  }
  // ---- routing (before the residual budgets: a flipped row is excluded) ----
  if (dump.tensors.count("route_ids")) {
    const Dump::Tensor& rt = dump.tensor("route_ids");
    const Dump::Tensor& rm = dump.tensor("route_margins");
    const int K = static_cast<int>(rt.shape[2]);
    require(rt.shape[0] == dump.moe_layers && out.route_ids.size() == static_cast<size_t>(dump.moe_layers),
            "dump: route_ids layers");
    require(rm.dtype == "F32" && rm.shape[0] == dump.moe_layers && rm.shape[1] == T, "dump: route_margins shape");
    const int32_t* ref = reinterpret_cast<const int32_t*>(rt.data);
    const float* marg = reinterpret_cast<const float*>(rm.data);
    long diff = 0, total = 0, route_uncertified = 0;
    for (int l = 0; l < dump.moe_layers; ++l)
      for (int t = 0; t < T; ++t) {
        if (excluded[static_cast<size_t>(t)]) continue;
        bool row_diff = false;
        for (int i = 0; i < K; ++i, ++total)
          if (out.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i] !=
              ref[(static_cast<size_t>(l) * T + t) * K + i]) {
            ++diff;
            row_diff = true;
          }
        if (row_diff) {
          // The router's boundary gap (sigmoid units): the engine's input
          // hidden drifts ~1 % from the reference's by the deeper layers,
          // so a gap under 2e-2 is a near tie; a wider one a wiring bug.
          const float m = marg[static_cast<size_t>(l) * T + t];
          const bool certified = m <= 2e-2f;
          if (!certified) ++route_uncertified;
          std::printf("[ .. ]   route flip: moe layer %d token %d (reference boundary margin %.3g%s) got", l, t, m,
                      certified ? "" : ", UNCERTIFIED");
          for (int i = 0; i < K; ++i) std::printf(" %d", out.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i]);
          std::printf(" ref");
          for (int i = 0; i < K; ++i) std::printf(" %d", ref[(static_cast<size_t>(l) * T + t) * K + i]);
          std::printf("\n");
          excluded[static_cast<size_t>(t)] = true;
        }
      }
    std::printf("[ .. ] routing: %ld of %ld slots differ from the reference, %ld uncertified\n", diff, total,
                route_uncertified);
    if (route_uncertified > 0 || (total > 0 && diff > total / 10)) ok = false;
  }
  // ---- per-layer hidden states ------------------------------------------------
  const Dump::Tensor& ls = dump.tensor("layer_states");
  require(ls.dtype == "BF16" && ls.shape.size() == 3 && ls.shape[0] == cfg.num_hidden_layers &&
              ls.shape[1] == T && ls.shape[2] == W,
          "dump: layer_states shape");
  const auto layer_stats = [&](const uint16_t* got, const uint16_t* want, Stats* kept) {
    // The kept rows' statistics (flipped rows reported separately).
    std::vector<uint16_t> g, w;
    for (int t = 0; t < T; ++t) {
      if (excluded[static_cast<size_t>(t)]) continue;
      g.insert(g.end(), got + static_cast<size_t>(t) * W, got + static_cast<size_t>(t + 1) * W);
      w.insert(w.end(), want + static_cast<size_t>(t) * W, want + static_cast<size_t>(t + 1) * W);
    }
    *kept = compare_bf16(g.data(), w.data(), g.size(), 16, 128);
  };
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const uint16_t* want = reinterpret_cast<const uint16_t*>(ls.data) + static_cast<size_t>(l) * T * W;
    const uint16_t* got = out.layer_states[static_cast<size_t>(l)].data();
    Stats s;
    layer_stats(got, want, &s);
    std::printf("[ .. ] layer %d h (kept rows): l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", l, s.l2,
                s.max_ulps, s.soft, s.hard, s.total);
    std::vector<std::pair<double, int>> rows;
    for (int t = 0; t < T; ++t) {
      const Stats r = compare_bf16(got + static_cast<size_t>(t) * W, want + static_cast<size_t>(t) * W,
                                   static_cast<size_t>(W), 16, 128);
      rows.emplace_back(r.l2, t);
    }
    std::sort(rows.rbegin(), rows.rend());
    std::printf("[ .. ]   worst rows:");
    for (int i = 0; i < 4 && i < T; ++i)
      std::printf(" t%d %.3g%s", rows[static_cast<size_t>(i)].second, rows[static_cast<size_t>(i)].first,
                  excluded[static_cast<size_t>(rows[static_cast<size_t>(i)].second)] ? "(flip)" : "");
    std::printf("\n");
    // The per-layer budget: 0.01 relative l2, or 0.015 under the int8
    // activation contract (the Mixed346 checkpoint): its code flips roughly
    // double a layer's response to upstream noise (glm_moe_test's
    // sensitivity probe: row l2 0.0044 per input ulp against 0.0025 for the
    // bf16-activation formats), so the reference's own reassociation reaches
    // the output amplified; the final read's budget was 0.015 already.
    const double layer_budget = cfg.expert_activation_bits == 8 ? 0.015 : 0.01;
    if (s.total > 0 && (s.l2 > layer_budget || static_cast<double>(s.hard) / s.total > 0.005)) ok = false;
  }
  // ---- the final read ---------------------------------------------------------
  const Dump::Tensor& fh = dump.tensor("final_hidden");
  require(fh.dtype == "BF16" && fh.shape.size() == 2 && fh.shape[0] == T && fh.shape[1] == H,
          "dump: final_hidden shape");
  {
    Stats fs_;
    layer_stats(out.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(fh.data), &fs_);
    std::printf("[ .. ] final hidden (kept rows): l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", fs_.l2,
                fs_.max_ulps, fs_.soft, fs_.hard, fs_.total);
    if (fs_.total > 0 && (fs_.l2 > 0.015 || static_cast<double>(fs_.hard) / fs_.total > 0.005)) ok = false;
  }
  // ---- logits: top-1 (near ties certified), top-k overlap --------------------
  const Dump::Tensor& tid = dump.tensor("topk_ids");
  const Dump::Tensor& tlog = dump.tensor("topk_logits");
  const int k = dump.top_k;
  require(tid.dtype == "I32" && tlog.dtype == "F32" && tid.shape[0] == T && tid.shape[1] == k, "dump: topk shape");
  const int32_t* ref_ids = reinterpret_cast<const int32_t*>(tid.data);
  const float* ref_vals = reinterpret_cast<const float*>(tlog.data);
  const auto got = GlmDsaModel::topk(out.logits, T, cfg.vocab_size, k);
  int top1_hard = 0, top1_soft = 0, set_miss = 0, kept_rows = 0;
  double max_rel = 0;
  for (int t = 0; t < T; ++t) {
    if (excluded[static_cast<size_t>(t)]) continue;
    ++kept_rows;
    const int32_t r1 = ref_ids[static_cast<size_t>(t) * k];
    const float v1 = ref_vals[static_cast<size_t>(t) * k], v2 = ref_vals[static_cast<size_t>(t) * k + 1];
    const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
    if (got[static_cast<size_t>(t)][0].first != r1) {
      if (margin < 0.02) ++top1_soft; else ++top1_hard;
    }
    for (int i = 0; i < k; ++i) {
      const int32_t id = got[static_cast<size_t>(t)][i].first;
      const int32_t* row = ref_ids + static_cast<size_t>(t) * k;
      const int32_t* f = std::find(row, row + k, id);
      if (f == row + k) { ++set_miss; continue; }
      const float rv = ref_vals[static_cast<size_t>(t) * k + (f - row)];
      max_rel = std::max(max_rel, std::fabs(rv - got[static_cast<size_t>(t)][i].second) / (std::fabs(rv) + 1e-30));
    }
  }
  std::printf("[ .. ] logits (kept rows): top-1 hard mismatches %d, near-tie %d of %d rows; top-%d set misses %d of %d; "
              "matched logits within %.3g relative\n",
              top1_hard, top1_soft, kept_rows, k, set_miss, kept_rows * k, max_rel);
  if (top1_hard != 0 || set_miss > kept_rows * k / 4) ok = false;
  // ---- the draft block: its rows over the prompt ------------------------------
  if (dump.tensors.count("mtp_topk_ids") && cfg.mtp_layer() >= 0) {
    GlmDsaModel mtp(cfg, dir, T, std::max(256, T), dgpp::GlmDsaResidency::Streaming, nullptr, 0, 1, 1, /*mtp=*/true);
    const GlmDsaModel::Outputs d = mtp.mtp_forward(tokens);
    const int R = T - 1;
    const Dump::Tensor& mh = dump.tensor("mtp_final_hidden");
    require(mh.dtype == "BF16" && mh.shape.size() == 2 && mh.shape[0] == R && mh.shape[1] == H, "dump: mtp_final_hidden shape");
    // The draft rows take the main stack's hidden (a flipped token's row is
    // excluded as above) and select on their own over the draft cache; the
    // budget is the layer budget doubled for the selections this side
    // cannot certify.
    std::vector<uint16_t> dg, dw;
    for (int t = 0; t < R; ++t) {
      if (excluded[static_cast<size_t>(t)]) continue;
      dg.insert(dg.end(), d.final_hidden_bits.begin() + static_cast<size_t>(t) * H,
                d.final_hidden_bits.begin() + static_cast<size_t>(t + 1) * H);
      const uint16_t* w = reinterpret_cast<const uint16_t*>(mh.data);
      dw.insert(dw.end(), w + static_cast<size_t>(t) * H, w + static_cast<size_t>(t + 1) * H);
    }
    const Stats msn = compare_bf16(dg.data(), dw.data(), dg.size(), 16, 128);
    std::printf("[ .. ] draft final hidden (kept rows): l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", msn.l2,
                msn.max_ulps, msn.soft, msn.hard, msn.total);
    if (msn.total > 0 && (msn.l2 > 0.03 || static_cast<double>(msn.hard) / msn.total > 0.01)) ok = false;
    const Dump::Tensor& mid = dump.tensor("mtp_topk_ids");
    const Dump::Tensor& mlog = dump.tensor("mtp_topk_logits");
    require(mid.dtype == "I32" && mlog.dtype == "F32" && mid.shape[0] == R && mid.shape[1] == k, "dump: mtp topk shape");
    const int32_t* mref_ids = reinterpret_cast<const int32_t*>(mid.data);
    const float* mref_vals = reinterpret_cast<const float*>(mlog.data);
    const auto mgot = GlmDsaModel::topk(d.logits, R, cfg.vocab_size, k);
    int mhard = 0, msoft = 0;
    for (int t = 0; t < R; ++t) {
      if (excluded[static_cast<size_t>(t)]) continue;
      const int32_t r1 = mref_ids[static_cast<size_t>(t) * k];
      const float v1 = mref_vals[static_cast<size_t>(t) * k], v2 = mref_vals[static_cast<size_t>(t) * k + 1];
      const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
      if (mgot[static_cast<size_t>(t)][0].first != r1) {
        if (margin < 0.02) ++msoft; else ++mhard;
      }
    }
    std::printf("[ .. ] draft logits: top-1 hard mismatches %d, near-tie %d of %d rows\n", mhard, msoft, R);
    if (mhard > R / 10) ok = false;
  }
  std::printf("[ %s ] glm_dsa_forward_dump_parity\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, nf4i8_fixture, attn4_fixture, m346_fixture, prefill_fixture, smoke, checkpoint, dump;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--write-fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--write-fixture-nf4i8" && i + 1 < argc) nf4i8_fixture = argv[++i];
    else if (a == "--write-fixture-attn4" && i + 1 < argc) attn4_fixture = argv[++i];
    else if (a == "--write-fixture-m346" && i + 1 < argc) m346_fixture = argv[++i];
    else if (a == "--write-prefill-fixture" && i + 1 < argc)
      prefill_fixture = argv[++i];
    else if (a == "--smoke" && i + 1 < argc) smoke = argv[++i];
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--dump-file" && i + 1 < argc) dump = argv[++i];
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    if (!prefill_fixture.empty()) {
      // A separate fixture audits bulk packed GEMM while every attention
      // row is dense and every expert is selected. This isolates continuous
      // arithmetic from discrete attention/routing boundary changes; the
      // original sparse-selection fixture retains those boundaries.
      const std::string json = glmdsafx::tiny_config_json(5, 1, true, 256, 8);
      const auto tree = dgpp::minijson::parse(json);
      const auto cfg = GlmDsaTextConfig::parse(tree.root);
      glmdsafx::write_fixture(cfg, prefill_fixture, json.c_str());
      std::printf("[ OK ] wrote the bulk-prefill fixture to %s\n", prefill_fixture.c_str());
      return 0;
    }
    if (!fixture.empty()) {
      glmdsafx::write_fixture(glmdsafx::tiny_config(), fixture);
      std::printf("[ OK ] wrote the fixture to %s\n", fixture.c_str());
      return 0;
    }
    if (!attn4_fixture.empty()) {
      // The int4-attention candidate: the attention projections packed at 4
      // bits (the compressed-tensors contract), the shared expert int8.
      glmdsafx::write_fixture(glmdsafx::tiny_config(5, 1, true, false, 4), attn4_fixture, false, 4);
      std::printf("[ OK ] wrote the int4-attention fixture to %s\n", attn4_fixture.c_str());
      return 0;
    }
    if (!m346_fixture.empty()) {
      // The Mixed346 contract's fixture (tests/cuda/glm_dsa_fixture.hpp): every
      // expert form on every packed layer, int8 activation codes.
      glmdsafx::write_fixture(glmdsafx::tiny_config(5, 1, true, false, 8, true), m346_fixture, false, 8, true);
      std::printf("[ OK ] wrote the mixed346 fixture to %s\n", m346_fixture.c_str());
      return 0;
    }
    if (!nf4i8_fixture.empty()) {
      // The NF4I8 contract's fixture (codebook routed experts in the
      // H32-rotated basis, a 512-wide inter): the same gates run on it.
      glmdsafx::write_fixture(glmdsafx::tiny_config(5, 1, true, true), nf4i8_fixture, true);
      std::printf("[ OK ] wrote the nf4i8 fixture to %s\n", nf4i8_fixture.c_str());
      return 0;
    }
    if (!smoke.empty()) return run_smoke(smoke);
    if (!checkpoint.empty() && !dump.empty()) return run_dump_parity(checkpoint, dump);
    std::fprintf(stderr,
                 "usage: --write-fixture DIR | --write-fixture-nf4i8 DIR | --write-fixture-attn4 DIR | --write-fixture-m346 DIR | --write-prefill-fixture DIR | --smoke DIR | "
                 "--checkpoint-dir DIR --dump-file FILE\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
