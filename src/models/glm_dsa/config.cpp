#include "models/glm_dsa/config.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <regex>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("GLM-5.3 config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return static_cast<int>(f.as_int());
}
int optional_int(const minijson::Value& v, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}
double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
double optional_double(const minijson::Value& v, std::string_view field, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  const double d = f->as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string optional_string(const minijson::Value& v, std::string_view field,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}
// A field that must be absent or null (a feature the engine does not
// implement and must not silently ignore).
void require_absent(const minijson::Value& v, std::string_view field, std::string_view why) {
  const minijson::Value* f = v.find(field);
  if (f && !f->is_null()) reject(field, why);
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open config {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

// --- the indexer schedule ----------------------------------------------------

// transformers' rule: layer i is "full" when max(i - offset + 1, 0) % freq == 0.
std::vector<uint8_t> schedule_from_freq(int layers, int freq, int offset) {
  std::vector<uint8_t> full(static_cast<size_t>(layers), 0);
  for (int i = 0; i < layers; ++i)
    full[static_cast<size_t>(i)] = (std::max(i - offset + 1, 0) % freq) == 0;
  return full;
}

void parse_indexer_schedule(const minijson::Value& root, GlmDsaTextConfig& c) {
  const int L = c.num_hidden_layers;
  c.index_topk_freq = optional_int(root, "index_topk_freq", 1);
  c.index_skip_topk_offset = optional_int(root, "index_skip_topk_offset", 2);
  if (c.index_topk_freq < 1) reject("index_topk_freq", "must be >= 1");
  if (c.index_skip_topk_offset < 0) reject("index_skip_topk_offset", "must be >= 0");
  std::vector<uint8_t> from_freq = schedule_from_freq(L, c.index_topk_freq, c.index_skip_topk_offset);
  // An explicit pattern ("FSSF...") overrides the freq/offset schedule.
  bool pattern_given = false;
  std::vector<uint8_t> from_pattern;
  if (const minijson::Value* p = root.find("index_topk_pattern"); p && !p->is_null()) {
    if (!p->is_string()) reject("index_topk_pattern", "not a string");
    const std::string_view s = p->as_string();
    if (static_cast<int>(s.size()) != L) reject("index_topk_pattern", "length != num_hidden_layers");
    for (char ch : s) {
      if (ch != 'F' && ch != 'S') reject("index_topk_pattern", "characters must be F or S");
      from_pattern.push_back(ch == 'F');
    }
    pattern_given = true;
  }
  const std::vector<uint8_t>& derived = pattern_given ? from_pattern : from_freq;
  // The explicit list, when present, must agree with the derived schedule:
  // the two are one fact written twice and the loader trusts neither alone.
  if (const minijson::Value* it = root.find("indexer_types"); it && !it->is_null()) {
    if (!it->is_array()) reject("indexer_types", "not an array");
    if (static_cast<int>(it->items().size()) != L) reject("indexer_types", "length != num_hidden_layers");
    std::vector<uint8_t> listed;
    for (const auto& item : it->items()) {
      if (!item.is_string()) reject("indexer_types", "non-string entry");
      const std::string_view s = item.as_string();
      if (s == "full") listed.push_back(1);
      else if (s == "shared") listed.push_back(0);
      else reject("indexer_types", "entries must be 'full' or 'shared', got '" + std::string(s) + "'");
    }
    if (listed != derived)
      reject("indexer_types",
             pattern_given ? "disagrees with index_topk_pattern"
                           : "disagrees with the index_topk_freq / index_skip_topk_offset schedule");
    c.indexer_full = std::move(listed);
  } else {
    c.indexer_full = derived;
  }
  if (c.indexer_full.empty() || c.indexer_full[0] == 0)
    reject("indexer_types", "layer 0 must be 'full' (a shared layer needs a previous selection)");
}

// --- the quantization contract --------------------------------------------

// One compressed-tensors matching rule: a `re:` regex anchored at the start
// of the module name, or an exact module name.
struct MatchRule {
  bool is_regex = false;
  std::regex re;
  std::string literal;
  bool matches(const std::string& module) const {
    return is_regex ? std::regex_search(module, re) : module == literal;
  }
};

MatchRule make_rule(std::string_view field, const std::string& target) {
  MatchRule r;
  if (target.rfind("re:", 0) == 0) {
    r.is_regex = true;
    try {
      r.re = std::regex("^(?:" + target.substr(3) + ")", std::regex::ECMAScript);
    } catch (const std::regex_error& e) {
      reject(field, "unparseable regex '" + target + "': " + e.what());
    }
  } else {
    if (target == "Linear")
      reject(field, "class-name targets ('Linear') are not implemented; the file must name modules");
    r.literal = target;
  }
  return r;
}

struct QuantGroup {
  std::vector<MatchRule> targets;
  int bits = 0;
};

struct QuantRules {
  std::vector<QuantGroup> groups;
  std::vector<MatchRule> ignore;
  // The bits a module is stored at under these rules: 0 = not quantized.
  int bits_of(const std::string& module) const {
    for (const MatchRule& r : ignore)
      if (r.matches(module)) return 0;
    int bits = 0;
    for (const QuantGroup& g : groups)
      for (const MatchRule& r : g.targets)
        if (r.matches(module)) {
          if (bits != 0 && bits != g.bits)
            reject("quantization_config.config_groups", "module '" + module + "' matches two groups");
          bits = g.bits;
        }
    return bits;
  }
};

const char* kAttentionModules[] = {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"};
const char* kMlpModules[] = {"gate_proj", "up_proj", "down_proj"};

// The compressed-tensors branch (quant_method compressed-tensors): the
// int4/int8 pack-quantized checkpoint.
QuantRules parse_quantization_compressed_tensors(const minijson::Value* qc, GlmDsaTextConfig& c) {
  if (const std::string f = optional_string(*qc, "format", ""); f != "pack-quantized")
    reject("quantization_config.format", "only pack-quantized is implemented, got '" + f + "'");
  if (const std::string s = optional_string(*qc, "quantization_status", "compressed"); s != "compressed")
    reject("quantization_config.quantization_status", "must be compressed, got '" + s + "'");
  require_absent(*qc, "kv_cache_scheme", "a KV cache scheme is not implemented (the engine picks its own latent format)");
  // A transform (a Hadamard rotation of the weights) would change what the
  // packed codes mean; an empty object is the only accepted value.
  if (const minijson::Value* t = qc->find("transform_config"); t && !t->is_null()) {
    if (!t->is_object() || !t->members().empty())
      reject("quantization_config.transform_config", "weight transforms are not implemented");
  }
  if (const minijson::Value* s = qc->find("sparsity_config"); s && !s->is_null()) {
    if (!s->is_object() || !s->members().empty())
      reject("quantization_config.sparsity_config", "sparsity is not implemented");
  }
  const minijson::Value* groups = qc->find("config_groups");
  if (!groups || !groups->is_object() || groups->members().empty())
    reject("quantization_config.config_groups", "missing");
  QuantRules rules;
  int group_size = 0;
  for (const auto& m : groups->members()) {
    const std::string gf = "quantization_config.config_groups." + m.key;
    const minijson::Value& g = m.value;
    if (!g.is_object()) reject(gf, "group is not an object");
    if (const std::string f = optional_string(g, "format", "pack-quantized"); f != "pack-quantized")
      reject(gf + ".format", "only pack-quantized is implemented, got '" + f + "'");
    require_absent(g, "input_activations", "activation quantization is not implemented (W4A16 / W8A16 only)");
    require_absent(g, "output_activations", "activation quantization is not implemented (W4A16 / W8A16 only)");
    const minijson::Value* w = g.find("weights");
    if (!w || !w->is_object()) reject(gf + ".weights", "missing");
    QuantGroup qg;
    qg.bits = require_int(*w, "num_bits");
    if (qg.bits != 4 && qg.bits != 8) reject(gf + ".weights.num_bits", "the packed-int cores implement 4 and 8");
    if (const std::string t = optional_string(*w, "type", "int"); t != "int")
      reject(gf + ".weights.type", "must be int, got '" + t + "'");
    if (!optional_bool(*w, "symmetric", true)) reject(gf + ".weights.symmetric", "asymmetric (zero-point) weights are not implemented");
    if (const std::string s = optional_string(*w, "strategy", "group"); s != "group")
      reject(gf + ".weights.strategy", "only the group strategy is implemented, got '" + s + "'");
    if (optional_bool(*w, "dynamic", false)) reject(gf + ".weights.dynamic", "dynamic weight quantization is not a checkpoint format");
    require_absent(*w, "zp_dtype", "zero points are not implemented (symmetric only)");
    require_absent(*w, "actorder", "activation ordering is not implemented");
    require_absent(*w, "block_structure", "block structures are not implemented");
    const int gs = require_int(*w, "group_size");
    if (gs != 64) reject(gf + ".weights.group_size", "the packed-int cores implement 64, got " + std::to_string(gs));
    if (group_size != 0 && gs != group_size) reject(gf + ".weights.group_size", "differs between groups");
    group_size = gs;
    const minijson::Value* t = g.find("targets");
    if (!t || !t->is_array() || t->items().empty()) reject(gf + ".targets", "missing");
    for (const auto& item : t->items()) {
      if (!item.is_string()) reject(gf + ".targets", "non-string entry");
      qg.targets.push_back(make_rule(gf + ".targets", std::string(item.as_string())));
    }
    rules.groups.push_back(std::move(qg));
  }
  c.packed_group_size = group_size;
  if (const minijson::Value* ig = qc->find("ignore"); ig && !ig->is_null()) {
    if (!ig->is_array()) reject("quantization_config.ignore", "not an array");
    for (const auto& item : ig->items()) {
      if (!item.is_string()) reject("quantization_config.ignore", "non-string entry");
      rules.ignore.push_back(make_rule("quantization_config.ignore", std::string(item.as_string())));
    }
  }
  return rules;
}

// --- the NF4I8 contract (quant_method dgpp_nf4i8, 2026-10-08) ---------------
// HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128 (its FORMAT.md and config.json):
// `retained_int8` names the int8 g64 attention / shared-expert triples
// kept from the baseline on an inclusive layer range; `routed_experts`
// the 4-bit codebook indices (bf16 scales per 128) of the routed experts
// on the same range, with the input rotation they were quantized under.
// The branch derives the same per-module rules the compressed-tensors
// branch does (so the shape check below is shared) and records the routed
// format; every machine-meaningful field must equal what the kernels
// implement — the prose fields (`packing`, `modules`, `other_weights`,
// `arithmetic`) are not read.
std::pair<int, int> layer_range(const minijson::Value& v, const std::string& field, int layers) {
  const minijson::Value* l = v.find("layers");
  if (!l || !l->is_array() || l->items().size() != 2) reject(field, "must be [first, last] (inclusive)");
  const auto& it = l->items();
  if (!it[0].is_number() || !it[1].is_number()) reject(field, "non-numeric bound");
  const int lo = static_cast<int>(it[0].as_int()), hi = static_cast<int>(it[1].as_int());
  if (lo < 0 || hi < lo || hi >= layers) reject(field, "outside [0, num_hidden_layers)");
  return {lo, hi};
}

std::string layer_alternation(int lo, int hi) {
  std::string s = "(?:";
  for (int l = lo; l <= hi; ++l) {
    if (l != lo) s += "|";
    s += std::to_string(l);
  }
  return s + ")";
}

QuantRules parse_quantization_nf4i8(const minijson::Value* qc, GlmDsaTextConfig& c) {
  const std::string f = "quantization_config";
  if (const std::string s = optional_string(*qc, "format", ""); s != "mixed-codebook-packed")
    reject(f + ".format", "dgpp_nf4i8 is the mixed-codebook-packed format, got '" + s + "'");
  if (const int v = optional_int(*qc, "format_version", 0); v != 1)
    reject(f + ".format_version", "the loader implements version 1, got " + std::to_string(v));
  require_absent(*qc, "kv_cache_scheme", "a KV cache scheme is not implemented (the engine picks its own latent format)");
  // The retained int8 triples: the baseline's attention and shared expert.
  const minijson::Value* r8 = qc->find("retained_int8");
  if (!r8 || !r8->is_object()) reject(f + ".retained_int8", "missing");
  if (const int g = require_int(*r8, "group_size"); g != 64)
    reject(f + ".retained_int8.group_size", "the int8 attention / shared triples are group 64, got " + std::to_string(g));
  if (const std::string s = optional_string(*r8, "scale_dtype", "bfloat16"); s != "bfloat16")
    reject(f + ".retained_int8.scale_dtype", "must be bfloat16, got '" + s + "'");
  const auto [lo8, hi8] = layer_range(*r8, f + ".retained_int8.layers", c.num_hidden_layers);
  // The routed experts: the codebook, the group and the rotation.
  const minijson::Value* re = qc->find("routed_experts");
  if (!re || !re->is_object()) reject(f + ".routed_experts", "missing");
  const std::string rf = f + ".routed_experts";
  if (const int b = require_int(*re, "bits"); b != 4)
    reject(rf + ".bits", "the codebook core implements 4-bit indices, got " + std::to_string(b));
  if (const int g = require_int(*re, "group_size"); g != 128)
    reject(rf + ".group_size", "the codebook core implements group 128, got " + std::to_string(g));
  if (const std::string s = optional_string(*re, "indices_dtype", "int32"); s != "int32")
    reject(rf + ".indices_dtype", "must be int32, got '" + s + "'");
  if (const std::string s = optional_string(*re, "indices_key", "weight_indices"); s != "weight_indices")
    reject(rf + ".indices_key", "must be weight_indices, got '" + s + "'");
  if (const std::string s = optional_string(*re, "scale_dtype", "bfloat16"); s != "bfloat16")
    reject(rf + ".scale_dtype", "must be bfloat16, got '" + s + "'");
  {
    const minijson::Value* cb = re->find("codebook");
    if (!cb || !cb->is_array() || cb->items().size() != 16) reject(rf + ".codebook", "must list 16 levels");
    int i = 0;
    for (const auto& item : cb->items()) {
      if (!item.is_number() || item.as_int() != kNf4i8Codebook[i])
        reject(rf + ".codebook", "level " + std::to_string(i) + " differs from the compiled NF4I8 codebook");
      ++i;
    }
  }
  const auto [loE, hiE] = layer_range(*re, rf + ".layers", c.num_hidden_layers);
  if (loE != lo8 || hiE != hi8)
    reject(rf + ".layers", "must equal retained_int8.layers (the loader implements one packed range)");
  {
    const minijson::Value* t = re->find("input_transform");
    if (!t || !t->is_object()) reject(rf + ".input_transform", "missing (the H32 rotation the experts were quantized under)");
    const std::string tf = rf + ".input_transform";
    if (const std::string s = optional_string(*t, "type", ""); s != "normalized_hadamard")
      reject(tf + ".type", "the engine implements normalized_hadamard, got '" + s + "'");
    if (const int b = require_int(*t, "block_size"); b != 32)
      reject(tf + ".block_size", "the engine implements 32-wide blocks, got " + std::to_string(b));
    if (const std::string s = optional_string(*t, "axis", "input_channels"); s != "input_channels")
      reject(tf + ".axis", "must be input_channels, got '" + s + "'");
    const minijson::Value* a = t->find("apply_to");
    if (!a || !a->is_array()) reject(tf + ".apply_to", "missing");
    bool seen[3] = {false, false, false};
    for (const auto& item : a->items()) {
      if (!item.is_string()) reject(tf + ".apply_to", "non-string entry");
      const std::string_view s = item.as_string();
      int which = -1;
      for (int m = 0; m < 3; ++m)
        if (s == kMlpModules[m]) which = m;
      if (which < 0) reject(tf + ".apply_to", "unknown projection '" + std::string(s) + "'");
      seen[which] = true;
    }
    if (!(seen[0] && seen[1] && seen[2]))
      reject(tf + ".apply_to", "the rotation applies to gate_proj, up_proj and down_proj");
  }
  if (c.hidden_size % 128 != 0)
    reject("hidden_size", "must be a multiple of 128 (the codebook experts' scale group)");
  if (c.moe_intermediate_size % 128 != 0)
    reject("moe_intermediate_size", "must be a multiple of 128 (the codebook experts' scale group)");
  QuantRules rules;
  const std::string layers = layer_alternation(lo8, hi8);
  QuantGroup g8;
  g8.bits = 8;
  g8.targets.push_back(make_rule(f + ".retained_int8",
      "re:model\\.layers\\." + layers +
      "\\.(?:self_attn\\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)|"
      "mlp\\.shared_experts\\.(?:gate_proj|up_proj|down_proj))$"));
  QuantGroup g4;
  g4.bits = 4;
  g4.targets.push_back(make_rule(rf, "re:model\\.layers\\." + layers +
                                         "\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"));
  rules.groups.push_back(std::move(g8));
  rules.groups.push_back(std::move(g4));
  c.packed_group_size = 64;
  c.expert_scale_fmt = kPackedScaleBf16G128Nf4i8;
  c.expert_input_hadamard32 = true;
  return rules;
}

// --- the Mixed346 contract (quant_method dgpp_mixed346, 2026-10-09) ----------
//
// HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128 (its FORMAT.md): config.json's
// quantization_config names the format, the group, the rotation block, the
// activation width and two sidecar files. The baseline block
// (baseline-quantization-config.json: the int4/int8 release's
// compressed-tensors rules) describes every tensor the checkpoint kept —
// the attention, shared-expert and dense/draft tensors and the "existing"
// experts — and is parsed as that contract. The recipe
// (quantization-recipe.json) lists every routed expert of every packed
// layer as three digits in {3, 4, 6} (gate, up, down; gate == up) or
// "existing", with the three codebooks and the activation policy; each is
// checked against what the kernels compile. The engine also accepts both
// blocks inline (quantization_config.baseline / .recipe objects) when no
// checkpoint directory is at hand — the fixtures' form.
constexpr const char* kMixed346Format = "dgpp_mixed346_h32_a8_g128_v1";

int recipe_key_int(const minijson::Member& m, const std::string& field, int limit) {
  if (m.key.empty()) reject(field, "empty key");
  int v = 0;
  for (char ch : m.key) {
    if (ch < '0' || ch > '9') reject(field, "key '" + m.key + "' is not an index");
    v = v * 10 + (ch - '0');
    if (v >= limit) reject(field, "key '" + m.key + "' is out of range");
  }
  return v;
}

void check_recipe_string(const minijson::Value& v, const std::string& prefix, const char* key, const char* want) {
  if (const std::string s = optional_string(v, key, ""); s != want)
    reject(prefix + "." + key, "must be '" + std::string(want) + "', got '" + s + "'");
}

void parse_mixed346_recipe(const minijson::Value& r, GlmDsaTextConfig& c, const std::string& f) {
  if (!r.is_object()) reject(f, "not an object");
  check_recipe_string(r, f, "format", kMixed346Format);
  if (const int v = optional_int(r, "version", 0); v != 1) reject(f + ".version", "the loader implements version 1, got " + std::to_string(v));
  if (const int g = require_int(r, "group_size"); g != 128) reject(f + ".group_size", "the Mixed346 core implements group 128, got " + std::to_string(g));
  if (const int b = require_int(r, "rotation_size"); b != 32) reject(f + ".rotation_size", "the engine implements 32-wide rotation blocks, got " + std::to_string(b));
  {
    const minijson::Value* po = r.find("projection_order");
    if (!po || !po->is_array() || po->items().size() != 3) reject(f + ".projection_order", "must list gate_proj, up_proj, down_proj");
    int i = 0;
    for (const auto& item : po->items()) {
      if (!item.is_string() || item.as_string() != kMlpModules[i]) reject(f + ".projection_order", "must be gate_proj, up_proj, down_proj in that order");
      ++i;
    }
  }
  {
    const minijson::Value* a = r.find("activation");
    if (!a || !a->is_object()) reject(f + ".activation", "missing");
    const std::string af = f + ".activation";
    check_recipe_string(*a, af, "format", "int8");
    if (const int g = require_int(*a, "group_size"); g != 128) reject(af + ".group_size", "the engine quantizes 128 values per scale, got " + std::to_string(g));
    check_recipe_string(*a, af, "rounding", "nearest ties-to-even");
    check_recipe_string(*a, af, "scale", "max(amax/127,1e-30)");
    check_recipe_string(*a, af, "scale_dtype", "float32");
    check_recipe_string(*a, af, "rotation", "H32 then BF16 rounding");
    const minijson::Value* cl = a->find("clamp");
    if (!cl || !cl->is_array() || cl->items().size() != 2 || !cl->items()[0].is_number() || !cl->items()[1].is_number() ||
        cl->items()[0].as_int() != -128 || cl->items()[1].as_int() != 127)
      reject(af + ".clamp", "must be [-128, 127]");
  }
  {
    const minijson::Value* cb = r.find("weight_codebooks");
    if (!cb || !cb->is_object()) reject(f + ".weight_codebooks", "missing");
    for (const char* w : {"3", "4", "6"}) {
      const minijson::Value* t = cb->find(w);
      const int bits = w[0] - '0';
      const size_t n = static_cast<size_t>(1) << bits;
      const std::string cf = f + ".weight_codebooks." + w;
      if (!t || !t->is_array() || t->items().size() != n) reject(cf, "must list " + std::to_string(n) + " levels");
      size_t i = 0;
      for (const auto& item : t->items()) {
        const int want = packed_code_level(static_cast<unsigned>(i), bits, kPackedScaleBf16G128Mixed346);
        if (!item.is_number() || item.as_double() != static_cast<double>(want))
          reject(cf, "level " + std::to_string(i) + " differs from the compiled codebook (" + std::to_string(want) + ")");
        ++i;
      }
    }
  }
  if (const minijson::Value* sl = r.find("scale_limits"); sl && !sl->is_null()) {
    if (!sl->is_object()) reject(f + ".scale_limits", "not an object");
    for (const char* w : {"3", "4", "6"}) {
      const int bits = w[0] - '0';
      const double want = bits == 3 ? 127.0 / 0.75 : (bits == 4 ? 127.0 : 31.0);
      const minijson::Value* t = sl->find(w);
      if (!t || !t->is_number() || std::fabs(t->as_double() - want) > 1e-9 * want)
        reject(f + ".scale_limits." + w, "differs from the format's (" + std::to_string(want) + ")");
    }
  }
  const int L = c.num_hidden_layers, E = c.n_routed_experts;
  const minijson::Value* lr = r.find("layer_expert_recipes");
  if (!lr || !lr->is_object() || lr->members().empty()) reject(f + ".layer_expert_recipes", "missing");
  c.expert_recipe.assign(static_cast<size_t>(L) * static_cast<size_t>(E), 0);
  std::vector<std::pair<std::string, int>> counts;
  auto count = [&](const std::string& k) {
    for (auto& [name, n] : counts)
      if (name == k) {
        ++n;
        return;
      }
    counts.emplace_back(k, 1);
  };
  for (const auto& lm : lr->members()) {
    const std::string lf = f + ".layer_expert_recipes." + lm.key;
    const int l = recipe_key_int(lm, lf, L);
    if (!c.is_moe_layer(l)) reject(lf, "layer " + std::to_string(l) + " carries no routed experts");
    if (!lm.value.is_object() || lm.value.members().size() != static_cast<size_t>(E))
      reject(lf, "must list every one of the " + std::to_string(E) + " routed experts");
    for (const auto& em : lm.value.members()) {
      const std::string ef = lf + "." + em.key;
      const int e = recipe_key_int(em, ef, E);
      if (!em.value.is_string()) reject(ef, "not a string");
      const std::string v(em.value.as_string());
      uint8_t code;
      if (v == "existing") {
        code = GlmDsaTextConfig::kExpertRecipeExisting;
      } else {
        if (v.size() != 3) reject(ef, "'" + v + "' is not three digits or 'existing'");
        for (char ch : v)
          if (ch != '3' && ch != '4' && ch != '6') reject(ef, "'" + v + "': the widths are 3, 4 and 6");
        if (v[0] != v[1]) reject(ef, "'" + v + "': gate and up must share one width (the kernels stage one input form)");
        code = static_cast<uint8_t>(((v[0] - '0') << 4) | (v[2] - '0'));
      }
      uint8_t& slot = c.expert_recipe[static_cast<size_t>(l) * static_cast<size_t>(E) + static_cast<size_t>(e)];
      if (slot != 0) reject(ef, "listed twice");
      slot = code;
      count(v);
    }
  }
  if (const minijson::Value* rc = r.find("recipe_counts"); rc && !rc->is_null()) {
    if (!rc->is_object()) reject(f + ".recipe_counts", "not an object");
    size_t listed = 0;
    for (const auto& m : rc->members()) {
      if (!m.value.is_number()) reject(f + ".recipe_counts." + m.key, "not a number");
      int have = 0;
      for (const auto& [name, n] : counts)
        if (name == m.key) have = n;
      if (have != static_cast<int>(m.value.as_int()))
        reject(f + ".recipe_counts." + m.key, "says " + std::to_string(m.value.as_int()) + ", the table lists " + std::to_string(have));
      ++listed;
    }
    if (listed != counts.size()) reject(f + ".recipe_counts", "does not name every form the table uses");
  }
}

QuantRules parse_quantization_mixed346(const minijson::Value* qc, GlmDsaTextConfig& c, const std::string& dir) {
  const std::string f = "quantization_config";
  if (const std::string s = optional_string(*qc, "format", ""); s != kMixed346Format)
    reject(f + ".format", "dgpp_mixed346 is the " + std::string(kMixed346Format) + " format, got '" + s + "'");
  if (const int v = optional_int(*qc, "version", 0); v != 1)
    reject(f + ".version", "the loader implements version 1, got " + std::to_string(v));
  if (const int g = require_int(*qc, "group_size"); g != 128)
    reject(f + ".group_size", "the Mixed346 core implements group 128, got " + std::to_string(g));
  if (const int b = require_int(*qc, "rotation_size"); b != 32)
    reject(f + ".rotation_size", "the engine implements 32-wide rotation blocks, got " + std::to_string(b));
  if (const int b = require_int(*qc, "activation_bits"); b != 8)
    reject(f + ".activation_bits", "the Mixed346 core implements int8 activation codes, got " + std::to_string(b));
  require_absent(*qc, "kv_cache_scheme", "a KV cache scheme is not implemented (the engine picks its own latent format)");
  // (minijson references the text it parses: each sidecar's text outlives
  // its values.)
  auto sidecar = [&](const char* key, const char* dflt) {
    const std::string name = optional_string(*qc, key, dflt);
    if (dir.empty())
      reject(f + "." + key, "the sidecar '" + name + "' needs the checkpoint directory (parse from the file, or inline the block)");
    return read_file(dir + "/" + name);
  };
  QuantRules rules;
  if (const minijson::Value* inl = qc->find("baseline"); inl && !inl->is_null()) {
    if (!inl->is_object()) reject(f + ".baseline", "not an object");
    if (optional_string(*inl, "quant_method", "") != "compressed-tensors")
      reject(f + ".baseline.quant_method", "the baseline block is the compressed-tensors contract");
    rules = parse_quantization_compressed_tensors(inl, c);
  } else {
    const std::string text = sidecar("baseline_config_file", "baseline-quantization-config.json");
    const auto parsed = minijson::parse(text);
    if (!parsed.root.is_object()) reject(f + ".baseline_config_file", "root is not an object");
    if (optional_string(parsed.root, "quant_method", "") != "compressed-tensors")
      reject(f + ".baseline_config_file", "the baseline block is the compressed-tensors contract");
    rules = parse_quantization_compressed_tensors(&parsed.root, c);
  }
  if (const minijson::Value* inl = qc->find("recipe"); inl && !inl->is_null()) {
    parse_mixed346_recipe(*inl, c, f + ".recipe");
  } else {
    const std::string text = sidecar("recipe_file", "quantization-recipe.json");
    const auto parsed = minijson::parse(text);
    parse_mixed346_recipe(parsed.root, c, optional_string(*qc, "recipe_file", "quantization-recipe.json"));
  }
  if (c.hidden_size % 128 != 0)
    reject("hidden_size", "must be a multiple of 128 (the Mixed346 experts' scale group)");
  if (c.moe_intermediate_size % 128 != 0)
    reject("moe_intermediate_size", "must be a multiple of 128 (the Mixed346 experts' scale group)");
  c.expert_scale_fmt = kPackedScaleBf16G128Mixed346;
  c.expert_input_hadamard32 = true;
  c.expert_activation_bits = 8;
  return rules;
}

// The recipe against the packed range the baseline derived: every packed
// layer's every expert listed, no unpacked layer listed.
void check_mixed346_coverage(const GlmDsaTextConfig& c) {
  const int L = c.num_hidden_layers, E = c.n_routed_experts;
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) {
      const uint8_t r = c.expert_recipe[static_cast<size_t>(l) * static_cast<size_t>(E) + static_cast<size_t>(e)];
      if (c.packed_layer(l) && r == 0)
        reject("quantization_config.recipe", "layer " + std::to_string(l) + " expert " + std::to_string(e) + " is not in the recipe");
      if (!c.packed_layer(l) && r != 0)
        reject("quantization_config.recipe", "layer " + std::to_string(l) + " is listed but not packed under the baseline rules");
    }
}

QuantRules parse_quantization(const minijson::Value& root, GlmDsaTextConfig& c, const std::string& dir) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the compressed-tensors pack-quantized "
           "int4/int8 checkpoint, the dgpp_nf4i8 codebook checkpoint and the "
           "dgpp_mixed346 checkpoint; a BF16 GlmMoeDsa checkpoint has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  const std::string m = optional_string(*qc, "quant_method", "");
  if (m == "compressed-tensors") return parse_quantization_compressed_tensors(qc, c);
  if (m == "dgpp_nf4i8") return parse_quantization_nf4i8(qc, c);
  if (m == "dgpp_mixed346") return parse_quantization_mixed346(qc, c, dir);
  reject("quantization_config.quant_method",
         "only compressed-tensors, dgpp_nf4i8 and dgpp_mixed346 are implemented, got '" + m + "'");
}

// Applies the file's rules to the module names the table will emit and
// checks they describe the one shape the loader implements: a contiguous
// range of MoE main layers whose attention (all five projections at one
// width), shared expert and routed experts are packed, every other module
// BF16.
void derive_packed_shape(const QuantRules& rules, GlmDsaTextConfig& c) {
  const int L = c.num_hidden_layers;
  const int E = c.n_routed_experts;
  auto layer_prefix = [](int l) { return "model.layers." + std::to_string(l) + "."; };
  auto expect_plain = [&](const std::string& module) {
    if (rules.bits_of(module) != 0)
      reject("quantization_config", "'" + module + "' is quantized, and the loader expects BF16 there");
  };
  // Globals and the draft are BF16.
  expect_plain("lm_head");
  expect_plain("model.embed_tokens");
  expect_plain("model.norm");
  const int sample_experts[] = {0, 1, E / 2, E - 1};
  int begin = -1, end = -1;
  for (int l = 0; l < L; ++l) {
    const std::string p = layer_prefix(l);
    int attn = -1;
    for (const char* m : kAttentionModules) {
      const int b = rules.bits_of(p + "self_attn." + m);
      if (attn == -1) attn = b;
      else if (attn != b)
        reject("quantization_config", "layer " + std::to_string(l) + ": the five attention projections must share one width");
    }
    expect_plain(p + "input_layernorm");
    expect_plain(p + "post_attention_layernorm");
    expect_plain(p + "self_attn.q_a_layernorm");
    expect_plain(p + "self_attn.kv_a_layernorm");
    if (c.owns_indexer(l))
      for (const char* m : {"wq_b", "wk", "weights_proj", "k_norm"}) expect_plain(p + "self_attn.indexer." + m);
    int shared = -1, routed = -1;
    if (c.is_moe_layer(l)) {
      expect_plain(p + "mlp.gate");
      for (const char* m : kMlpModules) {
        const int b = rules.bits_of(p + "mlp.shared_experts." + m);
        if (shared == -1) shared = b;
        else if (shared != b) reject("quantization_config", "layer " + std::to_string(l) + ": the shared expert's projections must share one width");
      }
      for (int e : sample_experts)
        for (const char* m : kMlpModules) {
          const int b = rules.bits_of(p + "mlp.experts." + std::to_string(e) + "." + m);
          if (routed == -1) routed = b;
          else if (routed != b) reject("quantization_config", "layer " + std::to_string(l) + ": the routed experts' projections must share one width");
        }
    } else {
      for (const char* m : kMlpModules) expect_plain(p + "mlp." + m);
      if (attn != 0)
        reject("quantization_config", "layer " + std::to_string(l) + ": a packed attention on a dense layer is not implemented");
    }
    const bool packed = attn != 0;
    if (packed) {
      if (shared == 0 || routed == 0)
        reject("quantization_config", "layer " + std::to_string(l) + ": packed attention but BF16 experts is not implemented");
      if (begin == -1) begin = l;
      else if (end != -1) reject("quantization_config", "the packed layers must be one contiguous range");
      if (c.attention_bits == 0 || begin == l) {
        c.attention_bits = attn;
        c.shared_bits = shared;
        c.expert_bits = routed;
      } else if (attn != c.attention_bits || shared != c.shared_bits || routed != c.expert_bits) {
        reject("quantization_config", "layer " + std::to_string(l) + ": widths differ from the first packed layer's");
      }
    } else {
      if (shared > 0 || routed > 0)
        reject("quantization_config", "layer " + std::to_string(l) + ": packed experts under BF16 attention is not implemented");
      if (begin != -1 && end == -1) end = l;
    }
  }
  if (begin == -1) reject("quantization_config", "no layer is packed — a BF16 expert path is not implemented");
  if (end == -1) end = L;
  c.packed_layer_begin = begin;
  c.packed_layer_end = end;
  if (c.mtp_layer() >= 0) {
    const std::string p = layer_prefix(c.mtp_layer());
    for (const char* m : kAttentionModules) expect_plain(p + "self_attn." + m);
    for (const char* m : kMlpModules) expect_plain(p + "mlp.shared_experts." + m);
    for (int e : sample_experts)
      for (const char* m : kMlpModules) expect_plain(p + "mlp.experts." + std::to_string(e) + "." + m);
    for (const char* m : {"enorm", "hnorm", "eh_proj", "shared_head.norm", "mlp.gate"}) expect_plain(p + m);
  }
}

}  // namespace

GlmDsaTextConfig GlmDsaTextConfig::parse(const minijson::Value& root, const std::string& dir) {
  if (!root.is_object()) reject("", "root is not an object");
  GlmDsaTextConfig c;
  const std::string model_type = optional_string(root, "model_type", "glm_moe_dsa");
  if (model_type != "glm_moe_dsa") reject("model_type", "expected glm_moe_dsa, got " + model_type);
  if (root.find("text_config"))
    reject("text_config", "a nested text_config is the GLM-5.3-Flash layout, not glm_moe_dsa's");

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(root, "rms_norm_eps"));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.hidden_act = optional_string(root, "hidden_act", "silu");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  c.first_k_dense_replace = optional_int(root, "first_k_dense_replace", 0);
  if (c.hidden_size <= 0 || c.hidden_size % 64 != 0)
    reject("hidden_size", "must be a positive multiple of 64 (the packed-int cores' group)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (c.first_k_dense_replace < 0 || c.first_k_dense_replace > c.num_hidden_layers)
    reject("first_k_dense_replace", "outside [0, num_hidden_layers]");
  if (optional_int(root, "moe_layer_freq", 1) != 1) reject("moe_layer_freq", "only 1 is implemented");
  if (const minijson::Value* lt = root.find("layer_types"); lt && !lt->is_null()) {
    if (!lt->is_array() || static_cast<int>(lt->items().size()) != c.num_hidden_layers)
      reject("layer_types", "must list every layer");
    for (const auto& item : lt->items())
      if (!item.is_string() || item.as_string() != "deepseek_sparse_attention")
        reject("layer_types", "every layer is deepseek_sparse_attention in this family");
  }
  if (const minijson::Value* mt = root.find("mlp_layer_types"); mt && !mt->is_null()) {
    if (!mt->is_array() || static_cast<int>(mt->items().size()) != c.num_hidden_layers)
      reject("mlp_layer_types", "must list every layer");
    int i = 0;
    for (const auto& item : mt->items()) {
      const std::string_view s = item.is_string() ? item.as_string() : std::string_view{};
      const std::string_view want = i < c.first_k_dense_replace ? "dense" : "sparse";
      if (s != want)
        reject("mlp_layer_types", "layer " + std::to_string(i) + " must be '" + std::string(want) +
                                      "' (first_k_dense_replace)");
      ++i;
    }
  }

  // --- tokens ---------------------------------------------------------------
  if (const minijson::Value* eos = root.find("eos_token_id"); eos && !eos->is_null()) {
    if (eos->is_array()) {
      for (const auto& item : eos->items()) {
        if (!item.is_number()) reject("eos_token_id", "non-numeric element");
        c.eos_token_ids.push_back(item.as_int());
      }
    } else if (eos->is_number()) {
      c.eos_token_ids.push_back(eos->as_int());
    } else {
      reject("eos_token_id", "not a number or array");
    }
    for (int64_t id : c.eos_token_ids)
      if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, vocab_size)");
  }
  if (c.eos_token_ids.empty()) reject("eos_token_id", "missing");
  c.pad_token_id = optional_int(root, "pad_token_id", -1);

  // --- MLA attention ---------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  if (const int kv = optional_int(root, "num_key_value_heads", c.num_attention_heads); kv != c.num_attention_heads)
    reject("num_key_value_heads", "MLA has one latent per token: num_key_value_heads must equal num_attention_heads");
  c.q_lora_rank = require_int(root, "q_lora_rank");
  c.kv_lora_rank = require_int(root, "kv_lora_rank");
  c.qk_nope_head_dim = require_int(root, "qk_nope_head_dim");
  c.qk_rope_head_dim = require_int(root, "qk_rope_head_dim");
  c.v_head_dim = require_int(root, "v_head_dim");
  c.attention_bias = optional_bool(root, "attention_bias", false);
  c.rope_interleave = optional_bool(root, "rope_interleave", true);
  if (c.num_attention_heads <= 0) reject("num_attention_heads", "must be positive");
  if (c.q_lora_rank <= 0 || c.q_lora_rank % 64 != 0)
    reject("q_lora_rank", "must be a positive multiple of 64 (a packed q_b input)");
  if (c.kv_lora_rank <= 0 || c.kv_lora_rank % 64 != 0)
    reject("kv_lora_rank", "must be a positive multiple of 64 (a packed kv_b input)");
  if (c.qk_nope_head_dim <= 0) reject("qk_nope_head_dim", "must be positive");
  if (c.qk_rope_head_dim != 0 && c.qk_rope_head_dim != 64)
    reject("qk_rope_head_dim", "the MLA kernels implement a 0- or 64-wide rope key");
  if (c.v_head_dim <= 0) reject("v_head_dim", "must be positive");
  if (const int qk = optional_int(root, "qk_head_dim", c.qk_head_dim()); qk != c.qk_head_dim())
    reject("qk_head_dim", "must equal qk_nope_head_dim + qk_rope_head_dim");
  if (c.attention_bias) reject("attention_bias", "biased MLA projections are not implemented");
  if (!c.rope_interleave) reject("rope_interleave", "only the interleaved (DeepSeek) RoPE is implemented");
  c.rope_theta = optional_double(root, "rope_theta", 10000.0);
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  if (const minijson::Value* rp = root.find("rope_parameters"); rp && !rp->is_null()) {
    if (!rp->is_object()) reject("rope_parameters", "not an object");
    if (optional_string(*rp, "rope_type", "default") != "default")
      reject("rope_parameters.rope_type", "only the default rope is implemented");
    c.rope_theta = optional_double(*rp, "rope_theta", c.rope_theta);
  }
  if (!(c.rope_theta > 0)) reject("rope_theta", "must be positive");

  // --- DSA indexer ------------------------------------------------------------
  c.index_n_heads = require_int(root, "index_n_heads");
  c.index_head_dim = require_int(root, "index_head_dim");
  c.index_topk = require_int(root, "index_topk");
  c.indexer_rope_interleave = optional_bool(root, "indexer_rope_interleave", true);
  c.index_share_for_mtp_iteration = optional_bool(root, "index_share_for_mtp_iteration", false);
  if (c.index_n_heads <= 0) reject("index_n_heads", "must be positive");
  if (c.index_head_dim != 128) reject("index_head_dim", "the indexer implements 128 (Hadamard-128 is pinned)");
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0)
    reject("index_topk", "must be a power of two (the select networks)");
  if (c.qk_rope_head_dim > c.index_head_dim) reject("qk_rope_head_dim", "exceeds index_head_dim");
  if (!c.indexer_rope_interleave) reject("indexer_rope_interleave", "only the interleaved indexer RoPE is implemented");
  require_absent(root, "index_kpool", "pooled indexing is the GLM-5.3-Flash layout; this family selects per token");
  require_absent(root, "index_kpool_compress", "pooled indexing is the GLM-5.3-Flash layout");
  require_absent(root, "index_kpool_always_select_tail", "pooled indexing is the GLM-5.3-Flash layout");
  parse_indexer_schedule(root, c);

  // --- MLP / MoE --------------------------------------------------------------
  c.intermediate_size = require_int(root, "intermediate_size");
  c.moe_intermediate_size = require_int(root, "moe_intermediate_size");
  c.n_routed_experts = require_int(root, "n_routed_experts");
  c.n_shared_experts = optional_int(root, "n_shared_experts", 0);
  c.num_experts_per_tok = require_int(root, "num_experts_per_tok");
  c.norm_topk_prob = optional_bool(root, "norm_topk_prob", true);
  c.routed_scaling_factor = static_cast<float>(optional_double(root, "routed_scaling_factor", 1.0));
  c.n_group = optional_int(root, "n_group", 1);
  c.topk_group = optional_int(root, "topk_group", 1);
  if (c.intermediate_size <= 0 || c.intermediate_size % 64 != 0)
    reject("intermediate_size", "must be a positive multiple of 64");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 64 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 64 (a packed down input)");
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096)
    reject("n_routed_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 ||
      c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, min(n_routed_experts, 16)]");
  if (c.n_shared_experts != 1)
    reject("n_shared_experts", "the MoE chain implements exactly one shared expert");
  if (c.n_group != 1 || c.topk_group != 1)
    reject("n_group", "group-limited routing (n_group/topk_group != 1) is not implemented");
  if (!(c.routed_scaling_factor > 0)) reject("routed_scaling_factor", "must be positive");
  if (const std::string sf = optional_string(root, "scoring_func", "sigmoid"); sf != "sigmoid")
    reject("scoring_func", "only sigmoid is implemented");
  if (const std::string tm = optional_string(root, "topk_method", "noaux_tc"); tm != "noaux_tc")
    reject("topk_method", "only noaux_tc is implemented");
  if (const std::string rd = optional_string(root, "moe_router_dtype", "float32"); rd != "float32")
    reject("moe_router_dtype", "only float32 routing is implemented");
  require_absent(root, "swiglu_limit", "clamped swiglu is the GLM-5.3-Flash expert; this family's experts are unclamped");

  // --- MTP --------------------------------------------------------------------
  c.num_nextn_predict_layers = optional_int(root, "num_nextn_predict_layers", 0);
  if (c.num_nextn_predict_layers != 0 && c.num_nextn_predict_layers != 1)
    reject("num_nextn_predict_layers", "only the single draft layer is implemented");
  if (c.mtp_layer() >= 0 && !c.is_moe_layer(c.mtp_layer()))
    reject("num_nextn_predict_layers", "the draft layer carries the MoE (first_k_dense_replace)");

  const QuantRules rules = parse_quantization(root, c, dir);
  derive_packed_shape(rules, c);
  if (c.packed_layer_begin < c.first_k_dense_replace)
    reject("quantization_config", "a packed dense layer is not implemented");
  if (c.expert_mixed346()) check_mixed346_coverage(c);
  return c;
}

GlmDsaTextConfig GlmDsaTextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  const size_t slash = path.find_last_of('/');
  return parse(parsed.root, slash == std::string::npos ? "." : path.substr(0, slash));
}

GlmMoeConfig GlmDsaTextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = n_shared_experts;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = std::numeric_limits<float>::infinity();  // GlmMoeDsaMLP: no clamps
  m.router_mode = MoeRouterMode::SigmoidBias;
  m.routed_int8_activations = expert_activation_bits == 8;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
