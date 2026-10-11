#pragma once
// Expected-tensor table for full GLM-5.3 (GlmMoeDsaForCausalLM, 2026-09-12,
// docs/glm53_plan.md §1.5): every tensor the checkpoint must contain —
// names, dtypes, exact shapes — derived from the parsed config, not
// observed from one file. The table drives the offline validator and the
// resident loader, as the other three tables do.
//
// Naming is checkpoint truth (HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64):
// main layers under `model.layers.L.` (one shard each), the draft layer
// under `model.layers.<num_hidden_layers>.` (in passthrough.safetensors,
// no copies of the embedding or the head), the globals
// `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`.
//
// Format contract (compressed-tensors pack-quantized, symmetric, group
// 64): a packed [N, K] matrix X is the triple X.weight_packed I32
// [N, K*bits/32] (unsigned codes offset 2^(bits-1), the low nibble/byte
// first along K), X.weight_scale BF16 [N, K/64] and X.weight_shape I64 [2]
// (= [N, K], checked at load, never trusted for allocation); the
// dequantized weight is bf16(code x scale). Which layers and classes are
// packed, and at what width, comes from the config's rules (attention at
// `attention_bits`, the shared expert at `shared_bits`, the routed experts
// at `expert_bits`, on layers [packed_layer_begin, packed_layer_end)).
// Everything else is BF16 `.weight`; the router bias is F32. The draft
// layer is BF16 throughout (its experts are requantized at load, plan D5).
// Under the NF4I8 contract (config.hpp, 2026-10-08) the routed experts'
// triple is X.weight_indices I32 [N, K/8] (4-bit codebook indices, the low
// nibble first), X.weight_scale BF16 [N, K/128] and X.weight_shape: the
// expected tensor's `scale_fmt` (kPackedScale*) names the form, the
// attention and shared triples keep format 0.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/glm_dsa/config.hpp"

namespace dgpp {

enum class GlmDsaWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,      // input_layernorm, post_attention_layernorm, q_a_layernorm, kv_a_layernorm
  Attention,      // q_a, q_b, kv_a, kv_b, o_proj
  Indexer,        // wq_b, wk, weights_proj, k_norm (+ bias)
  Router,         // mlp.gate.weight, mlp.gate.e_score_correction_bias
  DenseMlp,       // layers < first_k_dense_replace
  SharedExpert,
  RoutedExpert,
  MtpHead,        // enorm, hnorm, eh_proj, shared_head.norm
};

enum class GlmDsaTensorRole : uint8_t {
  Plain,
  IntPacked,    // I32 [N, K*bits/32] — weight_packed (or weight_indices under the codebook format)
  IntScale,     // BF16 [N, K/group] — weight_scale
  IntShape,     // I64 [2] — weight_shape (checked at load)
  Bf16Expert,   // BF16 [N, K] — a draft-layer expert / shared-expert matrix, requantized at load
};

struct GlmDsaExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  GlmDsaWeightClass cls = GlmDsaWeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals; mtp_layer() for the draft
  int expert = -1;  // routed-expert id, -1 otherwise
  GlmDsaTensorRole role = GlmDsaTensorRole::Plain;
  int bits = 0;     // the packed width of a triple's members; 0 for BF16
  int scale_fmt = 0;  // the triple's packed scale format (kPackedScale*, quant_matrix.hpp)

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool packed() const { return role == GlmDsaTensorRole::IntPacked; }
};

// The checkpoint name prefix of a layer ("model.layers.L." for main and
// draft layers alike).
std::string glm_dsa_layer_prefix(const GlmDsaTextConfig& cfg, int layer);

// The three names of one packed matrix: base + ".weight_packed" (or
// ".weight_indices" under the codebook format) / ".weight_scale" /
// ".weight_shape" (`base` is e.g. "...gate_proj").
const char* glm_dsa_packed_words_suffix(int scale_fmt);
std::vector<GlmDsaExpectedTensor> glm_dsa_expected_text_tensors(const GlmDsaTextConfig& cfg);
std::vector<GlmDsaExpectedTensor> glm_dsa_expected_layer_tensors(const GlmDsaTextConfig& cfg, int layer);
std::vector<GlmDsaExpectedTensor> glm_dsa_expected_global_tensors(const GlmDsaTextConfig& cfg);

struct GlmDsaTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct GlmDsaBindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t packed_int4_matrices = 0;   // offset-code int4 AND 4-bit codebook matrices
  size_t packed_int8_matrices = 0;
  size_t packed_mixed_matrices = 0;  // the Mixed346 3- and 6-bit `weight_indices` matrices
  size_t codebook_matrices = 0;      // every `weight_indices` matrix (NF4I8 and Mixed346, any width)
  size_t bf16_expert_matrices = 0;
  std::vector<std::string> errors;
  // Tensors of main layers beyond the config's stack (a truncated
  // diagnostic forward over the first layers of a full checkpoint,
  // glm_dsa_forward_check --layers N): counted, not errors.
  int beyond_stack = 0;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

GlmDsaBindReport glm_dsa_validate_text_binding(
    const GlmDsaTextConfig& cfg,
    const std::unordered_map<std::string, GlmDsaTensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/glm53_plan.md §2): throws
// std::invalid_argument naming the dim that does not divide. world must
// divide the attention heads and the vocabulary; every expert / shared /
// dense intermediate slice and every o_proj input slice must be a multiple
// of the packed group (64), which is also the packed-word boundary; the
// routed expert slice a multiple of the experts' own group (128 under the
// codebook format, which also keeps every 32-wide rotation block inside
// one rank's down-projection columns).
void glm_dsa_tp_validate_geometry(const GlmDsaTextConfig& cfg, int rank, int world);

}  // namespace dgpp
