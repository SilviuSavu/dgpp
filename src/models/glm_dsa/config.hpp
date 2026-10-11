#pragma once
// Full GLM-5.3 (GlmMoeDsaForCausalLM, model_type glm_moe_dsa) configuration,
// parsed from the checkpoint's config.json root (2026-09-12,
// docs/glm53_plan.md §1.1). The same policy as the other three parsers:
// every field the assembly consumes is parsed into a known-supported value
// or rejected with a message naming the field, at load time. The reference
// is transformers' modeling_glm_moe_dsa.py (main, 2026-09) and vLLM's
// deepseek_v2.py / deepseek_mtp.py for the draft layer.
//
// The quantization contract (HawkBearPig/GLM-5.3-Int4-Int8Mix-{RTN,AWQ}-g64,
// compressed-tensors 0.18 `pack-quantized`, plan §1.5): a contiguous range
// of main layers carries its attention projections (q_a, q_b, kv_a, kv_b,
// o_proj), its shared expert and its routed experts as packed-int triples —
// `weight_packed` I32 [N, K*bits/32] (unsigned codes offset 2^(bits-1), the
// low nibble/byte first), `weight_scale` BF16 [N, K/group] and
// `weight_shape` I64 [2] — symmetric, group 64 along K, no zero points.
// Everything else (the dense layers, the indexers, routers, norms, embedding,
// head and the whole draft layer) is BF16. Which module is packed, and at
// how many bits, is derived from the file's `config_groups` targets and
// `ignore` rules exactly as compressed-tensors applies them (a `re:` entry
// is a regex anchored at the start of the module name, anything else an
// exact name), then checked to be the one shape the loader implements.
//
// The NF4I8 contract (HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128, 2026-10-08,
// quant_method `dgpp_nf4i8`, format `mixed-codebook-packed` version 1):
// the same shape with the routed experts of the packed layers as
// `weight_indices` I32 [N, K/8] (4-bit indices into the fixed 16-level
// int8 codebook kNf4i8Codebook, the low nibble first), `weight_scale`
// BF16 [N, K/128] and `weight_shape`; the attention and shared-expert
// triples retained from the int8 g64 baseline. The experts were quantized
// in a rotated input basis (a normalized 32-wide Hadamard on their input
// channels: gate/up on the expert input, down on the SwiGLU output), so
// the engine rotates the routed experts' inputs the same way
// (expert_input_hadamard32). The file's codebook and transform must equal
// the ones the kernels compile; anything else is rejected by name.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct GlmDsaTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 6144;
  int vocab_size = 154880;
  int num_hidden_layers = 78;
  float rms_norm_eps = 1e-5f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 1048576;
  int first_k_dense_replace = 3;   // layers [0, first_k_dense_replace) carry the dense MLP
  std::vector<int64_t> eos_token_ids;  // config.json's; serving resolves generation stop IDs separately
  int64_t pad_token_id = -1;

  // --- MLA attention (DeepSeek-V3 shape with interleaved decoupled RoPE)
  int num_attention_heads = 64;
  int q_lora_rank = 2048;
  int kv_lora_rank = 512;
  int qk_nope_head_dim = 192;
  int qk_rope_head_dim = 64;
  int v_head_dim = 256;
  bool attention_bias = false;
  double rope_theta = 8e6;
  bool rope_interleave = true;      // pairs (2i, 2i+1); the only form implemented

  // --- DSA indexer ----------------------------------------------------------
  int index_n_heads = 32;
  int index_head_dim = 128;
  int index_topk = 2048;            // selected tokens per query; kpool is 1 (per-token)
  bool indexer_rope_interleave = true;
  bool index_share_for_mtp_iteration = false;
  int index_topk_freq = 1;
  int index_skip_topk_offset = 2;
  // Per main layer: 1 = "full" (owns an indexer, selects), 0 = "shared"
  // (attends with the last full layer's selection). The draft layer always
  // owns one.
  std::vector<uint8_t> indexer_full;

  // --- MLP / MoE -----------------------------------------------------------
  int intermediate_size = 12288;      // the dense layers
  int moe_intermediate_size = 2048;   // per routed expert (and the shared expert, x n_shared_experts)
  int n_routed_experts = 256;
  int n_shared_experts = 1;
  int num_experts_per_tok = 8;
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  int n_group = 1;
  int topk_group = 1;

  // --- MTP --------------------------------------------------------------------
  int num_nextn_predict_layers = 1;  // 0 or 1

  // --- packed-int weight formats (plan D2) -----------------------------------
  int packed_group_size = 64;   // elements per scale along K of the attention / shared triples (the cores implement 64)
  int attention_bits = 8;       // q_a, q_b, kv_a, kv_b, o_proj of the packed layers
  int shared_bits = 8;          // the shared expert of the packed layers
  int expert_bits = 4;          // the routed experts of the packed layers
  int packed_layer_begin = 3;   // [begin, end): the main layers whose matrices are packed
  int packed_layer_end = 78;
  // The routed experts' packed scale format on the packed layers
  // (kPackedScale*, models/quant_matrix.hpp): bf16 per 64 offset codes
  // (compressed-tensors), or the NF4I8 codebook with bf16 scales per 128.
  int expert_scale_fmt = kPackedScaleBf16G64;
  // The routed experts' inputs are rotated by a normalized 32-wide
  // Hadamard before every projection (the NF4I8 checkpoint's basis).
  bool expert_input_hadamard32 = false;
  int expert_group_size() const { return packed_scale_group(expert_scale_fmt); }
  // The Mixed346 contract (2026-10-09, quant_method dgpp_mixed346,
  // HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128): the routed experts of
  // the packed layers are, expert by expert, either a converted triple
  // (kPackedScaleBf16G128Mixed346: gate and up at one width of 3, 4 or 6
  // bits, down at its own; H32-rotated int8 activations, one fp32 scale per
  // 128) or the baseline's int4 g64 triple with bf16 activations
  // ("existing"). expert_recipe[l * n_routed_experts + e] for every main
  // layer l: 0 = not listed (an unpacked layer), kExpertRecipeExisting, or
  // (gate_up_bits << 4) | down_bits.
  std::vector<uint8_t> expert_recipe;
  static constexpr uint8_t kExpertRecipeExisting = 1;
  // The routed experts' activation codes: 8 = dynamic int8 per 128 rotated
  // values (the Mixed346 contract), 0 = bf16.
  int expert_activation_bits = 0;
  bool expert_mixed346() const { return expert_scale_fmt == kPackedScaleBf16G128Mixed346; }

  // Parses config.json's root object. Throws std::runtime_error naming the
  // offending field on anything unsupported. `dir` is the checkpoint
  // directory the contract's sidecar files (the Mixed346 recipe and
  // baseline config) are read from; without it those must be inlined.
  static GlmDsaTextConfig parse(const minijson::Value& root, const std::string& dir = "");
  static GlmDsaTextConfig from_json_file(const std::string& path);

  // Layer index of the draft layer (num_hidden_layers), -1 when absent.
  int mtp_layer() const { return num_nextn_predict_layers == 1 ? num_hidden_layers : -1; }
  // Whether layer `l` (a main layer or the draft) carries the MoE.
  bool is_moe_layer(int l) const { return l >= first_k_dense_replace; }
  int num_moe_layers() const { return num_hidden_layers - first_k_dense_replace; }
  int shared_expert_inter() const { return n_shared_experts * moe_intermediate_size; }
  int qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
  // Whether layer `l` (a main layer or the draft) runs its own indexer.
  bool owns_indexer(int l) const {
    if (l == mtp_layer()) return true;
    return l >= 0 && l < num_hidden_layers && indexer_full[static_cast<size_t>(l)] != 0;
  }
  // Main-stack layers that own an indexer (the index caches the pool holds).
  int num_indexer_layers() const {
    int n = 0;
    for (uint8_t f : indexer_full) n += f != 0;
    return n;
  }
  // Whether a main layer's attention, shared expert and routed experts are
  // packed-int triples (the draft is never packed in the file).
  bool packed_layer(int l) const { return l >= packed_layer_begin && l < packed_layer_end; }
  // The bits a module of layer `l` is stored at: 0 = BF16 `.weight`.
  int attention_bits_of(int l) const { return packed_layer(l) ? attention_bits : 0; }
  int shared_bits_of(int l) const { return packed_layer(l) ? shared_bits : 0; }
  int expert_bits_of(int l) const { return packed_layer(l) ? expert_bits : 0; }
  // The routed experts' scale format on layer `l` (the draft's requantized
  // experts take the baseline form, group 64, unrotated).
  int expert_scale_fmt_of(int l) const { return packed_layer(l) ? expert_scale_fmt : kPackedScaleBf16G64; }
  bool expert_input_hadamard32_of(int l) const { return packed_layer(l) && expert_input_hadamard32; }
  int expert_activation_bits_of(int l) const { return packed_layer(l) ? expert_activation_bits : 0; }
  // One routed expert's triple under the Mixed346 contract: its scale
  // format (3 for a converted expert, 0 for an existing one, the draft's
  // and every other contract's) and the width of projection `which` (0
  // gate, 1 up, 2 down).
  uint8_t expert_recipe_of(int l, int e) const {
    if (!expert_mixed346() || !packed_layer(l)) return 0;
    return expert_recipe[static_cast<size_t>(l) * static_cast<size_t>(n_routed_experts) + static_cast<size_t>(e)];
  }
  int expert_scale_fmt_of(int l, int e) const {
    const uint8_t r = expert_recipe_of(l, e);
    return r > kExpertRecipeExisting ? kPackedScaleBf16G128Mixed346 : expert_scale_fmt_of(l) == kPackedScaleBf16G128Mixed346 ? kPackedScaleBf16G64 : expert_scale_fmt_of(l);
  }
  int expert_proj_bits_of(int l, int e, int which) const {
    const uint8_t r = expert_recipe_of(l, e);
    if (r > kExpertRecipeExisting) return which == 2 ? (r & 15) : (r >> 4);
    return expert_bits_of(l);
  }

  // The routed chain's configuration (models/glm/moe.hpp): the GLM sigmoid
  // router, the shared expert in the chain, no swiglu clamps. `local_inter`
  // is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
