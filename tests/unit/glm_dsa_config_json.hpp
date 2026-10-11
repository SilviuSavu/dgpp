#pragma once
// The full GLM-5.3 config.json (HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64,
// transcribed 2026-09-12; the NF4I8 quantization block from
// HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128, transcribed 2026-10-08) as the
// unit tests build it: the per-layer lists are generated, the rest is the
// file's text. Shared by the config and binding tests.
#include <stdexcept>
#include <string>

namespace glm_dsa_test {

inline const char* kHeadPrefix = R"({
  "architectures": ["GlmMoeDsaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "dtype": "bfloat16", "eos_token_id": [154820, 154827, 154829], "ep_size": 1,
  "first_k_dense_replace": 3, "head_dim": 192, "hidden_act": "silu", "hidden_size": 6144,
  "index_head_dim": 128, "index_n_heads": 32, "index_share_for_mtp_iteration": true,
  "index_skip_topk_offset": 3, "index_topk": 2048, "index_topk_freq": 4, "index_topk_pattern": null,
  "indexer_rope_interleave": true, "indexer_types": [INDEXER_TYPES], "initializer_range": 0.02,
  "intermediate_size": 12288, "kv_lora_rank": 512, "max_position_embeddings": 1048576,
  "mlp_layer_types": [MLP_TYPES], "model_type": "glm_moe_dsa", "moe_intermediate_size": 2048,
  "moe_layer_freq": 1, "moe_router_dtype": "float32", "n_group": 1, "n_routed_experts": 256,
  "n_shared_experts": 1, "norm_topk_prob": true, "num_attention_heads": 64,
  "num_experts_per_tok": 8, "num_hidden_layers": 78, "num_key_value_heads": 64,
  "num_nextn_predict_layers": 1, "pad_token_id": 154820, "pretraining_tp": 1, "q_lora_rank": 2048,
  "qk_head_dim": 256, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "rms_norm_eps": 1e-05,
  "rope_interleave": true, "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
  "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": false,
  "topk_group": 1, "topk_method": "noaux_tc", "transformers_version": "5.15.0", "use_cache": true,
  "v_head_dim": 256, "vocab_size": 154880,
)";

// The compressed-tensors pack-quantized block (the int4/int8 RTN release).
inline const char* kCompressedTensorsQuantization = R"(  "quantization_config": {
    "config_groups": {
      "group_0": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\.(?:self_attn\\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)|mlp\\.shared_experts\\.(?:gate_proj|up_proj|down_proj))$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 8, "observer": "memoryless_minmax", "observer_kwargs": {},
                    "scale_dtype": null, "strategy": "group", "symmetric": true, "type": "int",
                    "zp_dtype": null}
      },
      "group_1": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 4, "observer": "memoryless_minmax", "observer_kwargs": {},
                    "scale_dtype": null, "strategy": "group", "symmetric": true, "type": "int",
                    "zp_dtype": null}
      }
    },
    "format": "pack-quantized", "global_compression_ratio": null,
    "ignore": ["lm_head", "re:.*embed_tokens.*", "re:model\\.layers\\.[0-2]\\..*",
               "re:.*self_attn\\.indexer\\..*", "re:.*norm.*", "re:.*mlp\\.gate$"],
    "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}, "version": "0.18.0"
  }
)";

// The NF4I8 mixed-codebook-packed block (HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128).
inline const char* kNf4i8Quantization = R"(  "quantization_config": {
    "format": "mixed-codebook-packed",
    "format_version": 1,
    "other_weights": "Unchanged from the pinned baseline, including all MTP draft tensors",
    "quant_method": "dgpp_nf4i8",
    "retained_int8": {
      "group_size": 64,
      "layers": [3, 77],
      "modules": "Attention and shared expert projections already quantized in the baseline",
      "packing": "Offset-binary signed INT8, 4 codes per int32, low byte first",
      "scale_dtype": "bfloat16"
    },
    "routed_experts": {
      "bits": 4,
      "codebook": [-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127],
      "group_size": 128,
      "indices_dtype": "int32",
      "indices_key": "weight_indices",
      "input_transform": {
        "apply_to": ["gate_proj", "up_proj", "down_proj"],
        "arithmetic": "float32 transform then bfloat16 rounding",
        "axis": "input_channels",
        "block_size": 32,
        "type": "normalized_hadamard"
      },
      "layers": [3, 77],
      "packing": "8 indices per word, low nibble first",
      "scale_dtype": "bfloat16"
    }
  }
)";

inline const char* kTail = "})";

// The Mixed346 block (HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128,
// transcribed 2026-10-09) with its two sidecars inlined: the baseline is
// the int4/int8 release's compressed-tensors block, the recipe lists every
// routed expert of layers 3..77 — expert e as kMixed346Forms[(l + e) % 8]
// (the release's seven width forms and the "existing" fallback) — with the
// release's codebooks, activation policy and scale limits.
inline const char* kMixed346Forms[8] = {"444", "334", "446", "333", "666", "443", "336", "existing"};
inline std::string mixed346_quantization(int layers = 78, int dense = 3, int experts = 256) {
  std::string baseline(kCompressedTensorsQuantization);
  // The block's own text past the key: "  \"quantization_config\": {" ... "  }\n".
  baseline = baseline.substr(baseline.find('{'));
  const size_t close = baseline.rfind('}');
  baseline = baseline.substr(0, close + 1);
  std::string recipes, counts;
  int n_form[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int l = dense; l < layers; ++l) {
    if (l > dense) recipes += ",\n";
    recipes += "      \"" + std::to_string(l) + "\": {";
    for (int e = 0; e < experts; ++e) {
      if (e) recipes += ", ";
      const int f = (l + e) % 8;
      recipes += "\"" + std::to_string(e) + "\": \"" + kMixed346Forms[f] + "\"";
      ++n_form[f];
    }
    recipes += "}";
  }
  for (int i = 0; i < 8; ++i) {
    if (i) counts += ", ";
    counts += "\"" + std::string(kMixed346Forms[i]) + "\": " + std::to_string(n_form[i]);
  }
  std::string six;
  for (int i = 0; i < 64; ++i) six += (i ? ", " : "") + std::to_string(i - 32);
  return std::string(R"json(  "quantization_config": {
    "activation_bits": 8,
    "baseline_config_file": "baseline-quantization-config.json",
    "format": "dgpp_mixed346_h32_a8_g128_v1",
    "group_size": 128,
    "quant_method": "dgpp_mixed346",
    "recipe_file": "quantization-recipe.json",
    "rotation_size": 32,
    "version": 1,
    "baseline": )json") + baseline + R"json(,
    "recipe": {
      "activation": {"clamp": [-128, 127], "format": "int8", "group_size": 128,
                     "rotation": "H32 then BF16 rounding", "rounding": "nearest ties-to-even",
                     "scale": "max(amax/127,1e-30)", "scale_dtype": "float32"},
      "fallback": "existing: byte-exact INT4 g64, BF16 activations, no H32",
      "format": "dgpp_mixed346_h32_a8_g128_v1",
      "group_size": 128,
      "layer_expert_recipes": {
)json" + recipes + R"json(
      },
      "packing": "I32 [N,K*bits/32], unsigned words, low bits first, codes cross word boundaries",
      "projection_order": ["gate_proj", "up_proj", "down_proj"],
      "recipe_counts": {)json" + counts + R"json(},
      "rotation_size": 32,
      "scale_limits": {"3": 169.33333333333334, "4": 127, "6": 31},
      "version": 1,
      "weight_codebooks": {"3": [-127.0, -79.0, -45.0, -14.0, 14.0, 45.0, 79.0, 127.0],
                           "4": [-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127],
                           "6": [)json" + six + R"json(]}
    }
  }
)json";
}

// transformers' schedule: full when max(i - offset + 1, 0) % freq == 0.
inline std::string indexer_types_json(int layers = 78, int freq = 4, int offset = 3) {
  std::string s;
  for (int i = 0; i < layers; ++i) {
    if (i) s += ", ";
    const int k = i - offset + 1 > 0 ? i - offset + 1 : 0;
    s += (k % freq == 0) ? "\"full\"" : "\"shared\"";
  }
  return s;
}

inline std::string mlp_types_json(int layers = 78, int dense = 3) {
  std::string s;
  for (int i = 0; i < layers; ++i) {
    if (i) s += ", ";
    s += i < dense ? "\"dense\"" : "\"sparse\"";
  }
  return s;
}

inline std::string assemble(const char* quantization, const std::string& patch_from, const std::string& patch_to,
                            const std::string& indexer_types) {
  std::string s = std::string(kHeadPrefix) + quantization + kTail;
  s.replace(s.find("[INDEXER_TYPES]"), 15, "[" + indexer_types + "]");
  s.replace(s.find("[MLP_TYPES]"), 11, "[" + mlp_types_json() + "]");
  if (!patch_from.empty()) {
    const size_t at = s.find(patch_from);
    if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + patch_from);
    s.replace(at, patch_from.size(), patch_to);
  }
  return s;
}

// The int4/int8 release's config, optionally with one substring patched.
inline std::string config_json(const std::string& patch_from = "", const std::string& patch_to = "",
                               const std::string& indexer_types = indexer_types_json()) {
  return assemble(kCompressedTensorsQuantization, patch_from, patch_to, indexer_types);
}

// The NF4I8 release's config, optionally with one substring patched.
inline std::string config_json_nf4i8(const std::string& patch_from = "", const std::string& patch_to = "",
                                     const std::string& indexer_types = indexer_types_json()) {
  return assemble(kNf4i8Quantization, patch_from, patch_to, indexer_types);
}

// The Mixed346 release's config (sidecars inlined), optionally patched.
inline std::string config_json_mixed346(const std::string& patch_from = "", const std::string& patch_to = "",
                                        const std::string& indexer_types = indexer_types_json()) {
  static const std::string block = mixed346_quantization();
  return assemble(block.c_str(), patch_from, patch_to, indexer_types);
}

}  // namespace glm_dsa_test
