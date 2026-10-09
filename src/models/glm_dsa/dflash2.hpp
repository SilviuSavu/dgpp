#pragma once
// The DFlash2 drafter for the Qwen3.5-27B family (docs/
// performance_improvement_plan.md §7): a block-diffusion draft model from a
// SEPARATE checkpoint (z-lab/Qwen3.8-27B-DFlash2) — five bidirectional
// sliding-window Qwen3 layers over a KV context built from fused target
// features (the tap layers' residual streams concatenated through fc),
// two-tap dynamic grouped convolutions, and a codebook path selector over
// the mask rows' top-K candidates. The embeddings and lm head are the
// target's (the draft checkpoint ships neither); the draft planes live in
// the model's KV pool so block tables, prefix sharing and rollback ride
// the existing protocol.
//
// v1 scope: eager greedy C1 (the plan's implementation sequence); graph
// capture, batching and sampled verification stay open.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "loaders/minijson.hpp"
#include "common/cuda_check.hpp"
#include "models/glm_dsa/config.hpp"

namespace dgpp {

struct DFlash2Config {
  // dflash_config.
  int block_size = 8;             // query rows per draft: 1 anchor + (block_size - 1) mask rows
  int mask_token_id = -1;
  int conv_taps = 2;
  int conv_group_size = 16;
  int selector_rank = 256;
  int selector_top_k = 16;
  std::vector<int> target_layer_ids;  // into the TARGET stack (features = layer outputs)
  // Qwen3 backbone.
  int num_hidden_layers = 5;
  int hidden_size = 5120;
  int vocab_size = 248320;
  int num_attention_heads = 32;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int intermediate_size = 17408;
  float rms_norm_eps = 1e-6f;
  double rope_theta = 1e7;
  int64_t sliding_window = 2048;  // INT64_MAX when the backbone is not sliding
  int64_t max_position_embeddings = 262144;

  static DFlash2Config from_json_file(const std::string& path);
  static DFlash2Config parse(const minijson::Value& root);

  int drafts() const { return block_size - 1; }
  int query_rows() const { return block_size; }
  int kv_row() const { return num_key_value_heads * head_dim; }
  int q_row() const { return num_attention_heads * head_dim; }
  // The tensor-parallel slices (2026-10-04): the attention heads, the kv
  // heads and the MLP rows split across `world` ranks (each must divide);
  // the convs, norms, fc taps and the selector stay replicated, and the
  // o / down projections' partial outputs fold at the boundary.
  bool tp_divisible(int world) const {
    return world >= 1 && num_attention_heads % world == 0 && num_key_value_heads % world == 0 &&
           intermediate_size % world == 0;
  }
  int local_heads(int world) const { return num_attention_heads / world; }
  int local_kv_heads(int world) const { return num_key_value_heads / world; }
  int local_q_row(int world) const { return local_heads(world) * head_dim; }
  int local_kv_row(int world) const { return local_kv_heads(world) * head_dim; }
  int local_intermediate(int world) const { return intermediate_size / world; }
  int conv_groups() const { return hidden_size / conv_group_size; }
  // fc's input width: the taps' outputs concatenated.
  int fc_in() const { return static_cast<int>(target_layer_ids.size()) * hidden_size; }

  // Cross-checks the draft/target pair (silent-acceptance-failure hazards:
  // §7 step 2 — widths, vocab and the tap indices). Throws with a message
  // naming the field.
  void validate_against(const GlmDsaTextConfig& target) const;
};

// One draft layer's device weights (BF16, arena-resident; at world > 1 the
// q|k|v rows, o columns, gate/up rows and down columns are this rank's
// slices — QW, KW and I below are the local widths).
struct DFlash2LayerWeights {
  const uint16_t* input_norm = nullptr;   // [H]
  const uint16_t* post_norm = nullptr;    // [H]
  const uint16_t* qkv = nullptr;          // [QW + 2*KW, H] q|k|v rows stacked (null under fp8)
  const uint16_t* kv_rows = nullptr;      // [2*KW, H] the k|v rows: a view into qkv, or their own bf16 slot under fp8 (the context features' GEMM reads bf16)
  const uint16_t* o = nullptr;            // [H, QW]
  const uint16_t* gate = nullptr;         // [I, H]
  const uint16_t* up = nullptr;           // [I, H]
  const uint16_t* down = nullptr;         // [H, I]
  const uint16_t* q_norm = nullptr;       // [head_dim]
  const uint16_t* k_norm = nullptr;       // [head_dim]
  const uint16_t* attn_conv_base = nullptr;   // [2 sides][taps][H]
  const uint16_t* attn_conv_kp = nullptr;     // [2*taps*groups, H]
  const uint16_t* mlp_conv_base = nullptr;    // [2 sides][taps][H]
  const uint16_t* mlp_conv_kp = nullptr;      // [2*taps*groups, H]
  // engine.dflash_weights fp8 (2026-10-05): the five block matrices as
  // block-128 E4M3 payloads with fp32 scales (kernels/mma_gemv's decode
  // form); no bf16 copy of the five is kept — only the k|v rows (kv_rows),
  // which the feature path's GEMM reads.
  // Null without the recipe. Lossy for the PROPOSALS only (the target's
  // verify is exact whatever the drafter proposes).
  const uint8_t* qkv_fp8 = nullptr;   const float* qkv_scales = nullptr;
  const uint8_t* o_fp8 = nullptr;     const float* o_scales = nullptr;
  const uint8_t* gate_fp8 = nullptr;  const float* gate_scales = nullptr;
  const uint8_t* up_fp8 = nullptr;    const float* up_scales = nullptr;
  const uint8_t* down_fp8 = nullptr;  const float* down_scales = nullptr;
};

struct DFlash2Weights {
  std::vector<DFlash2LayerWeights> layers;
  const uint16_t* fc = nullptr;        // nTaps slices [H, H] (the [H, nTaps*H] split)
  const uint16_t* hidden_norm = nullptr;  // [H]
  const uint16_t* norm = nullptr;      // [H]
  const uint16_t* pred_codebook = nullptr;  // [vocab, rank]
  const uint16_t* succ_codebook = nullptr;  // [vocab, rank]
  const uint16_t* hidden_projection = nullptr;  // [rank, H]
  void* arena = nullptr;
  size_t bytes = 0;
  void* fp8_arena = nullptr;  // the fp8 payloads + scales (engine.dflash_weights fp8)
  size_t fp8_bytes = 0;
  ~DFlash2Weights();
  DFlash2Weights() = default;
  DFlash2Weights(const DFlash2Weights&) = delete;
  DFlash2Weights& operator=(const DFlash2Weights&) = delete;
  DFlash2Weights(DFlash2Weights&& o) noexcept
      : layers(std::move(o.layers)), fc(o.fc), hidden_norm(o.hidden_norm), norm(o.norm),
        pred_codebook(o.pred_codebook), succ_codebook(o.succ_codebook),
        hidden_projection(o.hidden_projection), arena(o.arena), bytes(o.bytes),
        fp8_arena(o.fp8_arena), fp8_bytes(o.fp8_bytes) {
    o.arena = nullptr;
    o.bytes = 0;
    o.fp8_arena = nullptr;
    o.fp8_bytes = 0;
  }
  DFlash2Weights& operator=(DFlash2Weights&& o) noexcept {
    if (this != &o) {
      cudaFree(arena);
      cudaFree(fp8_arena);
      layers = std::move(o.layers);
      fc = o.fc;
      hidden_norm = o.hidden_norm;
      norm = o.norm;
      pred_codebook = o.pred_codebook;
      succ_codebook = o.succ_codebook;
      hidden_projection = o.hidden_projection;
      arena = o.arena;
      bytes = o.bytes;
      fp8_arena = o.fp8_arena;
      fp8_bytes = o.fp8_bytes;
      o.arena = nullptr;
      o.bytes = 0;
      o.fp8_arena = nullptr;
      o.fp8_bytes = 0;
    }
    return *this;
  }
};

// Reads the checkpoint's safetensors shards (BF16, full-vocab), validates
// every name/shape against cfg and leaves the device-resident set — rank
// `rank` of `world`'s slices of the sharded matrices (cfg.tp_divisible).
// The caller's stream orders the uploads.
DFlash2Weights load_dflash2_weights(const DFlash2Config& cfg, const std::string& dir,
                                    cudaStream_t stream, int rank = 0, int world = 1, bool fp8 = false);
// The arena size load_dflash2_weights allocates (the memory plan's line).
size_t dflash2_weights_bytes(const DFlash2Config& cfg, int world = 1, bool fp8 = false);


// ---------------------------------------------------------------------------
// The DFlash2 drafter's own KV planes (the GLM port's dedicated store).
//
// The drafter is a plain GQA stack with a 2048-token sliding window; GLM's
// cache pool is DSA-latent and cannot host it. Each of the drafter's layers
// gets one k and one v plane over a per-request ring of blocks; the block
// table aliases absolute block indices onto the ring (entry j -> ring slot
// j % ring_blocks), which is exactly the window's reuse semantics: writes
// land at table[pos / bt] and the windowed block attention reads the same
// aliasing, so a slot is overwritten only after it has slid out of every
// readable window.
//
// Layout contract (kernels/qsa.hpp): caches are bf16 [slots, kv_heads *
// dim] with the physical slot block_tables[req][pos / block_tokens] *
// block_tokens + pos % block_tokens.
class GlmDFlash2Planes {
 public:
  static constexpr int kBlockTokens = 32;  // self-contained; not the DSA pool's

  // layers: the drafter's layer count; kv_row: kv_heads * head_dim (local);
  // max_requests / window / query_rows / max_context_tokens size the rings
  // and the alias tables.
  void init(int layers, int kv_row, int max_requests, int64_t window, int query_rows,
            int64_t max_context_tokens, cudaStream_t stream);

  int layers() const { return layers_; }
  int ring_blocks() const { return ring_blocks_; }
  int block_tokens() const { return kBlockTokens; }
  int64_t window() const { return window_; }
  int64_t table_stride() const { return table_stride_; }  // alias-table entries per request
  const int32_t* device_tables() const { return tables_; }
  uint16_t* k_cache(int layer) const { return k_[layer]; }
  uint16_t* v_cache(int layer) const { return v_[layer]; }
  size_t bytes() const { return bytes_; }

  void free();

 private:
  int layers_ = 0, ring_blocks_ = 0, max_requests_ = 0;
  int64_t window_ = 0, table_stride_ = 0;
  uint16_t* k_[8] = {};
  uint16_t* v_[8] = {};
  int32_t* tables_ = nullptr;  // [max_requests][ctx_blocks] alias table
  size_t bytes_ = 0;
};

inline void GlmDFlash2Planes::init(int layers, int kv_row, int max_requests, int64_t window,
                                   int query_rows, int64_t max_context_tokens, cudaStream_t stream) {
  layers_ = layers;
  max_requests_ = max_requests;
  window_ = window;
  ring_blocks_ = static_cast<int>((window + query_rows + kBlockTokens - 1) / kBlockTokens) + 1;
  const int64_t ctx_blocks = (max_context_tokens + kBlockTokens - 1) / kBlockTokens;
  table_stride_ = ctx_blocks;
  const int total_blocks = max_requests * ring_blocks_;
  const size_t plane = static_cast<size_t>(total_blocks) * kBlockTokens * kv_row * 2;
  bytes_ = static_cast<size_t>(layers) * 2 * plane +
           static_cast<size_t>(max_requests) * static_cast<size_t>(ctx_blocks) * 4;
  for (int l = 0; l < layers_; ++l) {
    DGPP_CUDA_OK(cudaMalloc(&k_[l], plane));
    DGPP_CUDA_OK(cudaMalloc(&v_[l], plane));
    DGPP_CUDA_OK(cudaMemsetAsync(k_[l], 0, plane, stream));
    DGPP_CUDA_OK(cudaMemsetAsync(v_[l], 0, plane, stream));
  }
  std::vector<int32_t> host(static_cast<size_t>(max_requests) * static_cast<size_t>(ctx_blocks));
  for (int r = 0; r < max_requests; ++r)
    for (int64_t j = 0; j < ctx_blocks; ++j)
      host[static_cast<size_t>(r) * ctx_blocks + j] = r * ring_blocks_ + static_cast<int>(j % ring_blocks_);
  DGPP_CUDA_OK(cudaMalloc(&tables_, host.size() * 4));
  DGPP_CUDA_OK(cudaMemcpyAsync(tables_, host.data(), host.size() * 4, cudaMemcpyHostToDevice, stream));
}

inline void GlmDFlash2Planes::free() {
  for (int l = 0; l < 8; ++l) {
    if (k_[l]) cudaFree(k_[l]);
    if (v_[l]) cudaFree(v_[l]);
    k_[l] = v_[l] = nullptr;
  }
  if (tables_) cudaFree(tables_);
  tables_ = nullptr;
  bytes_ = 0;
}

}  // namespace dgpp
