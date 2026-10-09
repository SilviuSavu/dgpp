#pragma once
// The DFlash2 drafter's GLM-5.3 bindings (docs/performance_improvement_plan.md §7,
// the port of the Qwen3.5-family integration to the full GLM/DSA model).
//
// The drafter's own types — DFlash2Config, DFlash2Weights, the checkpoint
// loader — are model-agnostic and live in models/qwen/dflash2.hpp (their
// first home); this header adds what the GLM side needs on top: the
// target-pair validation against GlmDsaTextConfig and the drafter's own KV
// planes (GLM's cache pool is DSA-latent and cannot host the drafter's
// plain GQA layers).
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/qwen/dflash2.hpp"

namespace dgpp {

// The silent-acceptance-failure hazards of a mismatched pair (§7 step 2:
// widths, vocab, tap indices) as free validation against the GLM target —
// DFlash2Config::validate_against is a member bound to the Qwen target.
// Throws with a message naming the field.
void dflash2_validate_against(const DFlash2Config& c, const GlmDsaTextConfig& target);

// ---------------------------------------------------------------------------
// The DFlash2 drafter's own KV planes (the GLM port's dedicated store).
//
// The drafter is a plain GQA stack with a sliding window; GLM's cache pool
// is DSA-latent and cannot host it. Each of the drafter's layers gets one
// k and one v plane over a per-request ring of blocks; the block table
// aliases absolute block indices onto the ring (entry j -> ring slot
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
