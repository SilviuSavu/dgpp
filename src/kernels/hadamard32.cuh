#pragma once
// Device side of kernels/hadamard32.hpp: the warp butterfly the row
// kernel (hadamard32.cu) is built on.
#include <cuda_runtime.h>

#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/hadamard32.hpp"

namespace dgpp {

// One 32-element block across a warp: lane j holds element j. Five xor
// butterflies in fp32, the normalization, one bf16 rounding — the op
// order of hadamard32_block_host exactly.
__device__ __forceinline__ float hadamard32_warp(float v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int s = 1; s < kHadamard32Block; s *= 2) {
    const float other = __shfl_xor_sync(0xFFFFFFFFu, v, s);
    v = (lane & s) ? other - v : v + other;
  }
  return v * kHadamard32Scale;
}
__device__ __forceinline__ uint16_t hadamard32_warp_bf16(uint16_t x) {
  return float_to_bf16_bits(hadamard32_warp(bf16_bits_to_float(x)));
}

}  // namespace dgpp
