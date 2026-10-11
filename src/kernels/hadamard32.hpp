#pragma once
// The 32-wide normalized Hadamard rotation of activation rows (2026-10-08,
// HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128, its FORMAT.md): the routed
// experts of that checkpoint were quantized after right-multiplying each
// weight matrix by a block-diagonal Sylvester Hadamard H32 / sqrt(32) on
// its input channels, so the engine applies the same rotation to every
// routed projection's input — the expert input before gate / up, the
// SwiGLU output before down. The residual stream, the router and the
// shared expert see the unrotated activations.
//
// The ONE arithmetic (the kernels, the host reference below and the
// checkpoint's exporter agree op for op): per block of 32 consecutive
// elements, five in-place butterfly stages in fp32 at strides 1, 2, 4, 8,
// 16 — element j takes v_j + v_{j^s} when bit s of j is clear and
// v_{j^s} - v_j when it is set — then one multiply by the fp32 constant
// kHadamard32Scale (1/sqrt(32)) and one rounding to bf16. The exporter's
// torch form divides by 32**.5 where the engine multiplies by the
// reciprocal, a last-ulp fp32 difference before the bf16 rounding; its
// stage order is this one, so the fp32 sums are the same.
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/dtypes.hpp"

// CUDA-free (the host-only model library holds the oracle): the stream
// type is the runtime's own forward declaration.
struct CUstream_st;

namespace dgpp {

constexpr int kHadamard32Block = 32;
constexpr float kHadamard32Scale = 0.17677669529663687f;  // fp32(1 / sqrt(32))

// out[r, :] = H32(in[r, :]) for `rows` bf16 rows of `k` elements (k a
// multiple of 32); strides in elements. in == out is allowed.
// The same over the rows whose index is not `period - 1` modulo `period`
// (period 0: every row): the decode slot layout t * (top_k + 1) + j, whose
// shared slot j == top_k keeps its row as staged. Skipped rows are not
// written (an in-place use).
void launch_hadamard32_rows_skip(const uint16_t* in, size_t in_stride, uint16_t* out,
                                 size_t out_stride, int rows, int k, int period,
                                 CUstream_st* stream);
void launch_hadamard32_rows(const uint16_t* in, size_t in_stride, uint16_t* out, size_t out_stride,
                            int rows, int k, CUstream_st* stream);

// The Mixed346 checkpoint's activation codes (2026-10-09, its FORMAT.md):
// each row rotated by H32 as above, then per group of 128 rotated bf16
// values the fp32 scale s = max(amax / 127, 1e-30) and the int8 codes
// clamp(rint(x / s), -128, 127) — the division rounded to nearest, rint
// ties to even. codes [rows, k] int8 at `code_stride` bytes, scales [rows,
// k / 128] fp32 at `scale_stride` floats; k a multiple of 128. Rows whose
// index is `period - 1` modulo `period` are not written (period 0: every
// row) — the decode slot layout's shared slot.
void launch_hadamard32_quant_int8_rows(const uint16_t* in, size_t in_stride, int8_t* codes, size_t code_stride,
                                       float* scales, size_t scale_stride, int rows, int k, int period,
                                       CUstream_st* stream);

// The host reference, the kernels' arithmetic exactly (the oracles).
inline void hadamard32_block_host(const uint16_t* in, uint16_t* out) {
  float v[kHadamard32Block];
  for (int j = 0; j < kHadamard32Block; ++j) v[j] = bf16_bits_to_float(in[j]);
  for (int s = 1; s < kHadamard32Block; s *= 2) {
    float next[kHadamard32Block];
    for (int j = 0; j < kHadamard32Block; ++j) {
      const float other = v[j ^ s];
      next[j] = (j & s) ? other - v[j] : v[j] + other;
    }
    for (int j = 0; j < kHadamard32Block; ++j) v[j] = next[j];
  }
  for (int j = 0; j < kHadamard32Block; ++j) out[j] = float_to_bf16_bits(v[j] * kHadamard32Scale);
}
inline void hadamard32_rows_host(const uint16_t* in, size_t in_stride, uint16_t* out, size_t out_stride,
                                 int rows, int k) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < k; c += kHadamard32Block)
      hadamard32_block_host(in + static_cast<size_t>(r) * in_stride + c,
                            out + static_cast<size_t>(r) * out_stride + c);
}
// The quantizer's arithmetic exactly on one rotated bf16 group of 128:
// the fp32 scale and the int8 codes (host fp32 division and std::rint
// under the default rounding mode are the kernel's __fdiv_rn and rintf).
inline float hadamard32_quant_int8_group_host(const uint16_t* rotated, int8_t* codes) {
  float amax = 0.0f;
  for (int j = 0; j < 128; ++j) amax = std::fmax(amax, std::fabs(bf16_bits_to_float(rotated[j])));
  const float s = std::fmax(amax / 127.0f, 1e-30f);
  for (int j = 0; j < 128; ++j) {
    float q = std::rint(bf16_bits_to_float(rotated[j]) / s);
    q = std::fmin(std::fmax(q, -128.0f), 127.0f);
    codes[j] = static_cast<int8_t>(static_cast<int>(q));
  }
  return s;
}
inline void hadamard32_quant_int8_rows_host(const uint16_t* in, size_t in_stride, int8_t* codes, size_t code_stride,
                                            float* scales, size_t scale_stride, int rows, int k) {
  uint16_t rot[128];
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < k; c += 128) {
      for (int b = 0; b < 128; b += kHadamard32Block)
        hadamard32_block_host(in + static_cast<size_t>(r) * in_stride + c + b, rot + b);
      scales[static_cast<size_t>(r) * scale_stride + c / 128] =
          hadamard32_quant_int8_group_host(rot, codes + static_cast<size_t>(r) * code_stride + c);
    }
}

}  // namespace dgpp
