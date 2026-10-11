#pragma once
// The Mixed346 GEMV as a single-matrix launcher (2026-10-09):
//   D[m, n] = A[m, k] x W[n, k]^T
// with W a Mixed346 matrix (kPackedScaleBf16G128Mixed346: 3-, 4- or 6-bit
// codebook indices as a dense bit stream per row, one bf16 scale per 128)
// and A the checkpoint's int8 activation codes (codes [m, k] at
// `code_stride` bytes, one fp32 scale per 128 codes at `scale_stride`
// floats — kernels/hadamard32.hpp's quantizer). Rows are chunked four at a
// time through the core the Mixed346 slot and grouped kernels use
// (packq_a8_gemv.cuh), so every output row is bitwise invariant to m and to
// the launcher — the tests' reference for those paths.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

void launch_packq_a8_gemv_bf16(const int8_t* codes, size_t code_stride, const float* scales, size_t scale_stride,
                               const GlmPackedMatrix& w, uint16_t* out, int m, int n, int k, cudaStream_t stream);
void launch_packq_a8_gemv_f32(const int8_t* codes, size_t code_stride, const float* scales, size_t scale_stride,
                              const GlmPackedMatrix& w, float* out, int m, int n, int k, cudaStream_t stream);

// True when the core accepts this matrix (format 3 at 3, 4 or 6 bits, k a
// multiple of 128 in the compiled set, a 16-byte aligned payload).
bool packq_a8_gemv_accepts(const GlmPackedMatrix& w);

}  // namespace dgpp
