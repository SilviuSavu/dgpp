#pragma once
// The host packed-int encoder (2026-09-12, docs/glm53_plan.md D5): a BF16
// [N, K] matrix into the compressed-tensors `pack-quantized` triple the
// checkpoint carries — symmetric int4 or int8 codes, one bf16 scale per 64
// elements of a row, the codes stored unsigned (offset 2^(bits-1)) 32/bits
// per I32 word with the low nibble / byte the first element — with
// llm-compressor's round-to-nearest recipe (QuantizationModifier,
// memoryless_minmax, symmetric):
//
//   s_g  = bf16( max|w_g| / ((2^bits - 1) / 2) )     (int4: / 7.5, int8: / 127.5)
//   code = clamp( rne(w / s_g), -2^(bits-1), 2^(bits-1) - 1 )
//
// so that the kernels' exact dequant (code x s_g) reproduces w to the
// format's precision. The full GLM-5.3 loader applies it to the draft
// layer's BF16 experts and shared expert (every other packed tensor comes
// quantized from the checkpoint). An all-zero group encodes as zero codes
// under a scale of 1.
//
// The Mixed346 format (kPackedScaleBf16G128Mixed346, 2026-10-09,
// HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128, its FORMAT.md): 3-, 4-
// or 6-bit codes as one dense bit stream per row (code k at bit k x bits,
// the low bits first, codes crossing word boundaries), one bf16 scale per
// 128 codes; the level is the width's codebook entry (quant_matrix.hpp).
// packq_field reads such a code; packq_encode_row_m346 is the test
// encoder of that format's RTN form (the checkpoint's codes are GPTQ's).
#include <cmath>
#include <cstdint>

#include "common/dtypes.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// The scale of one 64-element group from its absolute maximum.
inline uint16_t packq_group_scale(float amax, int bits) {
  if (!(amax > 0.0f) || !std::isfinite(amax)) return float_to_bf16_bits(1.0f);
  const float half_range = static_cast<float>((1 << bits) - 1) * 0.5f;
  return float_to_bf16_bits(amax / half_range);
}

// Encodes one row of `cols` fp32 values (cols % 64 == 0): `words` receives
// cols * bits / 32 I32 words, `scales` cols / 64 bf16 scales.
inline void packq_encode_row(const float* row, int64_t cols, int bits, uint32_t* words,
                             uint16_t* scales) {
  const int per = 32 / bits;
  const int q_min = -(1 << (bits - 1)), q_max = (1 << (bits - 1)) - 1;
  const uint32_t mask = (1u << bits) - 1u;
  for (int64_t w = 0; w < cols / per; ++w) words[w] = 0u;
  for (int64_t g = 0; g < cols / kPackedGroup; ++g) {
    float amax = 0.0f;
    for (int j = 0; j < kPackedGroup; ++j) {
      const float a = std::fabs(row[g * kPackedGroup + j]);
      amax = a > amax ? a : amax;
    }
    const uint16_t sc = packq_group_scale(amax, bits);
    scales[g] = sc;
    const float inv = 1.0f / bf16_bits_to_float(sc);
    for (int j = 0; j < kPackedGroup; ++j) {
      const int64_t e = g * kPackedGroup + j;
      float q = std::nearbyint(row[e] * inv);
      if (q < static_cast<float>(q_min)) q = static_cast<float>(q_min);
      if (q > static_cast<float>(q_max)) q = static_cast<float>(q_max);
      const uint32_t u = static_cast<uint32_t>(static_cast<int>(q) + (1 << (bits - 1))) & mask;
      words[e / per] |= u << (bits * (e % per));
    }
  }
}

// The absolute maximum of `n` bf16 values.
inline float packq_amax_bf16(const uint16_t* w, int64_t n) {
  float amax = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const float a = std::fabs(bf16_bits_to_float(w[i]));
    amax = a > amax ? a : amax;
  }
  return amax;
}

// The unsigned stored field of element (r, c): `bits` bits at bit c x bits
// of row r's stream (rows are cols x bits / 32 words). At 4 and 8 bits a
// field never crosses a word; at 3 and 6 (the Mixed346 format) it may, and
// the high part comes from the next word.
inline unsigned packq_field(const uint32_t* words, int64_t cols, int bits, int64_t r, int64_t c) {
  const int64_t wpr = cols * bits / 32;
  const int64_t pos = c * bits;
  const int64_t w = pos / 32;
  const int shift = static_cast<int>(pos % 32);
  const uint32_t mask = (1u << bits) - 1u;
  uint64_t v = words[r * wpr + w] >> shift;
  if (shift + bits > 32) v |= static_cast<uint64_t>(words[r * wpr + w + 1]) << (32 - shift);
  return static_cast<unsigned>(v & mask);
}

// The signed code of element (r, c) of an encoded matrix (the offset form).
inline int packq_code(const uint32_t* words, int64_t cols, int bits, int64_t r, int64_t c) {
  return static_cast<int>(packq_field(words, cols, bits, r, c)) - (1 << (bits - 1));
}

// The integer level of element (r, c): the signed code, or under a
// codebook format (kPackedScaleBf16G128Nf4i8, kPackedScaleBf16G128Mixed346)
// the codebook entry the stored index selects.
inline int packq_level(const uint32_t* words, int64_t cols, int bits, int64_t r, int64_t c,
                       int scale_fmt = kPackedScaleBf16G64) {
  return packed_code_level(packq_field(words, cols, bits, r, c), bits, scale_fmt);
}

// The Mixed346 format's RTN encoder (the tests' and fixtures' producer; the
// checkpoint's codes are GPTQ's): per 128-code group, s = bf16(max|w| /
// limit) with the width's limit (127 / 0.75 at 3 bits, 127 at 4, 31 at 6);
// a codebook width takes the nearest level of w / s (a midpoint the lower
// index), the 6-bit width rounds to nearest even and clamps to [-32, 31].
// `words` receives cols x bits / 32 words, `scales` cols / 128 bf16 scales.
inline float packq_m346_limit(int bits) {
  return bits == 3 ? 127.0f / 0.75f : (bits == 4 ? 127.0f : 31.0f);
}
inline void packq_encode_row_m346(const float* row, int64_t cols, int bits, uint32_t* words,
                                  uint16_t* scales) {
  const int64_t wpr = cols * bits / 32;
  for (int64_t w = 0; w < wpr; ++w) words[w] = 0u;
  const int levels = 1 << bits;
  for (int64_t g = 0; g < cols / 128; ++g) {
    float amax = 0.0f;
    for (int j = 0; j < 128; ++j) {
      const float a = std::fabs(row[g * 128 + j]);
      amax = a > amax ? a : amax;
    }
    const uint16_t sc = (amax > 0.0f && std::isfinite(amax)) ? float_to_bf16_bits(amax / packq_m346_limit(bits))
                                                              : float_to_bf16_bits(1.0f);
    scales[g] = sc;
    const float s = bf16_bits_to_float(sc);
    for (int j = 0; j < 128; ++j) {
      const int64_t e = g * 128 + j;
      const float x = row[e] / s;
      unsigned u;
      if (bits == 6) {
        float q = std::nearbyint(x);
        if (q < -32.0f) q = -32.0f;
        if (q > 31.0f) q = 31.0f;
        u = static_cast<unsigned>(static_cast<int>(q) + 32);
      } else {
        u = 0;
        for (int i = 0; i + 1 < levels; ++i) {
          const float mid = 0.5f * (static_cast<float>(packed_code_level(static_cast<unsigned>(i), bits, kPackedScaleBf16G128Mixed346)) +
                                    static_cast<float>(packed_code_level(static_cast<unsigned>(i + 1), bits, kPackedScaleBf16G128Mixed346)));
          if (x > mid) u = static_cast<unsigned>(i + 1);
        }
      }
      const int64_t pos = e * bits;
      const int64_t w = pos / 32;
      const int shift = static_cast<int>(pos % 32);
      words[w] |= u << shift;
      if (shift + bits > 32) words[w + 1] |= u >> (32 - shift);
    }
  }
}

// Decodes element (r, c) back to the value the kernels compute (level x
// scale, exact in fp32): the tests' oracle and the loader's bf16 bridge.
// scale_fmt selects the group, the scale dtype and the code meaning
// (kPackedScale*).
inline float packq_decode(const uint32_t* words, const uint16_t* scales, int64_t cols, int bits,
                          int64_t r, int64_t c, int scale_fmt = kPackedScaleBf16G64) {
  const int g = packed_scale_group(scale_fmt);
  return static_cast<float>(packq_level(words, cols, bits, r, c, scale_fmt)) *
         packed_scale_to_float(scales[r * (cols / g) + c / g], scale_fmt);
}

}  // namespace dgpp
