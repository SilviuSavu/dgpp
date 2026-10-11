// Parity tests for the packed-int GEMV core (docs/glm53_plan.md G2): the
// single-matrix launcher against a double oracle across every row
// geometry the core has at both code widths (32 .. 1 lanes per row, one
// to eight passes), the row-count invariance the batch family relies on,
// the f32 epilogue as the unrounded bf16, NaN scale propagation, the
// geometry contract, and — with --checkpoint-dir pointing at the
// GLM-5.3 int4/int8 checkpoint — real slices.
//
// The oracle: w = (code - 2^(bits-1)) x bf16(scale) exactly (the exact
// policy — no weight rounding, none in the engine), fp64 accumulation,
// rounded to bf16 as the launcher's bf16 epilogue rounds. The kernel
// differs from it by fp32 accumulation order only, so the budget is the
// strict FP8 suite's: 2 bf16 ulps with a 1e-3 cancellation floor, zero
// mismatches.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/hadamard32.hpp"
#include "kernels/packq_a8_gemv.hpp"
#include "kernels/packq_gemv.hpp"
#include "loaders/packq_quant.hpp"
#include "models/quant_matrix.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int bits, m, n, k;
  int sf = 0;                     // the scale format: 0 bf16 per 64, 1 f16 per 128, 2 the NF4I8 codebook (bf16 per 128)
  std::vector<uint16_t> act;      // [m, k] bf16
  std::vector<uint32_t> packed;   // [n, k*bits/32]
  std::vector<uint16_t> scales;   // [n, k/group] in the format's dtype
  int group() const { return dgpp::packed_scale_group(sf); }
};

// The checkpoint's packing: unsigned codes offset 2^(bits-1), 32/bits per
// word, the low nibble / byte first.
void pack_codes(const std::vector<int>& codes, int bits, std::vector<uint32_t>& out) {
  const int per = 32 / bits;
  out.assign(codes.size() / per, 0u);
  for (size_t i = 0; i < codes.size(); ++i) {
    const uint32_t u = static_cast<uint32_t>(codes[i] + (1 << (bits - 1)));
    out[i / per] |= u << (bits * (i % per));
  }
}

Problem make_problem(int bits, int m, int n, int k, uint64_t seed, int sf = 0) {
  Problem p;
  p.bits = bits;
  p.m = m;
  p.n = n;
  p.k = k;
  p.sf = sf;
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * k);
  fill_act(rng, p.act);
  const int lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
  std::vector<int> codes(static_cast<size_t>(n) * k);
  for (auto& c : codes) c = lo + static_cast<int>(rng.next() % static_cast<uint64_t>(hi - lo + 1));
  pack_codes(codes, bits, p.packed);
  p.scales.resize(static_cast<size_t>(n) * k / p.group());
  for (auto& s : p.scales) {
    // The codebook's levels reach 127, the offset codes 7: scales 16x lower.
    const float v = static_cast<float>(std::exp2(rng.unit() * 2.0) * (dgpp::packed_codebook(sf) ? 0.0625 / 16 : 0.0625));
    s = sf == 1 ? dgpp::float_to_fp16_bits(v) : dgpp::float_to_bf16_bits(v);
  }
  return p;
}

// The integer level of element (nn, kk): the signed code, or the codebook
// entry the stored nibble indexes (format 2).
int code_at(const Problem& p, int nn, int kk) {
  const int per = 32 / p.bits;
  const uint32_t word = p.packed[static_cast<size_t>(nn) * (p.k / per) + kk / per];
  const uint32_t u = (word >> (p.bits * (kk % per))) & ((1u << p.bits) - 1u);
  return dgpp::packed_code_level(u, p.bits, p.sf);
}

std::vector<double> oracle(const Problem& p) {
  std::vector<double> out(static_cast<size_t>(p.m) * p.n, 0.0);
  std::vector<double> wrow(p.k);
  for (int nn = 0; nn < p.n; ++nn) {
    const int g = p.group();
    for (int kk = 0; kk < p.k; ++kk) {
      const float s = dgpp::packed_scale_to_float(
          p.scales[static_cast<size_t>(nn) * (p.k / g) + kk / g], p.sf);
      wrow[kk] = static_cast<double>(code_at(p, nn, kk)) * static_cast<double>(s);
    }
    for (int mm = 0; mm < p.m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow = p.act.data() + static_cast<size_t>(mm) * p.k;
      for (int kk = 0; kk < p.k; ++kk) acc += bf16_to_float(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * p.n + nn] =
          bf16_to_float(dgpp::float_to_bf16_bits(static_cast<float>(acc)));
    }
  }
  return out;
}

struct DevMatrix {
  uint32_t* packed = nullptr;
  uint16_t* scales = nullptr;
  dgpp::GlmPackedMatrix view;
  explicit DevMatrix(const Problem& p) {
    DGPP_CUDA_OK(cudaMallocManaged(&packed, p.packed.size() * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&scales, p.scales.size() * 2));
    std::memcpy(packed, p.packed.data(), p.packed.size() * 4);
    std::memcpy(scales, p.scales.data(), p.scales.size() * 2);
    view = dgpp::GlmPackedMatrix{packed, scales, p.n, p.k, p.bits, p.sf};
  }
  ~DevMatrix() {
    cudaFree(packed);
    cudaFree(scales);
  }
};

std::vector<uint16_t> run_bf16(const Problem& p) {
  DevMatrix w(p);
  uint16_t* act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  dgpp::launch_packq_gemv_bf16(act, p.k, w.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  cudaFree(act);
  cudaFree(out);
  return got;
}

std::vector<float> run_f32(const Problem& p) {
  DevMatrix w(p);
  uint16_t* act = nullptr;
  float* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  dgpp::launch_packq_gemv_f32(act, p.k, w.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 4);
  cudaFree(act);
  cudaFree(out);
  return got;
}

void check_oracle(const Problem& p, const std::vector<uint16_t>& got, const char* label) {
  const auto want = oracle(p);
  const auto rep = compare_bf16_vs_oracle(got.data(), want, /*ulp_budget=*/2.0,
                                          /*floor_frac=*/1e-3);
  std::printf("[ OK ] %s: l2_rel=%.3g mismatches=%ld/%zu\n", label, rep.l2_rel,
              rep.mismatches, rep.total);
  require_report(rep, 1e-3, 0, label);
}

Problem row_of(const Problem& p, int r) {
  Problem q;
  q.bits = p.bits;
  q.sf = p.sf;
  q.m = 1;
  q.n = p.n;
  q.k = p.k;
  q.act.assign(p.act.begin() + static_cast<long>(r) * p.k,
               p.act.begin() + static_cast<long>(r + 1) * p.k);
  q.packed = p.packed;
  q.scales = p.scales;
  return q;
}

}  // namespace

DGPP_TEST(packq_gemv_matches_oracle_across_row_geometries) {
  // k picks the row geometry. int4: 64 (one lane, two chunks), 128 (one
  // lane, four), 256 (2 lanes) ... 4096 (32 lanes x 4 chunks), 6144 (32 x
  // 6 in two passes), 192 (2 x 3), 384 (4 x 3), 8192 (32 x 8), 12288 (32
  // x 12), 16384 (32 x 16 in four passes). int8: 64 (one lane, four
  // chunks), 128 (2 lanes x 4) ... 2048 (32 x 4), 4096 (32 x 8), 6144
  // (32 x 12), 8192 (32 x 16), 16384 (32 x 32 in eight passes), 192 (4 x
  // 3), 384 (8 x 3). n leaves partial blocks and dead lane groups.
  struct Shape {
    int bits, m, n, k;
  };
  const Shape shapes[] = {
      {4, 1, 520, 4096}, {4, 3, 264, 1024}, {4, 2, 100, 512},  {4, 4, 40, 256},
      {4, 1, 33, 128},   {4, 2, 17, 64},    {4, 4, 512, 4096}, {4, 2, 300, 2048},
      {4, 1, 390, 6144}, {4, 3, 130, 6144}, {4, 2, 41, 192},   {4, 1, 77, 384},
      {4, 4, 70, 768},   {4, 2, 20, 8192},  {4, 1, 20, 12288}, {4, 2, 12, 16384},
      {8, 1, 520, 2048}, {8, 3, 264, 1024}, {8, 2, 100, 512},  {8, 4, 40, 256},
      {8, 1, 33, 128},   {8, 2, 17, 64},    {8, 4, 300, 6144}, {8, 2, 200, 4096},
      {8, 1, 77, 192},   {8, 3, 41, 384},   {8, 2, 60, 8192},  {8, 1, 12, 16384},
      {8, 4, 70, 1536},  {8, 2, 30, 3072}};
  int i = 0;
  for (const Shape& s : shapes) {
    const Problem p = make_problem(s.bits, s.m, s.n, s.k, 0x9A0 + i++);
    const std::string label = "packq int" + std::to_string(s.bits) + " gemv M" +
                              std::to_string(s.m) + "xN" + std::to_string(s.n) + "xK" +
                              std::to_string(s.k);
    check_oracle(p, run_bf16(p), label.c_str());
  }
}

DGPP_TEST(packq_gemv_g128_f16_matches_oracle) {
  // The f16-per-128 scale format (docs/qwen38_autoround_int4_plan.md D1)
  // at the AutoRound hybrid's widths — int4 640 (4 lanes x 5 chunks) and
  // 2560 (16 x 5), int8 2560 (32 x 5) and 640 (8 x 5) — and the smaller
  // 128-multiples of the compiled set. Same oracle, same budget: the
  // chain is the format-0 chain with a different scale index and widening.
  struct Shape {
    int bits, m, n, k;
  };
  const Shape shapes[] = {
      {4, 1, 520, 640},  {4, 3, 264, 2560}, {4, 2, 100, 128},  {4, 4, 40, 256},
      {4, 1, 77, 384},   {4, 2, 30, 1024},  {4, 4, 17, 512},   {4, 2, 130, 2560},
      {8, 1, 520, 2560}, {8, 3, 264, 640},  {8, 2, 100, 128},  {8, 4, 40, 256},
      {8, 1, 77, 384},   {8, 2, 30, 2048},  {8, 4, 70, 1536},  {8, 1, 300, 640}};
  int i = 0;
  for (const Shape& s : shapes) {
    const Problem p = make_problem(s.bits, s.m, s.n, s.k, 0xF16 + i++, 1);
    const std::string label = "packq int" + std::to_string(s.bits) + " g128/f16 gemv M" +
                              std::to_string(s.m) + "xN" + std::to_string(s.n) + "xK" +
                              std::to_string(s.k);
    check_oracle(p, run_bf16(p), label.c_str());
  }
  // Row independence and the f32 epilogue at the production widths.
  for (int bits : {4, 8})
    for (int k : {640, 2560}) {
      const Problem p3 = make_problem(bits, 3, 296, k, 0xE3 + k + bits, 1);
      const std::vector<uint16_t> got3 = run_bf16(p3);
      for (int r = 0; r < 3; ++r) {
        const std::vector<uint16_t> got1 = run_bf16(row_of(p3, r));
        require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                            static_cast<size_t>(p3.n) * 2) == 0,
                "packq g128 row bits independent of m (3)");
      }
      const Problem p8 = make_problem(bits, 8, 136, k, 0xE8 + k + bits, 1);
      const std::vector<uint16_t> got8 = run_bf16(p8);
      const std::vector<float> got8f = run_f32(p8);
      for (int r = 0; r < 8; ++r) {
        const Problem p1 = row_of(p8, r);
        const std::vector<uint16_t> got1 = run_bf16(p1);
        const std::vector<float> got1f = run_f32(p1);
        require(std::memcmp(got1.data(), got8.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 2) == 0,
                "packq g128 chunked bf16 row bits independent of m (8)");
        require(std::memcmp(got1f.data(), got8f.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 4) == 0,
                "packq g128 chunked f32 row bits independent of m (8)");
      }
      for (size_t j = 0; j < got8.size(); ++j)
        require(dgpp::float_to_bf16_bits(got8f[j]) == got8[j], "g128 bf16(out_f32) == out_bf16");
    }
  // An f16 NaN scale poisons exactly its row.
  for (int bits : {4, 8}) {
    Problem p = make_problem(bits, 1, 300, 640, 0xA5 + bits, 1);
    p.scales[static_cast<size_t>(9) * (p.k / 128) + 3] = 0x7E00;
    p.scales[static_cast<size_t>(250) * (p.k / 128) + 0] = 0xFE00;
    const std::vector<uint16_t> got = run_bf16(p);
    for (int nn = 0; nn < p.n; ++nn) {
      const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
      require(got_nan == (nn == 9 || nn == 250), "g128 f16 NaN scale poisons exactly its row");
    }
  }
  // The contract: a 128-multiple inside the format's compiled set.
  auto rejects = [](int bits, int k, int sf) {
    Problem p = make_problem(bits, 1, 8, k % 128 == 0 ? k : 128, 0xBAD, 1);
    p.k = k;
    p.sf = sf;
    try {
      (void)run_bf16(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects(4, 192, 1), "g128: k=192 rejected (not a multiple of 128)");
  require(rejects(4, 320, 1), "g128: k=320 rejected");
  require(rejects(4, 4096, 1), "g128: k=4096 rejected (outside the format's compiled set)");
  require(rejects(8, 640, 2), "an unknown scale format is rejected");
  require(!rejects(4, 640, 1) && !rejects(4, 2560, 1) && !rejects(8, 2560, 1) && !rejects(8, 640, 1),
          "g128: the AutoRound widths accepted");
  std::printf("[ OK ] packq gemv g128/f16: oracle, row independence, NaN, contract\n");
}

DGPP_TEST(packq_gemv_nf4i8_codebook_matches_oracle) {
  // The NF4I8 codebook format (scale format 2: 4-bit indices into
  // kNf4i8Codebook, bf16 scales per 128) at the full GLM-5.3 routed widths
  // — 6144 (gate / up, 32 lanes x 6 chunks in two passes), 2048 / 1024 /
  // 512 (the down at worlds 1 / 2 / 4) — and the two test geometries 128
  // and 256. The oracle decodes every nibble through the codebook; the
  // chain is the format-0 chain with the level in place of the offset
  // code, so the budget is the same.
  struct Shape {
    int m, n, k;
  };
  const Shape shapes[] = {{1, 390, 6144}, {3, 130, 6144}, {4, 512, 6144}, {2, 300, 2048}, {3, 264, 1024},
                          {2, 100, 512},  {4, 40, 256},   {1, 33, 128},   {4, 17, 512},   {1, 520, 1024}};
  int i = 0;
  for (const Shape& s : shapes) {
    const Problem p = make_problem(4, s.m, s.n, s.k, 0xF418 + i++, dgpp::kPackedScaleBf16G128Nf4i8);
    const std::string label = "packq nf4i8 codebook gemv M" + std::to_string(s.m) + "xN" + std::to_string(s.n) +
                              "xK" + std::to_string(s.k);
    check_oracle(p, run_bf16(p), label.c_str());
  }
  // Every one of the 16 levels, including the asymmetric endpoints -127
  // and 127 and the zero at index 7, decodes exactly: a single-row
  // problem whose row r holds index r in every column against the
  // closed-form dot level[r] x scale x sum(x).
  {
    Problem p = make_problem(4, 1, 16, 128, 0x1EFE1, dgpp::kPackedScaleBf16G128Nf4i8);
    for (int r = 0; r < 16; ++r)
      for (int w = 0; w < 128 / 8; ++w) p.packed[static_cast<size_t>(r) * 16 + w] = 0x11111111u * static_cast<uint32_t>(r);
    for (auto& s : p.scales) s = dgpp::float_to_bf16_bits(1.0f);
    for (int kk = 0; kk < 128; ++kk) p.act[static_cast<size_t>(kk)] = dgpp::float_to_bf16_bits(kk % 2 ? 1.0f : 0.5f);
    const std::vector<float> got = run_f32(p);
    for (int r = 0; r < 16; ++r) {
      const float want = static_cast<float>(dgpp::kNf4i8Codebook[r]) * 96.0f;  // 64 x 1.0 + 64 x 0.5
      require(got[static_cast<size_t>(r)] == want, ("codebook level " + std::to_string(r) + " decodes exactly").c_str());
    }
  }
  // Row independence and the f32 epilogue at the production widths.
  for (int k : {512, 2048, 6144}) {
    const Problem p3 = make_problem(4, 3, 296, k, 0xE3 + k, dgpp::kPackedScaleBf16G128Nf4i8);
    const std::vector<uint16_t> got3 = run_bf16(p3);
    for (int r = 0; r < 3; ++r) {
      const std::vector<uint16_t> got1 = run_bf16(row_of(p3, r));
      require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n, static_cast<size_t>(p3.n) * 2) == 0,
              "packq nf4i8 row bits independent of m (3)");
    }
    const Problem p8 = make_problem(4, 8, 136, k, 0xE8 + k, dgpp::kPackedScaleBf16G128Nf4i8);
    const std::vector<uint16_t> got8 = run_bf16(p8);
    const std::vector<float> got8f = run_f32(p8);
    for (int r = 0; r < 8; ++r) {
      const Problem p1 = row_of(p8, r);
      const std::vector<uint16_t> got1 = run_bf16(p1);
      const std::vector<float> got1f = run_f32(p1);
      require(std::memcmp(got1.data(), got8.data() + static_cast<size_t>(r) * p8.n, static_cast<size_t>(p8.n) * 2) == 0,
              "packq nf4i8 chunked bf16 row bits independent of m (8)");
      require(std::memcmp(got1f.data(), got8f.data() + static_cast<size_t>(r) * p8.n, static_cast<size_t>(p8.n) * 4) == 0,
              "packq nf4i8 chunked f32 row bits independent of m (8)");
    }
    for (size_t j = 0; j < got8.size(); ++j)
      require(dgpp::float_to_bf16_bits(got8f[j]) == got8[j], "nf4i8 bf16(out_f32) == out_bf16");
  }
  // A NaN scale poisons exactly its row.
  {
    Problem p = make_problem(4, 1, 300, 512, 0xA5, dgpp::kPackedScaleBf16G128Nf4i8);
    p.scales[static_cast<size_t>(9) * (p.k / 128) + 3] = 0x7FC0;
    p.scales[static_cast<size_t>(250) * (p.k / 128) + 0] = 0xFFC0;
    const std::vector<uint16_t> got = run_bf16(p);
    for (int nn = 0; nn < p.n; ++nn)
      require(std::isnan(bf16_to_float(got[static_cast<size_t>(nn)])) == (nn == 9 || nn == 250),
              "nf4i8 NaN scale poisons exactly its row");
  }
  // The contract: 4-bit only, a 128-multiple in the format's compiled set.
  auto rejects = [](int bits, int k) {
    Problem p = make_problem(4, 1, 8, k % 128 == 0 ? k : 128, 0xBAD, dgpp::kPackedScaleBf16G128Nf4i8);
    p.k = k;
    p.bits = bits;
    try {
      (void)run_bf16(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects(4, 192), "nf4i8: k=192 rejected (not a multiple of 128)");
  require(rejects(4, 4096), "nf4i8: k=4096 rejected (outside the format's compiled set)");
  require(rejects(4, 2560), "nf4i8: k=2560 rejected (outside the format's compiled set)");
  require(rejects(8, 512), "nf4i8: an int8 width is rejected");
  require(!rejects(4, 512) && !rejects(4, 1024) && !rejects(4, 2048) && !rejects(4, 6144),
          "nf4i8: the routed widths accepted");
  std::printf("[ OK ] packq gemv nf4i8 codebook: oracle, every level, row independence, NaN, contract\n");
}

// ---- the Mixed346 core (packq_a8_gemv.cuh) ----------------------------------
// Weights as 3-, 4- or 6-bit codebook indices in a dense bit stream (one
// bf16 scale per 128), activations as int8 codes with one fp32 scale per
// 128 (the quantizer's output form); the oracle multiplies the represented
// values in double.
struct ProblemA8 {
  int bits, m, n, k;
  std::vector<int8_t> codes;      // [m, k]
  std::vector<float> ascales;     // [m, k/128]
  std::vector<uint32_t> packed;   // [n, k*bits/32] dense stream
  std::vector<uint16_t> scales;   // bf16 [n, k/128]
};

void pack_dense(const std::vector<unsigned>& idx, int bits, std::vector<uint32_t>& out) {
  out.assign(idx.size() * static_cast<size_t>(bits) / 32, 0u);
  for (size_t i = 0; i < idx.size(); ++i) {
    const size_t pos = i * static_cast<size_t>(bits);
    const size_t w = pos / 32;
    const int shift = static_cast<int>(pos % 32);
    out[w] |= idx[i] << shift;
    if (shift + bits > 32) out[w + 1] |= idx[i] >> (32 - shift);
  }
}

ProblemA8 make_problem_a8(int bits, int m, int n, int k, uint64_t seed) {
  ProblemA8 p;
  p.bits = bits;
  p.m = m;
  p.n = n;
  p.k = k;
  Rng rng(seed);
  p.codes.resize(static_cast<size_t>(m) * k);
  for (auto& c : p.codes) c = static_cast<int8_t>(static_cast<int>(rng.next() % 256) - 128);
  p.ascales.resize(static_cast<size_t>(m) * (k / 128));
  for (auto& s : p.ascales) s = static_cast<float>(std::exp2(rng.unit() * 2.0) * 0.01);
  std::vector<unsigned> idx(static_cast<size_t>(n) * k);
  for (auto& u : idx) u = static_cast<unsigned>(rng.next() % (1u << bits));
  pack_dense(idx, bits, p.packed);
  p.scales.resize(static_cast<size_t>(n) * (k / 128));
  // Levels reach 127 at 3 and 4 bits, 31 at 6: release-like weight magnitudes.
  for (auto& s : p.scales)
    s = dgpp::float_to_bf16_bits(static_cast<float>(std::exp2(rng.unit() * 2.0) * (bits == 6 ? 0.016 : 0.004)));
  return p;
}

std::vector<double> oracle_a8(const ProblemA8& p) {
  std::vector<double> out(static_cast<size_t>(p.m) * p.n, 0.0);
  const int groups = p.k / 128;
  for (int nn = 0; nn < p.n; ++nn)
    for (int mm = 0; mm < p.m; ++mm) {
      double acc = 0.0;
      for (int g = 0; g < groups; ++g) {
        long dot = 0;
        for (int j = 0; j < 128; ++j) {
          const int kk = g * 128 + j;
          dot += static_cast<long>(dgpp::packq_level(p.packed.data(), p.k, p.bits, nn, kk,
                                                     dgpp::kPackedScaleBf16G128Mixed346)) *
                 static_cast<long>(p.codes[static_cast<size_t>(mm) * p.k + kk]);
        }
        const double sw = dgpp::bf16_bits_to_float(p.scales[static_cast<size_t>(nn) * groups + g]);
        const double sa = p.ascales[static_cast<size_t>(mm) * groups + g];
        acc += static_cast<double>(dot) * sw * sa;
      }
      out[static_cast<size_t>(mm) * p.n + nn] = bf16_to_float(dgpp::float_to_bf16_bits(static_cast<float>(acc)));
    }
  return out;
}

struct DevA8 {
  uint32_t* packed = nullptr;
  uint16_t* scales = nullptr;
  int8_t* codes = nullptr;
  float* ascales = nullptr;
  dgpp::GlmPackedMatrix view;
  explicit DevA8(const ProblemA8& p) {
    DGPP_CUDA_OK(cudaMallocManaged(&packed, p.packed.size() * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&scales, p.scales.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&codes, p.codes.size()));
    DGPP_CUDA_OK(cudaMallocManaged(&ascales, p.ascales.size() * 4));
    std::memcpy(packed, p.packed.data(), p.packed.size() * 4);
    std::memcpy(scales, p.scales.data(), p.scales.size() * 2);
    std::memcpy(codes, p.codes.data(), p.codes.size());
    std::memcpy(ascales, p.ascales.data(), p.ascales.size() * 4);
    view = dgpp::GlmPackedMatrix{packed, scales, p.n, p.k, p.bits, dgpp::kPackedScaleBf16G128Mixed346};
  }
  ~DevA8() {
    cudaFree(packed);
    cudaFree(scales);
    cudaFree(codes);
    cudaFree(ascales);
  }
};

std::vector<uint16_t> run_a8_bf16(const ProblemA8& p) {
  DevA8 d(p);
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  dgpp::launch_packq_a8_gemv_bf16(d.codes, p.k, d.ascales, p.k / 128, d.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  cudaFree(out);
  return got;
}
std::vector<float> run_a8_f32(const ProblemA8& p) {
  DevA8 d(p);
  float* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
  dgpp::launch_packq_a8_gemv_f32(d.codes, p.k, d.ascales, p.k / 128, d.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 4);
  cudaFree(out);
  return got;
}

DGPP_TEST(packq_a8_mixed346_core_matches_oracle) {
  // Every width at the full GLM-5.3 routed widths (6144 gate / up, 2048 /
  // 1024 / 512 the down at worlds 1 / 2 / 4) and the test geometries; the
  // group dots are exact integers, so the budget is the fp32 accumulation's.
  struct Shape { int m, n, k; };
  const Shape shapes[] = {{1, 48, 128}, {3, 40, 256}, {2, 64, 512}, {4, 33, 1024}, {1, 24, 2048}, {2, 20, 6144}};
  int i = 0;
  for (const int bits : {3, 4, 6})
    for (const Shape& s : shapes) {
      const ProblemA8 p = make_problem_a8(bits, s.m, s.n, s.k, 0xA8000 + 37 * i++);
      const auto got = run_a8_bf16(p);
      const auto want = oracle_a8(p);
      const auto rep = compare_bf16_vs_oracle(got.data(), want, /*ulp_budget=*/2.0, /*floor_frac=*/1e-3);
      const std::string label = "packq a8 " + std::to_string(bits) + "-bit M" + std::to_string(s.m) + "xN" +
                                std::to_string(s.n) + "xK" + std::to_string(s.k);
      std::printf("[ OK ] %s: l2_rel=%.3g mismatches=%ld/%zu\n", label.c_str(), rep.l2_rel,
                  static_cast<long>(rep.mismatches), want.size());
      require_report(rep, 1e-3, 0, label.c_str());
    }
  // Every level of every width decodes exactly: one row of codes 1 under
  // scale 1 against a matrix whose row r is level r everywhere (scale 1).
  for (const int bits : {3, 4, 6}) {
    const int levels = 1 << bits;
    ProblemA8 p = make_problem_a8(bits, 1, levels, 128, 0x1E5E1 + bits);
    std::vector<unsigned> idx(static_cast<size_t>(levels) * 128);
    for (int r = 0; r < levels; ++r)
      for (int j = 0; j < 128; ++j) idx[static_cast<size_t>(r) * 128 + j] = static_cast<unsigned>(r);
    pack_dense(idx, bits, p.packed);
    for (auto& s : p.scales) s = dgpp::float_to_bf16_bits(1.0f);
    for (auto& c : p.codes) c = 1;
    for (auto& s : p.ascales) s = 1.0f;
    const auto got = run_a8_f32(p);
    for (int r = 0; r < levels; ++r) {
      const float want = 128.0f * static_cast<float>(dgpp::packed_code_level(static_cast<unsigned>(r), bits,
                                                                              dgpp::kPackedScaleBf16G128Mixed346));
      require(got[static_cast<size_t>(r)] == want, ("a8 " + std::to_string(bits) + "-bit level " + std::to_string(r) +
                                                     " decodes exactly").c_str());
    }
  }
  // Row independence: a row's bits do not depend on m (the chunking) or
  // on the output dtype.
  for (const int bits : {3, 4, 6}) {
    const int k = 512;
    const ProblemA8 p3 = make_problem_a8(bits, 3, 296, k, 0xE3 + bits);
    const ProblemA8 p1{bits, 1, 296, k, std::vector<int8_t>(p3.codes.begin(), p3.codes.begin() + k),
                       std::vector<float>(p3.ascales.begin(), p3.ascales.begin() + k / 128), p3.packed, p3.scales};
    const auto got3 = run_a8_bf16(p3), got1 = run_a8_bf16(p1);
    require(std::equal(got1.begin(), got1.end(), got3.begin()), "a8 row bits independent of m (3)");
    const ProblemA8 p8 = make_problem_a8(bits, 8, 136, k, 0xE8 + bits);
    const ProblemA8 p8a{bits, 1, 136, k, std::vector<int8_t>(p8.codes.begin() + 5 * k, p8.codes.begin() + 6 * k),
                        std::vector<float>(p8.ascales.begin() + 5 * (k / 128), p8.ascales.begin() + 6 * (k / 128)),
                        p8.packed, p8.scales};
    const auto got8 = run_a8_bf16(p8), got8a = run_a8_bf16(p8a);
    require(std::equal(got8a.begin(), got8a.end(), got8.begin() + 5 * 136), "a8 chunked row bits independent of m (8)");
    const auto got8f = run_a8_f32(p8);
    for (size_t j = 0; j < got8.size(); ++j)
      require(dgpp::float_to_bf16_bits(got8f[j]) == got8[j], "a8 bf16(out_f32) == out_bf16");
  }
  {
    // A NaN weight scale poisons exactly its row.
    ProblemA8 p = make_problem_a8(4, 1, 300, 512, 0xA5);
    p.scales[7 * (512 / 128) + 2] = 0x7FC0;
    const auto got = run_a8_bf16(p);
    bool ok = true;
    for (int nn = 0; nn < p.n; ++nn) ok = ok && (((got[static_cast<size_t>(nn)] & 0x7FFF) > 0x7F80) == (nn == 7));
    require(ok, "a8 NaN scale poisons exactly its row");
  }
  {
    // The contract: format 3 at 3 / 4 / 6 bits, k a multiple of 128 in the compiled set.
    auto rejects = [](int bits, int k, int fmt = dgpp::kPackedScaleBf16G128Mixed346) {
      dgpp::GlmPackedMatrix w;
      w.bits = bits;
      w.scale_fmt = fmt;
      w.cols = k;
      w.rows = 8;
      alignas(16) static uint32_t words[64];
      static uint16_t sc[64];
      w.packed = words;
      w.scales = sc;
      return !dgpp::packq_a8_gemv_accepts(w);
    };
    require(rejects(8, 512) && rejects(5, 512) && rejects(4, 192) && rejects(4, 4096) &&
                rejects(4, 512, dgpp::kPackedScaleBf16G128Nf4i8),
            "a8 contract: widths, k and format");
    require(!rejects(3, 6144) && !rejects(4, 2048) && !rejects(6, 512), "a8 contract: the routed widths accepted");
  }
  std::printf("[ OK ] packq a8 mixed346 core: oracle, every level, row independence, NaN, contract\n");
}

DGPP_TEST(hadamard32_quant_int8_kernel_matches_host_reference) {
  // The Mixed346 activation quantizer: H32 then the group codes and scales,
  // bitwise the host reference, the skip period leaving rows unwritten.
  Rng rng(0xA8A8);
  for (const int k : {128, 512, 6144}) {
    const int rows = 11, period = 4;
    const size_t stride = static_cast<size_t>(k);
    std::vector<uint16_t> in(static_cast<size_t>(rows) * stride);
    fill_act(rng, in);
    // A few large and tiny groups: the scale's floor and the clamp.
    for (int j = 0; j < 128 && k >= 256; ++j) in[128 + j] = dgpp::float_to_bf16_bits(j == 5 ? 300.0f : 0.001f * j);
    for (int j = 0; j < 128; ++j) in[static_cast<size_t>(3) * stride + j] = 0;
    std::vector<int8_t> want(in.size());
    std::vector<float> want_s(static_cast<size_t>(rows) * (k / 128));
    dgpp::hadamard32_quant_int8_rows_host(in.data(), stride, want.data(), stride, want_s.data(), k / 128, rows, k);
    uint16_t* d_in = nullptr;
    int8_t* d_codes = nullptr;
    float* d_s = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_in, in.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_codes, in.size()));
    DGPP_CUDA_OK(cudaMallocManaged(&d_s, want_s.size() * 4));
    std::memcpy(d_in, in.data(), in.size() * 2);
    std::memset(d_codes, 0x55, in.size());
    for (size_t i = 0; i < want_s.size(); ++i) d_s[i] = -1.0f;
    dgpp::launch_hadamard32_quant_int8_rows(d_in, stride, d_codes, stride, d_s, k / 128, rows, k, 0, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_codes, want.data(), want.size()) == 0, "quant codes bitwise the host reference");
    require(std::memcmp(d_s, want_s.data(), want_s.size() * 4) == 0, "quant scales bitwise the host reference");
    // The zero row: scale 1e-30, codes 0.
    require(d_s[3 * (k / 128)] == 1e-30f && d_codes[3 * stride] == 0, "a zero group takes the scale floor");
    std::memset(d_codes, 0x55, in.size());
    for (size_t i = 0; i < want_s.size(); ++i) d_s[i] = -1.0f;
    dgpp::launch_hadamard32_quant_int8_rows(d_in, stride, d_codes, stride, d_s, k / 128, rows, k, period, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    bool ok = true;
    for (int r = 0; r < rows; ++r) {
      const bool skipped = r % period == period - 1;
      for (int c = 0; c < k; ++c)
        ok = ok && (skipped ? d_codes[r * stride + c] == 0x55 : d_codes[r * stride + c] == want[r * stride + c]);
      for (int g = 0; g < k / 128; ++g)
        ok = ok && (skipped ? d_s[r * (k / 128) + g] == -1.0f : d_s[r * (k / 128) + g] == want_s[r * (k / 128) + g]);
    }
    require(ok, "quant skip period leaves the shared rows unwritten");
    cudaFree(d_in);
    cudaFree(d_codes);
    cudaFree(d_s);
  }
  std::printf("[ OK ] hadamard32 int8 quantizer: bitwise the host reference, the skip period\n");
}

// The post-attention norm with the quantizer fused (glm_rmsnorm_bf16_quant_int8)
// against the norm then the quantizer: y, the codes and the scales bitwise.
DGPP_TEST(glm_rmsnorm_quant_int8_matches_norm_then_quantizer) {
  std::mt19937 rng(0x346u);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (const int dim : {128, 512, 6144}) {
    const int rows = 5;
    std::vector<uint16_t> x(static_cast<size_t>(rows) * dim), w(dim);
    for (auto& v : x) v = dgpp::float_to_bf16_bits(nd(rng) * 3.f);
    for (auto& v : w) v = dgpp::float_to_bf16_bits(1.f + 0.1f * nd(rng));
    uint16_t *d_x = nullptr, *d_w = nullptr, *d_y = nullptr, *d_y2 = nullptr;
    int8_t *d_c = nullptr, *d_c2 = nullptr;
    float *d_s = nullptr, *d_s2 = nullptr;
    const size_t groups = static_cast<size_t>(dim / 128);
    DGPP_CUDA_OK(cudaMalloc(&d_x, x.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_w, w.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_y, x.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_y2, x.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_c, x.size()));
    DGPP_CUDA_OK(cudaMalloc(&d_c2, x.size()));
    DGPP_CUDA_OK(cudaMalloc(&d_s, rows * groups * 4));
    DGPP_CUDA_OK(cudaMalloc(&d_s2, rows * groups * 4));
    DGPP_CUDA_OK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_w, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    dgpp::glm_rmsnorm_bf16(d_x, d_w, d_y, rows, dim, 1e-5f, nullptr);
    dgpp::launch_hadamard32_quant_int8_rows(d_y, static_cast<size_t>(dim), d_c, static_cast<size_t>(dim), d_s, groups,
                                            rows, dim, 0, nullptr);
    dgpp::glm_rmsnorm_bf16_quant_int8(d_x, d_w, d_y2, d_c2, static_cast<size_t>(dim), d_s2, groups, rows, dim, 1e-5f,
                                      nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> y(x.size()), y2(x.size());
    std::vector<int8_t> c(x.size()), c2(x.size());
    std::vector<float> sa(rows * groups), sb(rows * groups);
    DGPP_CUDA_OK(cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(y2.data(), d_y2, y2.size() * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(c.data(), d_c, c.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(c2.data(), d_c2, c2.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(sa.data(), d_s, sa.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(sb.data(), d_s2, sb.size() * 4, cudaMemcpyDeviceToHost));
    require(y == y2, "fused norm: y bitwise the norm");
    require(c == c2, "fused norm: codes bitwise the quantizer");
    require(std::memcmp(sa.data(), sb.data(), sa.size() * 4) == 0, "fused norm: scales bitwise the quantizer");
    for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_w), static_cast<void*>(d_y), static_cast<void*>(d_y2),
                    static_cast<void*>(d_c), static_cast<void*>(d_c2), static_cast<void*>(d_s), static_cast<void*>(d_s2)})
      DGPP_CUDA_OK(cudaFree(p));
  }
  std::printf("[ OK ] fused norm + int8 quantizer: y, codes and scales bitwise the norm then the quantizer\n");
}

DGPP_TEST(hadamard32_rows_kernel_matches_host_reference) {
  // The device rotation is the host reference bit for bit (the oracles'
  // arithmetic) at the routed widths, with strides and in place; applied
  // twice it returns the input to within the two bf16 roundings
  // (H32 / sqrt(32) is an involution).
  for (int k : {512, 6144}) {
    const int rows = 13;
    const size_t stride = static_cast<size_t>(k) + 32;
    std::vector<uint16_t> in(rows * stride), want(rows * stride, 0), got(rows * stride, 0);
    Rng rng(0x4A32 + k);
    fill_act(rng, in);
    dgpp::hadamard32_rows_host(in.data(), stride, want.data(), stride, rows, k);
    uint16_t *d_in = nullptr, *d_out = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_in, in.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_out, in.size() * 2));
    std::memcpy(d_in, in.data(), in.size() * 2);
    std::memset(d_out, 0, in.size() * 2);
    dgpp::launch_hadamard32_rows(d_in, stride, d_out, stride, rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(got.data(), d_out, got.size() * 2);
    require(got == want, ("hadamard32 rows: device bitwise the host reference (k=" + std::to_string(k) + ")").c_str());
    // In place (the pad columns past k keep the input: compare the rows).
    dgpp::launch_hadamard32_rows(d_in, stride, d_in, stride, rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int r = 0; r < rows; ++r) {
      require(std::memcmp(d_in + r * stride, want.data() + r * stride, static_cast<size_t>(k) * 2) == 0,
              "hadamard32 rows in place");
      require(std::memcmp(d_in + r * stride + k, in.data() + r * stride + k, 32 * 2) == 0,
              "hadamard32 rows in place leaves the pad columns alone");
    }
    // Twice: back to the input within two bf16 roundings.
    dgpp::launch_hadamard32_rows(d_in, stride, d_out, stride, rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < k; ++c) {
        const float a = bf16_to_float(in[r * stride + c]), b = bf16_to_float(d_out[r * stride + c]);
        require(std::fabs(a - b) <= 0.02f * std::fabs(a) + 0.02f, "H32 twice returns the input (involution)");
      }
    cudaFree(d_in);
    cudaFree(d_out);
  }
  // The skipping form (the decode slot layout: every period-th row, the
  // shared slot's, left as it is).
  {
    const int rows = 9, k = 64, period = 3;
    std::vector<uint16_t> in(static_cast<size_t>(rows) * k), want(in.size());
    for (size_t i = 0; i < in.size(); ++i) in[i] = dgpp::float_to_bf16_bits(static_cast<float>(static_cast<int>(i * 37 % 61) - 30) / 7.f);
    dgpp::hadamard32_rows_host(in.data(), k, want.data(), k, rows, k);
    for (int r = period - 1; r < rows; r += period)
      std::copy(in.begin() + r * k, in.begin() + (r + 1) * k, want.begin() + r * k);
    uint16_t* d = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d, in.size() * 2));
    std::copy(in.begin(), in.end(), d);
    dgpp::launch_hadamard32_rows_skip(d, k, d, k, rows, k, period, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::equal(want.begin(), want.end(), d), "hadamard32 rows with a period: rotated rows bitwise the host, the period's rows untouched");
    cudaFree(d);
  }
  // The contract.
  bool refused = false;
  try {
    uint16_t* p = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&p, 64 * 2));
    dgpp::launch_hadamard32_rows(p, 48, p, 48, 1, 48, nullptr);
    cudaFree(p);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "k must be a multiple of 32");
  std::printf("[ OK ] hadamard32 rows: device == host reference, in place, involution, contract\n");
}

DGPP_TEST(packq_gemv_rows_are_independent_of_row_count) {
  // Each row's chain is the same sequence of FMAs whatever m: m=3 against
  // three m=1 runs, and the m=8 chunked launch against eight singles, for
  // both epilogues, both widths and the production geometries.
  for (int bits : {4, 8})
    for (int k : {6144, 512, 2048, 192}) {
      const Problem p3 = make_problem(bits, 3, 296, k, 0xE3 + k + bits);
      const std::vector<uint16_t> got3 = run_bf16(p3);
      for (int r = 0; r < 3; ++r) {
        const std::vector<uint16_t> got1 = run_bf16(row_of(p3, r));
        require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                            static_cast<size_t>(p3.n) * 2) == 0,
                "packq row bits independent of m (3)");
      }
      const Problem p8 = make_problem(bits, 8, 136, k, 0xE8 + k + bits);
      const std::vector<uint16_t> got8 = run_bf16(p8);
      const std::vector<float> got8f = run_f32(p8);
      for (int r = 0; r < 8; ++r) {
        const Problem p1 = row_of(p8, r);
        const std::vector<uint16_t> got1 = run_bf16(p1);
        const std::vector<float> got1f = run_f32(p1);
        require(std::memcmp(got1.data(), got8.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 2) == 0,
                "packq chunked bf16 row bits independent of m (8)");
        require(std::memcmp(got1f.data(), got8f.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 4) == 0,
                "packq chunked f32 row bits independent of m (8)");
      }
    }
  std::printf("[ OK ] packq gemv rows independent of m through m=8 chunks\n");
}

DGPP_TEST(packq_gemv_f32_epilogue_is_the_unrounded_bf16) {
  for (int bits : {4, 8}) {
    const Problem p = make_problem(bits, 4, 200, 1024, 0xF32 + bits);
    const std::vector<uint16_t> b = run_bf16(p);
    const std::vector<float> f = run_f32(p);
    for (size_t i = 0; i < b.size(); ++i)
      require(dgpp::float_to_bf16_bits(f[i]) == b[i], "bf16(out_f32) == out_bf16");
  }
  std::printf("[ OK ] packq gemv f32 epilogue rounds to the bf16 epilogue\n");
}

DGPP_TEST(packq_gemv_propagates_nan_scales_exactly) {
  // Two poisoned scales: exactly their rows' outputs NaN.
  for (int bits : {4, 8}) {
    Problem p = make_problem(bits, 1, 300, 1024, 0xA5 + bits);
    p.scales[static_cast<size_t>(9) * (p.k / 64) + 3] = 0x7FC0;
    p.scales[static_cast<size_t>(250) * (p.k / 64) + 0] = 0xFFC0;
    const std::vector<uint16_t> got = run_bf16(p);
    for (int nn = 0; nn < p.n; ++nn) {
      const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
      if (nn == 9 || nn == 250)
        require(got_nan, "poisoned scale row is NaN");
      else
        require(!got_nan, "unpoisoned row stays finite");
    }
  }
  std::printf("[ OK ] packq gemv nan propagation: rows 9 and 250 NaN\n");
}

DGPP_TEST(packq_gemv_rejects_geometry_outside_the_contract) {
  auto rejects = [](int bits, int k) {
    Problem p = make_problem(bits, 1, 8, k, 0xBAD);
    if (bits != 4 && bits != 8) {
      p.bits = bits;  // an unsupported width on a valid int8 layout
    }
    try {
      (void)run_bf16(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects(4, 96), "k=96 rejected (not a multiple of 64)");
  require(rejects(4, 320), "k=320 rejected (not in the compiled set)");
  require(rejects(8, 320), "int8 k=320 rejected (not in the compiled set)");
  require(!rejects(4, 6144), "int4 k=6144 accepted");
  require(!rejects(8, 6144), "int8 k=6144 accepted");
  require(!rejects(4, 16384), "int4 k=16384 accepted (32 lanes x 16 chunks)");
  require(!rejects(8, 16384), "int8 k=16384 accepted (32 lanes x 32 chunks)");
  require(rejects(4, 32768), "k=32768 rejected (not in the compiled set)");
  Problem p = make_problem(8, 1, 8, 1024, 0xBAD);
  p.bits = 3;
  bool bad_width = false;
  try {
    (void)run_bf16(p);
  } catch (const std::invalid_argument&) {
    bad_width = true;
  }
  require(bad_width, "a 3-bit width is rejected");
  std::printf("[ OK ] packq gemv geometry contract enforced\n");
}

// Real-checkpoint slice parity lives in packq_gemv_checkpoint.cpp (host-only
// TU: the safetensors reader's JSON parser does not mix with nvcc).
int run_packq_gemv_checkpoint_parity(const char* checkpoint_dir);

int main(int argc, char** argv) {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
      const int prc = run_packq_gemv_checkpoint_parity(argv[i + 1]);
      if (prc != 0) return prc;
    }
  }
  return 0;
}
