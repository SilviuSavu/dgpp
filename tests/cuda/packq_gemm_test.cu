// Independent FP64 oracle for exact offset-code x BF16-scale weights.
// Also pin grouped/dense arithmetic, row maps, ragged tiles, output strides,
// BF16 epilogues, NaN propagation and cold CUDA graph capture.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_a8_gemv.hpp"
#include "kernels/packq_gemv.hpp"
#include "loaders/packq_quant.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {
using scale_gemm_test::require;
template <typename T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { cudaFree(p); }
};

struct Matrix {
  int bits, n, k, sf;
  Buffer<uint32_t> packed;
  Buffer<uint16_t> scales;
  int group() const { return dgpp::packed_scale_group(sf); }
  Matrix(int bits, int n, int k, uint64_t seed, int sf = 0)
      : bits(bits),
        n(n),
        k(k),
        sf(sf),
        packed(static_cast<size_t>(n) * k * bits / 32),
        scales(static_cast<size_t>(n) * k / dgpp::packed_scale_group(sf)) {
    scale_gemm_test::Rng rng(seed);
    for (size_t i = 0; i < static_cast<size_t>(n) * k * bits / 32; ++i)
      packed.p[i] = static_cast<uint32_t>(rng.next());
    for (size_t i = 0; i < static_cast<size_t>(n) * k / group(); ++i) {
      const float v = static_cast<float>(std::exp2(rng.unit() * 4) * (dgpp::packed_codebook(sf) ? 0.0031 / 16 : 0.0031));
      scales.p[i] = sf == 1 ? dgpp::float_to_fp16_bits(v) : dgpp::float_to_bf16_bits(v);
    }
    scales.p[0] = 0;  // zero scales must also have exact semantics
  }
  dgpp::GlmPackedMatrix view() const { return {packed.p, scales.p, n, k, bits, sf}; }
  double value(int row, int col) const {
    const int per = 32 / bits;
    const int g = group();
    const uint32_t w = packed.p[static_cast<size_t>(row) * (k / per) + col / per];
    const int code = dgpp::packed_code_level((w >> (bits * (col % per))) & ((1u << bits) - 1), bits, sf);
    return code * static_cast<double>(dgpp::packed_scale_to_float(
                      scales.p[static_cast<size_t>(row) * (k / g) + col / g], sf));
  }
};

void fill(uint16_t* a, size_t count, uint64_t seed) {
  scale_gemm_test::Rng rng(seed);
  for (size_t i = 0; i < count; ++i)
    a[i] = dgpp::float_to_bf16_bits(static_cast<float>(rng.unit() * std::exp2(rng.unit() * 3)));
}

void oracle_check(const Matrix& w, const uint16_t* a, int stride, const float* got, int os, int m,
                  const float* baseline = nullptr) {
  double error2 = 0, norm2 = 0, max_abs = 0, max_error = 0;
  double baseline_error2 = 0, bf16_error2 = 0, baseline_bf16_error2 = 0;
  int better = 0, worse = 0, changed_bf16 = 0;
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < w.n; ++col) {
      double expected = 0;
      for (int kk = 0; kk < w.k; ++kk)
        expected +=
            dgpp::bf16_bits_to_float(a[static_cast<size_t>(row) * stride + kk]) * w.value(col, kk);
      const double actual = got[static_cast<size_t>(row) * os + col];
      if (std::isnan(expected)) {
        require(std::isnan(actual), "NaN scale must poison exactly its output column");
        continue;
      }
      require(std::isfinite(actual), "finite oracle must produce finite output");
      const double e = std::abs(actual - expected);
      error2 += e * e;
      norm2 += expected * expected;
      max_abs = std::max(max_abs, std::abs(expected));
      max_error = std::max(max_error, e);
      if (baseline) {
        const double old = baseline[static_cast<size_t>(row) * w.n + col];
        require(std::isfinite(old), "finite oracle must produce finite GEMV output");
        baseline_error2 += (old - expected) * (old - expected);
        const uint16_t b = dgpp::float_to_bf16_bits(actual);
        const uint16_t ob = dgpp::float_to_bf16_bits(old);
        const double be = dgpp::bf16_bits_to_float(b) - expected;
        const double obe = dgpp::bf16_bits_to_float(ob) - expected;
        bf16_error2 += be * be;
        baseline_bf16_error2 += obe * obe;
        better += std::abs(be) < std::abs(obe);
        worse += std::abs(be) > std::abs(obe);
        changed_bf16 += b != ob;
      }
    }
  const double relative = std::sqrt(error2 / std::max(norm2, 1e-30));
  std::printf("int%d%s M%d N%d K%d: fp32 l2_rel %.3g max_error/max_abs %.3g\n", w.bits,
              w.sf == 1 ? " g128/f16" : w.sf == 2 ? " nf4i8" : "", m, w.n, w.k, relative,
              max_error / std::max(max_abs, 1e-30));
  require(relative < 5e-6 && max_error < std::max(1e-8, max_abs * 4e-5),
          "packed GEMM differs from exact FP64 oracle");
  if (baseline) {
    const double old_relative = std::sqrt(baseline_error2 / std::max(norm2, 1e-30));
    std::printf(
        "  GEMV fp32 l2_rel %.3g; BF16 vs FP64 GEMV %.9g GEMM %.9g; "
        "changed %d, GEMM closer %d farther %d (remaining changes equidistant)\n",
        old_relative, std::sqrt(baseline_bf16_error2 / std::max(norm2, 1e-30)),
        std::sqrt(bf16_error2 / std::max(norm2, 1e-30)), changed_bf16, better, worse);
    require(old_relative < 5e-6, "packed GEMV differs from exact FP64 oracle");
  }
}
}  // namespace

DGPP_TEST(packq_gemm_exact_weights_and_ragged_tiles) {
  // Format 1 (f16 per 128) at the AutoRound hybrid's widths and the small
  // 128-multiples: the tile's scale is the group of its 64-deep step.
  // Format 2 (the NF4I8 codebook, 4-bit) at the full GLM-5.3 routed
  // widths and the two test geometries: the tensor core takes the level.
  const std::vector<std::vector<int>> shapes0 = {{1, 1, 64},     {17, 70, 192},  {33, 129, 512},
                                                 {65, 35, 6144}, {32, 67, 2048}, {3, 17, 16384}};
  const std::vector<std::vector<int>> shapes1 = {{1, 1, 128},   {17, 70, 640},   {33, 129, 2560},
                                                 {32, 67, 256}, {65, 35, 1024},  {3, 17, 2560}};
  const std::vector<std::vector<int>> shapes2 = {{1, 1, 128},   {17, 70, 512},   {33, 129, 1024},
                                                 {32, 67, 256}, {65, 35, 6144},  {3, 17, 2048}};
  for (int sf : {0, 1, 2})
  for (int bits : {4, 8})
    for (const auto& shape : sf == 0 ? shapes0 : sf == 1 ? shapes1 : shapes2) {
      if (sf == 2 && bits != 4) continue;
      const int m = shape[0], n = shape[1], k = shape[2];
      Matrix w(bits, n, k, 917 + bits + k, sf);
      const int stride = k + (m % 2 ? 3 : 0);
      Buffer<uint16_t> storage(static_cast<size_t>(m) * stride + 1), b(static_cast<size_t>(m) * n);
      uint16_t* a = storage.p + 1;  // explicitly unaligned base
      fill(a, static_cast<size_t>(m) * stride, 519);
      Buffer<float> out(static_cast<size_t>(m) * n), baseline(static_cast<size_t>(m) * n);
      Buffer<uint16_t> aligned(static_cast<size_t>(m) * k);
      for (int row = 0; row < m; ++row)
        std::memcpy(aligned.p + static_cast<size_t>(row) * k, a + static_cast<size_t>(row) * stride,
                    k * sizeof(uint16_t));
      dgpp::launch_packq_gemm_f32(a, stride, w.view(), out.p, m, n, k, nullptr);
      dgpp::launch_packq_gemm_bf16(a, stride, w.view(), b.p, m, n, k, nullptr);
      dgpp::launch_packq_gemv_f32(aligned.p, k, w.view(), baseline.p, m, n, k, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      oracle_check(w, a, stride, out.p, n, m, baseline.p);
      for (int i = 0; i < m * n; ++i)
        require(b.p[i] == dgpp::float_to_bf16_bits(out.p[i]),
                "BF16 epilogue must round FP32 output exactly");
    }
}

DGPP_TEST(packq_gemm_grouped_maps_padding_and_graph) {
  constexpr int n = 70, os = 77, tokens = 137, ns = 9;
  for (int sf : {0, 1, 2})
  for (int bits : {4, 8}) {
    if (sf == 2 && bits != 4) continue;
    const int k = sf == 0 ? 192 : sf == 1 ? 640 : 512;
    Matrix w(bits, n, k, 611 + bits, sf);
    w.scales.p[9 * (k / w.group()) + 1] = sf == 1 ? 0x7e00 : 0x7fc0;
    Matrix other(bits, n, k, 975 + bits, sf);
    Buffer<dgpp::MoeExpertView> views(ns * 3);
    Buffer<dgpp::MoeSegment> segs(ns);
    const int counts[ns] = {0, 1, 16, 17, 31, 32, 33, 65, 130};
    int total = 0;
    for (int e = 0; e < ns; ++e) {
      segs.p[e] = {total, counts[e], e};
      total += counts[e];
      for (int which = 0; which < 3; ++which)
        views.p[e * 3 + which] = dgpp::MoeExpertView::of(which == 1 ? w.view() : other.view());
    }
    Buffer<int32_t> rows(total);
    Buffer<uint16_t> a(tokens * k), gathered(static_cast<size_t>(total) * k);
    fill(a.p, tokens * k, 617);
    for (int r = 0; r < total; ++r) {
      rows.p[r] = (r * 19) % tokens;  // duplicates and nonmonotonic input rows
      std::memcpy(gathered.p + static_cast<size_t>(r) * k, a.p + rows.p[r] * k, k * 2);
    }
    Buffer<float> out(static_cast<size_t>(total) * os), ref(static_cast<size_t>(total) * n);
    Buffer<uint16_t> bf(static_cast<size_t>(total) * os);
    std::fill(out.p, out.p + static_cast<size_t>(total) * os, -12345.f);
    std::fill(bf.p, bf.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, tokens, views.p, 1, out.p, os, n, k,
                                           bits, stream, rows.p, sf);
    dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, tokens, views.p, 1, bf.p, os, n, k,
                                            bits, stream, rows.p, sf);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 2; ++repeat) DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    dgpp::launch_packq_gemm_f32(gathered.p, k, w.view(), ref.p, total, n, k, stream);
    // The compact tile list (glm_moe_launch.hpp): the same tiles from a 1-D
    // grid, bitwise the segment-major launch; its entries as the host
    // expects them (one 64-row tile per started 64 rows, segment order).
    const int tile_cap = dgpp::moe_tile_list_capacity(ns, total, dgpp::kPackqGemmWideRows);
    Buffer<dgpp::MoeTile> tiles(tile_cap);
    Buffer<int32_t> tile_count(1);
    Buffer<float> listed(static_cast<size_t>(total) * os);
    std::fill(listed.p, listed.p + static_cast<size_t>(total) * os, -12345.f);
    dgpp::launch_moe_tile_list(segs.p, ns, dgpp::kPackqGemmWideRows, tiles.p, tile_count.p, stream);
    if (bits == 4)  // the wide kernel takes the list; int8 rows keep the narrow grid
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, /*max_rows=*/1, views.p, 1, listed.p, os,
                                             n, k, bits, stream, rows.p, sf, /*variant=*/1, tiles.p,
                                             tile_count.p, tile_cap);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    {
      int want = 0;
      for (int e = 0; e < ns; ++e) {
        for (int j = 0; j < counts[e]; j += dgpp::kPackqGemmWideRows) {
          require(want < tile_cap, "tile list within its capacity");
          require(tiles.p[want].seg == e && tiles.p[want].m0 == j, "tile list entry");
          ++want;
        }
      }
      require(tile_count.p[0] == want, "tile list count");
      if (bits == 4)
        require(std::memcmp(listed.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
                "the listed launch must be bitwise the segment-major one");
    }
    if (bits == 4) {
      // The paired launch: projection 1 (w) and projection 0 (other) in one
      // launch, each bitwise its own launch.
      Buffer<uint16_t> pa(static_cast<size_t>(total) * os), pb(static_cast<size_t>(total) * os);
      Buffer<uint16_t> sb(static_cast<size_t>(total) * os);
      std::fill(pa.p, pa.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
      std::fill(pb.p, pb.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
      std::fill(sb.p, sb.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
      dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, tokens, views.p, 1, pa.p, os, n, k, bits,
                                              stream, rows.p, sf, /*variant=*/1, tiles.p, tile_count.p, tile_cap,
                                              pb.p, /*which2=*/0);
      dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, tokens, views.p, 0, sb.p, os, n, k, bits,
                                              stream, rows.p, sf, /*variant=*/1);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(std::memcmp(pa.p, bf.p, static_cast<size_t>(total) * os * 2) == 0,
              "the paired launch's first projection must be bitwise its own launch");
      require(std::memcmp(pb.p, sb.p, static_cast<size_t>(total) * os * 2) == 0,
              "the paired launch's second projection must be bitwise its own launch");
    }
    if (bits == 4) {
      // The register-decode three-stage form (variant 2), listed and
      // segment-major: bitwise the decoded-tile kernel.
      Buffer<float> reg(static_cast<size_t>(total) * os);
      std::fill(reg.p, reg.p + static_cast<size_t>(total) * os, -12345.f);
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, /*max_rows=*/1, views.p, 1, reg.p, os, n, k,
                                             bits, stream, rows.p, sf, /*variant=*/2, tiles.p, tile_count.p,
                                             tile_cap);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(std::memcmp(reg.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
              "the register-decode kernel must be bitwise the decoded-tile one (listed)");
      std::fill(reg.p, reg.p + static_cast<size_t>(total) * os, -12345.f);
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, tokens, views.p, 1, reg.p, os, n, k, bits,
                                             stream, rows.p, sf, /*variant=*/2);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(std::memcmp(reg.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
              "the register-decode kernel must be bitwise the decoded-tile one (segment-major)");
      // The four-warp forms (variants 4 and 5, 2026-09-30: 64 x 32 warp
      // tiles, decoded-tile and register-decode), listed and segment-major:
      // bitwise the eight-warp kernel.
      for (const int wide4 : {4, 5}) {
        std::fill(reg.p, reg.p + static_cast<size_t>(total) * os, -12345.f);
        dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, /*max_rows=*/1, views.p, 1, reg.p, os, n, k,
                                               bits, stream, rows.p, sf, wide4, tiles.p, tile_count.p, tile_cap);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        require(std::memcmp(reg.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
                "the four-warp kernel must be bitwise the eight-warp one (listed)");
        std::fill(reg.p, reg.p + static_cast<size_t>(total) * os, -12345.f);
        dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, tokens, views.p, 1, reg.p, os, n, k, bits,
                                               stream, rows.p, sf, wide4);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        require(std::memcmp(reg.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
                "the four-warp kernel must be bitwise the eight-warp one (segment-major)");
      }
      // The folded-scale form (variant 3, engine.prefill_fold_scales): NOT
      // bitwise — each weight x scale rounded to bf16 and one accumulator —
      // but within the packed model's tolerance of the exact chain: the
      // relative RMS distance under 2^-7 (bf16's own rounding is 2^-9 a
      // value; the sum of k of them over the exact chain's fp32 stays
      // well inside), and every finite output finite.
      Buffer<float> fold(static_cast<size_t>(total) * os);
      std::fill(fold.p, fold.p + static_cast<size_t>(total) * os, -12345.f);
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, /*max_rows=*/1, views.p, 1, fold.p, os, n, k,
                                             bits, stream, rows.p, sf, /*variant=*/3, tiles.p, tile_count.p,
                                             tile_cap);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      double num = 0, den = 0;
      bool any_diff = false;
      for (int row = 0; row < total; ++row)
        for (int col = 0; col < n; ++col) {
          const double exact = out.p[static_cast<size_t>(row) * os + col];
          const double got = fold.p[static_cast<size_t>(row) * os + col];
          require(std::isfinite(got) == std::isfinite(exact), "the folded kernel's finiteness follows the exact chain");
          if (!std::isfinite(exact)) continue;
          num += (got - exact) * (got - exact);
          den += exact * exact;
          any_diff |= got != exact;
        }
      require(den > 0 && std::sqrt(num / den) < 1.0 / 128,
              "the folded-scale kernel must stay within 2^-7 relative RMS of the exact chain");
      (void)any_diff;  // it need not differ on every fixture; it must never be far
    }
    oracle_check(w, gathered.p, k, out.p, os, total);
    for (int row = 0; row < total; ++row) {
      require(std::memcmp(out.p + row * os, ref.p + row * n, n * 4) == 0,
              "grouped mapped FP32 output must be bitwise dense GEMM");
      for (int col = 0; col < n; ++col) {
        const uint16_t want = dgpp::float_to_bf16_bits(out.p[row * os + col]);
        require(bf.p[row * os + col] == want ||
                    (std::isnan(dgpp::bf16_bits_to_float(want)) &&
                     std::isnan(dgpp::bf16_bits_to_float(bf.p[row * os + col]))),
                "grouped BF16 epilogue");
      }
      for (int col = n; col < os; ++col)
        require(out.p[row * os + col] == -12345.f && bf.p[row * os + col] == 0x1234,
                "grouped output padding must remain untouched");
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
}

// The Mixed346 form (2026-10-09): a grouped launch over views of every
// width (3 / 4 / 6-bit converted triples) and an existing int4 g64 triple,
// the rows' int8 codes and fp32 scales through the row map, ragged tiles,
// both output dtypes, both wide forms (bitwise), the tile list (bitwise),
// NaN propagation — against the exact oracle (integer group dots scaled in
// double) and, within fp32 reassociation, the Mixed346 GEMV core.
DGPP_TEST(packq_gemm_mixed346_grouped_matches_oracle) {
  constexpr int n = 70, os = 77, tokens = 137, ns = 8, k = 512, groups = k / 128;
  struct M346Matrix {
    int bits, sf;
    Buffer<uint32_t> packed;
    Buffer<uint16_t> scales;
    M346Matrix(int bits_, int sf_, uint64_t seed)
        : bits(bits_), sf(sf_), packed(static_cast<size_t>(n) * k * bits_ / 32),
          scales(static_cast<size_t>(n) * k / dgpp::packed_scale_group(sf_)) {
      scale_gemm_test::Rng rng(seed);
      for (size_t i = 0; i < static_cast<size_t>(n) * k * bits / 32; ++i) packed.p[i] = static_cast<uint32_t>(rng.next());
      const double base = sf == dgpp::kPackedScaleBf16G128Mixed346 ? (bits == 6 ? 0.012 : 0.003) : 0.05;
      for (size_t i = 0; i < static_cast<size_t>(n) * k / dgpp::packed_scale_group(sf); ++i)
        scales.p[i] = dgpp::float_to_bf16_bits(static_cast<float>(std::exp2(rng.unit() * 3) * base));
    }
    dgpp::GlmPackedMatrix view() const { return {packed.p, scales.p, n, k, bits, sf}; }
    int level(int row, int col) const { return dgpp::packq_level(packed.p, k, bits, row, col, sf); }
    double scale(int row, int col) const {
      const int g = dgpp::packed_scale_group(sf);
      return dgpp::bf16_bits_to_float(scales.p[static_cast<size_t>(row) * (k / g) + col / g]);
    }
  };
  // Expert e's form: 3, 4, 6, existing, 3, 4, 6, existing.
  M346Matrix w3(3, dgpp::kPackedScaleBf16G128Mixed346, 0x346003), w4(4, dgpp::kPackedScaleBf16G128Mixed346, 0x346004),
      w6(6, dgpp::kPackedScaleBf16G128Mixed346, 0x346006), w0(4, dgpp::kPackedScaleBf16G64, 0x346000);
  w3.scales.p[9 * groups + 1] = 0x7fc0;  // a NaN scale: poisons column 9 of the 3-bit experts' rows
  const M346Matrix* forms[4] = {&w3, &w4, &w6, &w0};
  Buffer<dgpp::MoeExpertView> views(ns * 3);
  Buffer<dgpp::MoeSegment> segs(ns);
  const int counts[ns] = {1, 16, 17, 31, 32, 33, 65, 130};
  int total = 0;
  for (int e = 0; e < ns; ++e) {
    segs.p[e] = {total, counts[e], e};
    total += counts[e];
    for (int which = 0; which < 3; ++which) views.p[e * 3 + which] = dgpp::MoeExpertView::of(forms[e % 4]->view());
  }
  Buffer<int32_t> rows(total);
  for (int r = 0; r < total; ++r) rows.p[r] = (r * 19) % tokens;
  // The rows: bf16 (the existing experts) and the int8 codes + scales (the
  // converted ones), independent random data.
  Buffer<uint16_t> a(static_cast<size_t>(tokens) * k);
  fill(a.p, static_cast<size_t>(tokens) * k, 617);
  Buffer<int8_t> aq(static_cast<size_t>(tokens) * k);
  Buffer<float> as(static_cast<size_t>(tokens) * groups);
  {
    scale_gemm_test::Rng rng(0xA8);
    for (size_t i = 0; i < static_cast<size_t>(tokens) * k; ++i) aq.p[i] = static_cast<int8_t>(static_cast<int>(rng.next() % 256) - 128);
    for (size_t i = 0; i < static_cast<size_t>(tokens) * groups; ++i) as.p[i] = static_cast<float>(std::exp2(rng.unit() * 2) * 0.01);
  }
  Buffer<float> out(static_cast<size_t>(total) * os), out4(static_cast<size_t>(total) * os);
  Buffer<uint16_t> bf(static_cast<size_t>(total) * os);
  std::fill(out.p, out.p + static_cast<size_t>(total) * os, -12345.f);
  std::fill(out4.p, out4.p + static_cast<size_t>(total) * os, -12345.f);
  std::fill(bf.p, bf.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const int sf3 = dgpp::kPackedScaleBf16G128Mixed346;
  dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, 130, views.p, 1, out.p, os, n, k, 0, stream, rows.p, sf3,
                                         -1, nullptr, nullptr, 0, aq.p, k, as.p, groups);
  dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, 130, views.p, 1, out4.p, os, n, k, 0, stream, rows.p, sf3,
                                         4, nullptr, nullptr, 0, aq.p, k, as.p, groups);
  dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, 130, views.p, 1, bf.p, os, n, k, 0, stream, rows.p, sf3,
                                          -1, nullptr, nullptr, 0, nullptr, -1, aq.p, k, as.p, groups);
  // The tile list: bitwise the segment-major launch.
  const int tile_cap = dgpp::moe_tile_list_capacity(ns, total, dgpp::kPackqGemmWideRows);
  Buffer<dgpp::MoeTile> tiles(tile_cap);
  Buffer<int32_t> tile_count(1);
  Buffer<float> listed(static_cast<size_t>(total) * os);
  std::fill(listed.p, listed.p + static_cast<size_t>(total) * os, -12345.f);
  dgpp::launch_moe_tile_list(segs.p, ns, dgpp::kPackqGemmWideRows, tiles.p, tile_count.p, stream);
  dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, /*max_rows=*/1, views.p, 1, listed.p, os, n, k, 0, stream,
                                         rows.p, sf3, -1, tiles.p, tile_count.p, tile_cap, aq.p, k, as.p, groups);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  require(std::memcmp(out4.p, out.p, static_cast<size_t>(total) * os * 4) == 0, "mixed346: the four-warp form bitwise the eight-warp one");
  require(std::memcmp(listed.p, out.p, static_cast<size_t>(total) * os * 4) == 0, "mixed346: the listed launch bitwise the segment-major one");
  // The oracle: per gathered row, the exact group dots (converted) or the
  // bf16 x (code x scale) chain (existing), in double.
  double error2 = 0, norm2 = 0, max_abs = 0, max_error = 0;
  int nan_cols = 0;
  for (int e = 0; e < ns; ++e) {
    const M346Matrix& w = *forms[e % 4];
    const bool conv = w.sf == sf3;
    for (int j = 0; j < counts[e]; ++j) {
      const int r = segs.p[e].row0 + j, t = rows.p[r];
      for (int col = 0; col < n; ++col) {
        double expected = 0;
        if (conv) {
          for (int g = 0; g < groups; ++g) {
            long dot = 0;
            for (int q = 0; q < 128; ++q)
              dot += static_cast<long>(w.level(col, g * 128 + q)) * static_cast<long>(aq.p[static_cast<size_t>(t) * k + g * 128 + q]);
            expected += static_cast<double>(dot) * w.scale(col, g * 128) * as.p[static_cast<size_t>(t) * groups + g];
          }
        } else {
          for (int kk = 0; kk < k; ++kk)
            expected += dgpp::bf16_bits_to_float(a.p[static_cast<size_t>(t) * k + kk]) * w.level(col, kk) * w.scale(col, kk);
        }
        const float actual = out.p[static_cast<size_t>(r) * os + col];
        const uint16_t actual_bf = bf.p[static_cast<size_t>(r) * os + col];
        if (std::isnan(expected)) {
          require(std::isnan(actual) && (actual_bf & 0x7FFF) > 0x7F80, "mixed346: a NaN scale poisons exactly its column");
          ++nan_cols;
          continue;
        }
        if (!std::isfinite(actual)) {
          static int reported = 0;
          if (reported++ < 12)
            std::printf("  non-finite: expert %d (form %s) row %d (token %d) col %d: got %g expected %g\n", e,
                        conv ? (w.bits == 3 ? "3-bit" : w.bits == 4 ? "4-bit" : "6-bit") : "existing", j, t, col,
                        static_cast<double>(actual), expected);
        }
        require(std::isfinite(actual), "mixed346: finite output");
        require(dgpp::float_to_bf16_bits(actual) == actual_bf, "mixed346: bf16(out_f32) == out_bf16");
        const double err = std::abs(actual - expected);
        error2 += err * err;
        norm2 += expected * expected;
        max_abs = std::max(max_abs, std::abs(expected));
        max_error = std::max(max_error, err);
      }
    }
  }
  const double relative = std::sqrt(error2 / std::max(norm2, 1e-30));
  std::printf("mixed346 grouped M%d N%d K%d: fp32 l2_rel %.3g max_error/max_abs %.3g, NaN columns %d\n", total, n, k, relative,
              max_error / std::max(max_abs, 1e-30), nan_cols);
  require(relative < 5e-6 && max_error < std::max(1e-8, max_abs * 4e-5), "mixed346 tile kernel differs from the exact oracle");
  require(nan_cols == counts[0] + counts[4], "mixed346: the NaN column on every row of the 3-bit experts");
  // The GEMV core on one converted expert's rows: the same group dots, fp32
  // reassociation apart.
  {
    const int e = 1, m = counts[e];
    Buffer<int8_t> cq(static_cast<size_t>(m) * k);
    Buffer<float> cs(static_cast<size_t>(m) * groups);
    for (int j = 0; j < m; ++j) {
      const int t = rows.p[segs.p[e].row0 + j];
      std::memcpy(cq.p + static_cast<size_t>(j) * k, aq.p + static_cast<size_t>(t) * k, k);
      std::memcpy(cs.p + static_cast<size_t>(j) * groups, as.p + static_cast<size_t>(t) * groups, groups * 4);
    }
    Buffer<float> gemv(static_cast<size_t>(m) * n);
    dgpp::launch_packq_a8_gemv_f32(cq.p, k, cs.p, groups, w4.view(), gemv.p, m, n, k, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    double e2 = 0, n2 = 0;
    for (int j = 0; j < m; ++j)
      for (int col = 0; col < n; ++col) {
        const double g = gemv.p[static_cast<size_t>(j) * n + col], t = out.p[static_cast<size_t>(segs.p[e].row0 + j) * os + col];
        e2 += (g - t) * (g - t);
        n2 += t * t;
      }
    const double rel = std::sqrt(e2 / std::max(n2, 1e-30));
    std::printf("mixed346 tile vs GEMV core (4-bit expert): l2_rel %.3g\n", rel);
    require(rel < 5e-6, "mixed346: the tile kernel and the GEMV core agree within fp32 reassociation");
  }
  // The contract: the form is grouped-only, with codes.
  {
    bool refused = false;
    try {
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, 130, views.p, 1, out.p, os, n, k, 0, stream, rows.p, sf3);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "mixed346: a launch without the codes is refused");
    refused = false;
    try {
      Buffer<float> d(static_cast<size_t>(4) * n);
      dgpp::launch_packq_gemm_f32(a.p, k, w4.view(), d.p, 4, n, k, stream);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "mixed346: the dense launcher refuses the form");
  }
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  std::printf("[ OK ] packq gemm mixed346 grouped: oracle, forms, tile list, NaN, GEMV agreement, contract\n");
}

DGPP_TEST(packq_gemm_rejects_invalid_geometry) {
  Matrix w(4, 16, 64, 919);
  Buffer<uint16_t> a(64), out(16);
  auto rejects = [&](dgpp::GlmPackedMatrix matrix, int stride, int n, int k) {
    try {
      dgpp::launch_packq_gemm_bf16(a.p, stride, matrix, out.p, 1, n, k, nullptr);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  auto matrix = w.view();
  require(rejects(matrix, 63, 16, 64), "short activation stride");
  require(rejects(matrix, 64, 17, 64), "N exceeds matrix rows");
  require(rejects(matrix, 64, 16, 32), "K differs from matrix columns");
  matrix.bits = 3;
  require(rejects(matrix, 64, 16, 64), "invalid code width");
  matrix = w.view();
  matrix.packed += 1;
  require(rejects(matrix, 64, 16, 64), "unaligned packed row");
  matrix = w.view();
  matrix.scale_fmt = 1;
  require(rejects(matrix, 64, 16, 64), "g128 format with K=64 (not a multiple of 128)");
  matrix.scale_fmt = 2;
  require(rejects(matrix, 64, 16, 64), "unknown scale format");
  Matrix g(4, 16, 128, 921, 1);
  Buffer<uint16_t> a128(128);
  bool ok = true;
  try {
    dgpp::launch_packq_gemm_bf16(a128.p, 128, g.view(), out.p, 1, 16, 128, nullptr);
  } catch (const std::invalid_argument&) {
    ok = false;
  }
  require(ok, "g128 format with K=128 accepted");
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
