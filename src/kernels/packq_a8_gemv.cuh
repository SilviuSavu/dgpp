#pragma once
// The Mixed346 GEMV core (2026-10-09, HawkBearPig/GLM-5.3-Mixed346-GPTQ-
// H32-A8-g128, its FORMAT.md): 3-, 4- or 6-bit codebook weights
// (kPackedScaleBf16G128Mixed346, models/quant_matrix.hpp — a dense bit
// stream per row, one bf16 scale per 128 codes) against INT8 activation
// codes (one fp32 scale per 128 values of a row, the quantizer in
// kernels/hadamard32.cu), at m = 1..4 activation rows.
//
// SHAPE: the packed core's (packq_gemv.cuh) with a PIECE of the row's
// stream as the lane unit (Fmt below: 16 bytes at 4 bits, 48 bytes — a
// group or half a group — at 3 and 6 bits, the 48-byte widths loaded
// cooperatively and swapped through shared memory); a 128-code group
// carries exactly one weight scale and, per activation row, one activation
// scale. The row geometry is a compile-time function of (Bits, K):
// row_pieces = K / codes_per_piece, lanes_per_row the largest power of two
// <= 32 dividing it (a multiple of the lanes a group spans), each lane's
// pieces consumed in passes of at most max_pieces_per_pass. Which lane
// owns which group is a function of K alone, and every row's chain is the
// same sequence of operations whatever kRows, the pass depth or which
// launcher issued it — the property the slot path and the grouped path
// are pinned on, bitwise.
//
// NUMERICS (the checkpoint's arithmetic, FORMAT.md): a group's dot is the
// exact INT32 sum of 128 products level x code (dp4a, four a step), then
// one fp32 multiply by the scale product and one fp32 fma into the row's
// accumulator:
//   acc = fma(float(sum_int32), float(s_w) * s_a, acc)
// in each lane's group order, then the fixed xor tree across the row's
// lanes. The scale product is one fp32 rounding; the tensor-core prefill
// kernel (packq_gemm.cu) computes the same group dots exactly (integer
// operands below 2^24 in bf16 MMAs) and differs by the accumulation order
// only, the packed paths' stance.
//
// DECODE: 4-bit indices to levels by the NF4I8 byte permutes
// (expand_nf4_int8, packq_gemv.cuh) — the four bytes are the dp4a operand
// directly. 3-bit indices: four consecutive codes (12 bits, a funnel shift
// across a word boundary) spread to nibble positions select the eight
// Gaussian levels through one PRMT. 6-bit codes: four codes (24 bits) to
// four unsigned bytes, multiplied as u8 x s8 (the dp4a mixed form) with
// the codebook's offset (code - 32) folded out once per group as 32 x
// (the group's activation code sum), computed at staging. No
// integer-to-float conversion inside the chain.
//
// CONTRACT: K a positive multiple of 128 with at most 8 groups per lane
// (K <= 32768), in the compiled set (dispatch_k); packed rows 16-byte
// aligned; activation code rows 16-byte aligned. A NaN scale propagates
// as NaN.
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "common/dtypes.hpp"
#include "kernels/gemv_common.cuh"
#include "kernels/packq_gemv.cuh"
#include "models/quant_matrix.hpp"

namespace dgpp {
namespace packq_a8 {

using gemv::kMaxRows;
using gemv::kThreads;
using gemv::kWarps;

constexpr int kGroup = 128;             // codes per scale (weights and activations)
constexpr int kMaxGroupsPerLane = 8;    // K <= 32768

__host__ __device__ constexpr bool bits_known(int bits) { return bits == 3 || bits == 4 || bits == 6; }

template <int Bits>
struct Fmt {
  static_assert(bits_known(Bits), "packq_a8: 3-, 4- or 6-bit codes");
  // The lane unit (2026-10-10): a PIECE of a row's stream. At 4 bits a
  // 16-byte piece (32 codes, four lanes a group) — consecutive lanes read
  // consecutive pieces, so a warp's load is one contiguous span (the
  // packed core's pattern; a group per lane put sixteen cache lines behind
  // every load instruction and stalled the warps 83 % of their cycles on
  // the L1TEX scoreboard — ncu). At 3 and 6 bits a 48-byte piece — a whole
  // group (128 codes, no cross-lane reduce, the four-code extraction
  // positions compile-time) or half a group (64 codes, two lanes) — loaded
  // COOPERATIVELY: the row's lanes read the pass's span as contiguous
  // 16-byte vectors (lane `lig` vectors lig, lig + L, lig + 2L) and swap
  // them through a per-warp shared-memory tile into each lane's own piece
  // (three 16-byte stores, a warp sync, three loads). The earlier layouts
  // — three 16-byte pieces of a 3-bit group on three lanes of a quad (one
  // idle, per-lane window shifts and keep masks: 2.3 instructions a code,
  // instruction-bound at 84 % of the ceiling) and 48-byte lanes at a
  // 48-byte stride at 6 bits (two L1 wavefronts a load, 70 %) — are
  // replaced by this one (0.7 instructions a code at 3 bits).
  static constexpr bool transposed = Bits != 4;
  static constexpr int piece_bytes = Bits == 4 ? 16 : 48;
  static constexpr int piece_vecs = piece_bytes / 16;                 // 3 / 1 / 3 uint4
  static constexpr int piece_words = piece_bytes / 4;                 // 12 / 4 / 12
  static constexpr int codes_per_piece = piece_bytes * 8 / Bits;      // 128 / 32 / 64
  static constexpr int pieces_per_group = kGroup / codes_per_piece;   // 1 / 4 / 2 lanes a group
  static constexpr int group_bytes = 16 * Bits;
  // Pieces a lane holds in flight per pass: 64 bytes at 4 bits (the packed
  // core's depth); 96 bytes (two pieces) at 3 and 6 bits when one
  // activation row is staged (the slot kernels: 48 registers of loads for
  // the gate / up pair), one piece when several rows are (the grouped
  // kernels' 48 KB budget holds the rows and a one-piece swap tile).
  __host__ __device__ static constexpr int pass_depth(int rows) { return Bits == 4 ? 4 : (rows == 1 ? 2 : 1); }
};

__host__ __device__ constexpr int groups_of(int k) { return k / kGroup; }
template <int Bits>
__host__ __device__ constexpr int row_pieces_of(int k) { return groups_of(k) * Fmt<Bits>::pieces_per_group; }
// The lanes a row spans: at 4 bits as few as keep a pass's four 16-byte
// pieces in flight on every lane (the packed core's rule: at K = 512 four
// lanes a row and 64 bytes a lane instead of sixteen lanes with one piece
// each, whose four times the blocks and latency-bound passes ran the down
// GEMV at twice the int4 core's time, 2026-10-10); at 3 and 6 bits as many
// as the row's 48-byte pieces allow (a piece a lane: two a lane at K = 512
// cost the 3-bit slot path 10 %). Never fewer than a group's lanes, a
// power of two up to 32 that divides the row's pieces. A function of
// (Bits, K) alone: every path's chain is the same sequence whatever the
// row count.
template <int Bits>
__host__ __device__ constexpr int lanes_per_row_of(int k) {
  const int p = row_pieces_of<Bits>(k);
  const int depth = Bits == 4 ? 4 : 1;
  const int want = p / (p < depth ? p : depth);
  int l = 1;
  while (l < 32 && p % (l * 2) == 0 && l * 2 <= want) l *= 2;
  while (l < Fmt<Bits>::pieces_per_group && p % (l * 2) == 0) l *= 2;
  return l;
}
template <int Bits>
__host__ __device__ constexpr int pieces_per_lane_of(int k) { return row_pieces_of<Bits>(k) / lanes_per_row_of<Bits>(k); }
template <int Bits>
__host__ __device__ constexpr bool k_supported_bits(int k) {
  return k >= kGroup && k % kGroup == 0 && lanes_per_row_of<Bits>(k) >= Fmt<Bits>::pieces_per_group &&
         pieces_per_lane_of<Bits>(k) <= kMaxGroupsPerLane * Fmt<Bits>::pieces_per_group;
}
__host__ __device__ constexpr bool k_supported(int k) {
  return k_supported_bits<3>(k) && k_supported_bits<4>(k) && k_supported_bits<6>(k);
}
template <int Bits>
__host__ __device__ constexpr int rows_per_block_of(int k) { return kWarps * (32 / lanes_per_row_of<Bits>(k)); }

// The compile-time row geometry. kRows (the staged activation rows) sets
// the pass depth of the transposed widths only; the lane / row mapping and
// every row's chain are functions of (Bits, K) alone.
template <int Bits, int K, int kRows = 1>
struct Geom {
  static_assert(k_supported_bits<Bits>(K), "packq_a8: K must be a multiple of 128 with at most 8 groups per lane");
  using F = Fmt<Bits>;
  static constexpr int groups = groups_of(K);
  static constexpr int row_pieces = row_pieces_of<Bits>(K);
  static constexpr int lanes_per_row = lanes_per_row_of<Bits>(K);     // a multiple of pieces_per_group
  static constexpr int pieces_per_lane = row_pieces / lanes_per_row;
  static constexpr int max_pieces_per_pass = F::pass_depth(kRows);
  static constexpr int passes = (pieces_per_lane + max_pieces_per_pass - 1) / max_pieces_per_pass;
  // Pass slots of the warp's swap tile (the transposed widths): the pass
  // depth, or the lane's whole piece count when that is smaller.
  static constexpr int swap_slots = F::transposed ? (pieces_per_lane < max_pieces_per_pass ? pieces_per_lane : max_pieces_per_pass) : 0;
  static constexpr int rows_per_step = 32 / lanes_per_row;
  static constexpr int rows_per_warp = rows_per_step;
  static constexpr int rows_per_block = kWarps * rows_per_warp;
  static constexpr int row_bytes = K * Bits / 8;
  static constexpr int pass_pieces(int p) {
    const int left = pieces_per_lane - p * max_pieces_per_pass;
    return left < 1 ? 1 : (left < max_pieces_per_pass ? left : max_pieces_per_pass);
  }
};

// dp4a through PTX: the signed x signed and unsigned x signed forms (the
// intrinsic's overloads are ambiguous for uint32_t operands).
__device__ __forceinline__ int32_t dp4a_s8s8(uint32_t a, uint32_t b, int32_t c) {
  int32_t d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ int32_t dp4a_u8s8(uint32_t a, uint32_t b, int32_t c) {
  int32_t d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// The staged activation rows in shared memory: [kRows][K/128] groups of 128
// int8 codes at a 132-byte stride, then [kRows][K/128] fp32 scales, then
// [kRows][K/128] int32 code sums (the 6-bit offset fold), then the warps'
// swap tiles (the transposed widths' cooperative loads: kWarps x
// pass_depth x 1536 bytes — one pass slot holds a step's 32 lanes x 48
// bytes). The 132-byte group stride (33 words) makes every word read of
// the chain conflict-free: a warp's lanes read word q of their pieces at
// 33 g + 32 s / piece + q, and {g, s} over a warp's groups and sub-pieces
// cover all 32 banks at each width (at the natural 128 the eight quads of
// a 4-bit warp landed on eight banks — a four-way conflict on every dp4a
// operand; 2026-10-10). The caller provides smem_bytes(kRows, K) and syncs.
constexpr int kGroupStride = kGroup + 4;
constexpr int kSwapSlotBytes = 32 * 48;  // one pass slot of a warp's swap tile
__host__ __device__ constexpr size_t staged_bytes(int rows, int k) {
  const size_t raw = static_cast<size_t>(rows) * static_cast<size_t>(k / kGroup) * (kGroupStride + 8);
  return (raw + 15) / 16 * 16;
}
template <int Bits>
__host__ __device__ constexpr int swap_slots_of(int rows, int k) {
  const int depth = Fmt<Bits>::pass_depth(rows), per_lane = pieces_per_lane_of<Bits>(k);
  return Fmt<Bits>::transposed ? (per_lane < depth ? per_lane : depth) : 0;
}
__host__ __device__ constexpr size_t swap_bytes(int rows, int k) {
  const int a = swap_slots_of<3>(rows, k), b = swap_slots_of<6>(rows, k);
  return static_cast<size_t>(kWarps) * static_cast<size_t>(a > b ? a : b) * kSwapSlotBytes;
}
__host__ __device__ constexpr size_t smem_bytes(int rows, int k) { return staged_bytes(rows, k) + swap_bytes(rows, k); }
__host__ inline bool smem_fits(int rows, int k) { return smem_bytes(rows, k) <= gemv::kMaxSmemBytes; }

struct Staged {
  const uint8_t* codes;   // [kRows][K/128][kGroupStride]
  const float* scales;    // [kRows][K/128]
  const int32_t* sums;    // [kRows][K/128]
  uint8_t* swap;          // [kWarps][swap_slots][kSwapSlotBytes]
};
__device__ __forceinline__ Staged staged_of(uint8_t* smem, int rows, int k) {
  Staged s;
  const int groups = k / kGroup;
  s.codes = smem;
  s.scales = reinterpret_cast<const float*>(smem + static_cast<size_t>(rows) * groups * kGroupStride);
  s.sums = reinterpret_cast<const int32_t*>(s.scales + static_cast<size_t>(rows) * groups);
  s.swap = smem + staged_bytes(rows, k);
  return s;
}

// Stage kRows activation rows (codes at row stride `code_stride` bytes,
// scales at `scale_stride` floats) into shared memory and compute each
// group's code sum. The whole block participates; the caller syncs.
// `kSums`: compute the groups' code sums (the 6-bit offset fold) — a second
// barrier and 32 dp4a a group; the 3- and 4-bit widths skip it.
template <int kRows, bool kSums = true>
__device__ __forceinline__ void stage_activations(const uint8_t* __restrict__ codes, size_t code_stride,
                                                  const float* __restrict__ scales, size_t scale_stride,
                                                  int k, uint8_t* __restrict__ smem) {
  const int groups = k / kGroup;
  uint8_t* sc = smem;
  float* ss = reinterpret_cast<float*>(smem + static_cast<size_t>(kRows) * groups * kGroupStride);
  int32_t* sum = reinterpret_cast<int32_t*>(ss + static_cast<size_t>(kRows) * groups);
  // 16-byte global loads (eight a group), four-word stores at the 132-byte
  // group stride (a group's base is 4-byte aligned only); no division by
  // the runtime group count anywhere (2026-10-10: two per word cost the
  // four-row grouped kernel half its time).
  constexpr int kVecsPerGroup = kGroup / 16;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint8_t* src = codes + static_cast<size_t>(r) * code_stride;
    uint8_t* dst = sc + static_cast<size_t>(r) * groups * kGroupStride;
    for (int v = threadIdx.x; v < groups * kVecsPerGroup; v += blockDim.x) {
      const int g = v >> 3, w = v & 7;
      const uint4 x = *reinterpret_cast<const uint4*>(src + g * kGroup + w * 16);
      uint32_t* d = reinterpret_cast<uint32_t*>(dst + g * kGroupStride + w * 16);
      d[0] = x.x;
      d[1] = x.y;
      d[2] = x.z;
      d[3] = x.w;
    }
    for (int g = threadIdx.x; g < groups; g += blockDim.x)
      ss[r * groups + g] = scales[static_cast<size_t>(r) * scale_stride + g];
  }
  if constexpr (kSums) {
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(sc + (static_cast<size_t>(r) * groups + g) * kGroupStride);
        int32_t s = 0;
#pragma unroll
        for (int q = 0; q < 32; ++q) s = dp4a_s8s8(a[q], 0x01010101u, s);
        sum[r * groups + g] = s;
      }
    }
  }
}

// ---- the level decoders -----------------------------------------------------

// Four 3-bit Gaussian indices (12 bits at the low end of `x`) to four
// signed int8 levels: spread to nibble positions, then one PRMT into the
// eight-byte level table {-127,-79,-45,-14 | 14,45,79,127}.
__device__ __forceinline__ uint32_t gauss3_levels(uint32_t x) {
  uint32_t t = (x & 0x3Fu) | ((x & 0xFC0u) << 2);          // a | b<<3 | c<<8 | d<<11
  t = (t & 0x0707u) | ((t & 0x3838u) << 1);                // a | b<<4 | c<<8 | d<<12
  return __byte_perm(0xF2D3B181u, 0x7F4F2D0Eu, t);
}
// Four 6-bit codes (24 bits at the low end of `x`) to four unsigned bytes.
__device__ __forceinline__ uint32_t int6_bytes(uint32_t x) {
  return (x & 0x3Fu) | ((x & 0xFC0u) << 2) | ((x & 0x3F000u) << 4) | ((x & 0xFC0000u) << 6);
}

// One piece of a row (its bytes in `w`, the group's scale s_w) into kRows
// accumulators from the staged rows: the lane's exact int32 partial over
// its codes (dp4a over the matching activation words), the group's dot
// summed across the group's lanes (xor shuffles, exact), then — on the
// group's first lane — acc[r] = fma(float(dot), s_w * s_a[r], acc[r]). Every
// lane of the warp runs the shuffles (no early exit: a lane past the
// matrix's rows holds zero weights and a zero scale).
template <int Bits, int kRows>
__device__ __forceinline__ void consume_piece(const uint4 (&w)[Fmt<Bits>::piece_vecs], float s_w,
                                              const Staged& st, int k, int g, int sub, float (&acc)[kRows]) {
  using F = Fmt<Bits>;
  const int groups = k / kGroup;
  uint32_t words[F::piece_words + 1];
#pragma unroll
  for (int v = 0; v < F::piece_vecs; ++v) {
    words[4 * v] = w[v].x;
    words[4 * v + 1] = w[v].y;
    words[4 * v + 2] = w[v].z;
    words[4 * v + 3] = w[v].w;
  }
  words[F::piece_words] = 0u;
  // The dp4a operands of the piece: levels of codes 4j .. 4j + 3, each
  // four-code field at the compile-time bit 4 j Bits of the piece — decoded
  // one quad at a time and consumed into every staged row's dot before the
  // next (one live operand word, not a piece's 32: the grouped kernel held
  // 128 registers and two blocks an SM at the 512 width, 2026-10-10).
  constexpr int kQuads = F::codes_per_piece / 4;  // 32 / 8 / 16
  const uint32_t* a[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r)
    a[r] = reinterpret_cast<const uint32_t*>(st.codes + (static_cast<size_t>(r) * groups + g) * kGroupStride +
                                             sub * F::codes_per_piece);
  int32_t dot[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) dot[r] = 0;
#pragma unroll
  for (int j = 0; j < kQuads; ++j) {
    uint32_t lv;
    if constexpr (Bits == 4) {
      lv = (j & 1) ? packq_gemv::expand_nf4_int8(words[j / 2] >> 16) : packq_gemv::expand_nf4_int8(words[j / 2] & 0xFFFFu);
    } else {
      const int bit = 4 * Bits * j;
      const uint32_t x = __funnelshift_r(words[bit / 32], words[bit / 32 + 1], bit % 32);
      lv = Bits == 3 ? gauss3_levels(x) : int6_bytes(x);
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      if constexpr (Bits == 6) dot[r] = dp4a_u8s8(lv, a[r][j], dot[r]);
      else dot[r] = dp4a_s8s8(lv, a[r][j], dot[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    int32_t d = dot[r];
    if constexpr (F::pieces_per_group >= 2) d += __shfl_xor_sync(0xFFFFFFFFu, d, 1);
    if constexpr (F::pieces_per_group >= 4) d += __shfl_xor_sync(0xFFFFFFFFu, d, 2);
    if constexpr (Bits == 6) d -= 32 * st.sums[r * groups + g];
    if (sub == 0) {
      const float s = s_w * st.scales[r * groups + g];
      acc[r] = __fmaf_rn(static_cast<float>(d), s, acc[r]);
    }
  }
}

// The loads of one pass of a warp's rows (pieces [P0, P0 + NP) of each
// lane's row, rows [n0 + warp * rows_per_warp, +rows_per_warp) of an [n,
// K] matrix): every load issued, none consumed. At 4 bits a lane reads its
// own 16-byte pieces (consecutive lanes, consecutive pieces). At 3 and 6
// bits the row's L lanes read the pass slot's L x 48-byte span as
// contiguous 16-byte vectors (lane lig: vectors lig + L v), to be swapped
// into pieces by swap_pieces. A row past n loads zeros (and a zero scale).
template <int Bits, int K, int kRows, int P0, int NP>
__device__ __forceinline__ void load_pass(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                         int row, bool ok, int lig, uint4 (&wv)[NP][Fmt<Bits>::piece_vecs],
                                         uint16_t (&sv)[NP]) {
  using G = Geom<Bits, K, kRows>;
  using F = Fmt<Bits>;
  const uint8_t* rb = w + static_cast<size_t>(row) * G::row_bytes;
#pragma unroll
  for (int c = 0; c < NP; ++c) {
    const int pi = (P0 + c) * G::lanes_per_row + lig;  // the lane's piece in the row
    if constexpr (F::transposed) {
      const uint8_t* span = rb + static_cast<size_t>(P0 + c) * G::lanes_per_row * F::piece_bytes;
#pragma unroll
      for (int v = 0; v < F::piece_vecs; ++v)
        wv[c][v] = ok ? reinterpret_cast<const uint4*>(span)[lig + v * G::lanes_per_row] : make_uint4(0u, 0u, 0u, 0u);
    } else {
      const uint8_t* src = rb + static_cast<size_t>(pi) * F::piece_bytes;
#pragma unroll
      for (int v = 0; v < F::piece_vecs; ++v) wv[c][v] = ok ? reinterpret_cast<const uint4*>(src)[v] : make_uint4(0u, 0u, 0u, 0u);
    }
    sv[c] = ok ? scales[static_cast<size_t>(row) * G::groups + pi / F::pieces_per_group] : static_cast<uint16_t>(0);
  }
}

// The transposed widths' swap: the vectors a lane loaded (lig + L v of its
// row's pass slot) through the warp's shared tile into the lane's own
// piece (vectors 3 lig + v). Each pass slot c has its own 1536-byte region
// (32 lanes x 48 bytes: rows_per_step rows of L x 48); the stores and the
// 48-byte-stride loads are both conflict-free. The whole warp takes part
// (a row past n swaps zeros).
template <int Bits, int K, int kRows, int NP>
__device__ __forceinline__ void swap_pieces(const Staged& st, int warp, int step_row, int lig,
                                            uint4 (&wv)[NP][Fmt<Bits>::piece_vecs]) {
  using G = Geom<Bits, K, kRows>;
  using F = Fmt<Bits>;
  static_assert(F::transposed && F::piece_bytes == 48, "swap_pieces: the 48-byte widths");
  static_assert(NP <= G::swap_slots, "swap_pieces: the swap tile holds swap_slots pass slots");
  uint8_t* tile = st.swap + static_cast<size_t>(warp) * G::swap_slots * kSwapSlotBytes +
                  static_cast<size_t>(step_row) * G::lanes_per_row * F::piece_bytes;
#pragma unroll
  for (int c = 0; c < NP; ++c) {
    uint4* slot = reinterpret_cast<uint4*>(tile + c * kSwapSlotBytes);
#pragma unroll
    for (int v = 0; v < F::piece_vecs; ++v) slot[lig + v * G::lanes_per_row] = wv[c][v];
  }
  __syncwarp();
#pragma unroll
  for (int c = 0; c < NP; ++c) {
    const uint4* slot = reinterpret_cast<const uint4*>(tile + c * kSwapSlotBytes);
#pragma unroll
    for (int v = 0; v < F::piece_vecs; ++v) wv[c][v] = slot[F::piece_vecs * lig + v];
  }
  __syncwarp();  // the tile is free for the next swap
}

// One pass of a warp's row dots: pieces [P0, P0 + NP) of each lane's row
// against the staged rows. Every load of the pass is issued before any is
// consumed; the pieces are then consumed in order into the row's
// accumulator. No early return: the group shuffles are warp-uniform.
template <int Bits, int K, int kRows, int P0, int NP>
__device__ __forceinline__ void pass_row_dots(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                              const Staged& st, int n0, int n, float (&acc)[kRows]) {
  using G = Geom<Bits, K, kRows>;
  using F = Fmt<Bits>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row = n0 + warp * G::rows_per_warp + group;
  const bool ok = row < n;
  uint4 wv[NP][F::piece_vecs];
  uint16_t sv[NP];
  load_pass<Bits, K, kRows, P0, NP>(w, scales, row, ok, lig, wv, sv);
  if constexpr (F::transposed) swap_pieces<Bits, K, kRows, NP>(st, warp, group, lig, wv);
#pragma unroll
  for (int c = 0; c < NP; ++c) {
    const int pi = (P0 + c) * G::lanes_per_row + lig;
    consume_piece<Bits, kRows>(wv[c], bf16_bits_to_float(sv[c]), st, K, pi / F::pieces_per_group,
                               pi % F::pieces_per_group, acc);
  }
}

template <int Bits, int K, int kRows, int P>
__device__ __forceinline__ void pass_chain(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                           const Staged& st, int n0, int n, float (&acc)[kRows]) {
  using G = Geom<Bits, K, kRows>;
  if constexpr (P < G::passes) {
    pass_row_dots<Bits, K, kRows, P * G::max_pieces_per_pass, G::pass_pieces(P)>(w, scales, st, n0, n, acc);
    pass_chain<Bits, K, kRows, P + 1>(w, scales, st, n0, n, acc);
  }
}

// The dots of a warp's rows_per_warp rows against the kRows staged rows,
// reduced across each row's lane group; lane `lig == 0` holds the value.
template <int Bits, int K, int kRows>
__device__ __forceinline__ void warp_row_dots(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                              const Staged& st, int n0, int n, float (&acc)[kRows]) {
  using G = Geom<Bits, K, kRows>;
#pragma unroll
  for (int a = 0; a < kRows; ++a) acc[a] = 0.f;
  pass_chain<Bits, K, kRows, 0>(w, scales, st, n0, n, acc);
  packq_gemv::group_reduce<kRows, G::lanes_per_row>(acc);
}

// One pass of two matrices over the same rows and staged activations (gate
// and up): every load of both issued before either is consumed; each
// matrix's row chain is pass_row_dots's exactly (bitwise).
template <int Bits, int K, int kRows, int P0, int NP>
__device__ __forceinline__ void pass_row_dots_pair(const uint8_t* __restrict__ w0, const uint16_t* __restrict__ s0,
                                                   const uint8_t* __restrict__ w1, const uint16_t* __restrict__ s1,
                                                   const Staged& st, int n0, int n, float (&acc0)[kRows],
                                                   float (&acc1)[kRows]) {
  using G = Geom<Bits, K, kRows>;
  using F = Fmt<Bits>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row = n0 + warp * G::rows_per_warp + group;
  const bool ok = row < n;
  uint4 wv0[NP][F::piece_vecs], wv1[NP][F::piece_vecs];
  uint16_t sv0[NP], sv1[NP];
  load_pass<Bits, K, kRows, P0, NP>(w0, s0, row, ok, lig, wv0, sv0);
  load_pass<Bits, K, kRows, P0, NP>(w1, s1, row, ok, lig, wv1, sv1);
  if constexpr (F::transposed) {
    swap_pieces<Bits, K, kRows, NP>(st, warp, group, lig, wv0);
    swap_pieces<Bits, K, kRows, NP>(st, warp, group, lig, wv1);
  }
#pragma unroll
  for (int c = 0; c < NP; ++c) {
    const int pi = (P0 + c) * G::lanes_per_row + lig;
    const int g = pi / F::pieces_per_group, sub = pi % F::pieces_per_group;
    consume_piece<Bits, kRows>(wv0[c], bf16_bits_to_float(sv0[c]), st, K, g, sub, acc0);
    consume_piece<Bits, kRows>(wv1[c], bf16_bits_to_float(sv1[c]), st, K, g, sub, acc1);
  }
}
template <int Bits, int K, int kRows, int P>
__device__ __forceinline__ void pass_chain_pair(const uint8_t* __restrict__ w0, const uint16_t* __restrict__ s0,
                                                const uint8_t* __restrict__ w1, const uint16_t* __restrict__ s1,
                                                const Staged& st, int n0, int n, float (&acc0)[kRows],
                                                float (&acc1)[kRows]) {
  using G = Geom<Bits, K, kRows>;
  if constexpr (P < G::passes) {
    pass_row_dots_pair<Bits, K, kRows, P * G::max_pieces_per_pass, G::pass_pieces(P)>(w0, s0, w1, s1, st, n0, n,
                                                                                      acc0, acc1);
    pass_chain_pair<Bits, K, kRows, P + 1>(w0, s0, w1, s1, st, n0, n, acc0, acc1);
  }
}
// Two matrices over the same rows and staged activations (gate and up),
// each row's chain its own single-matrix chain exactly.
template <int Bits, int K, int kRows>
__device__ __forceinline__ void warp_row_dots_pair(const uint8_t* __restrict__ w0, const uint16_t* __restrict__ s0,
                                                   const uint8_t* __restrict__ w1, const uint16_t* __restrict__ s1,
                                                   const Staged& st, int n0, int n, float (&acc0)[kRows],
                                                   float (&acc1)[kRows]) {
  using G = Geom<Bits, K, kRows>;
#pragma unroll
  for (int a = 0; a < kRows; ++a) acc0[a] = acc1[a] = 0.f;
  pass_chain_pair<Bits, K, kRows, 0>(w0, s0, w1, s1, st, n0, n, acc0, acc1);
  packq_gemv::group_reduce<kRows, G::lanes_per_row>(acc0);
  packq_gemv::group_reduce<kRows, G::lanes_per_row>(acc1);
}

// Which lane owns the reduced dot of the warp's row, and that row's index.
template <int Bits, int K>
__device__ __forceinline__ int owned_row(int n0, bool& mine) {
  using G = Geom<Bits, K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  mine = (lane % G::lanes_per_row) == 0;
  return n0 + warp * G::rows_per_warp + lane / G::lanes_per_row;
}

// The block body: kWarps warps x rows_per_warp rows each of an [n, K]
// matrix (payload [n, K*Bits/8] bytes, scales bf16 [n, K/128]):
//   out[a * out_stride + row] = store_dot(dot(row, staged row a))
template <int Bits, int K, int kRows, typename OutT>
__device__ __forceinline__ void block_rows(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                           const Staged& st, int n0, int n, OutT* __restrict__ out,
                                           size_t out_stride) {
  float acc[kRows];
  warp_row_dots<Bits, K, kRows>(w, scales, st, n0, n, acc);
  bool mine = false;
  const int row = owned_row<Bits, K>(n0, mine);
  if (mine && row < n) {
#pragma unroll
    for (int a = 0; a < kRows; ++a) packq_gemv::store_dot(out + static_cast<size_t>(a) * out_stride + row, acc[a]);
  }
}

// The compiled K set: the full GLM-5.3 routed widths — 6144 (gate / up),
// 2048 / 1024 / 512 (the down at worlds 1 / 2 / 4) — and the test
// geometries 128 and 256.
__host__ __device__ constexpr bool k_compiled(int k) {
  return k == 128 || k == 256 || k == 512 || k == 1024 || k == 2048 || k == 6144;
}
template <typename F>
__host__ inline void dispatch_k(int k, F&& f) {
  switch (k) {
    case 128: f(std::integral_constant<int, 128>{}); return;
    case 256: f(std::integral_constant<int, 256>{}); return;
    case 512: f(std::integral_constant<int, 512>{}); return;
    case 1024: f(std::integral_constant<int, 1024>{}); return;
    case 2048: f(std::integral_constant<int, 2048>{}); return;
    case 6144: f(std::integral_constant<int, 6144>{}); return;
    default:
      throw std::invalid_argument("packq_a8: K is not in the compiled set (128, 256, 512, 1024, 2048, 6144)");
  }
}
template <typename F>
__host__ inline void dispatch_bits(int bits, F&& f) {
  if (bits == 3) f(std::integral_constant<int, 3>{});
  else if (bits == 4) f(std::integral_constant<int, 4>{});
  else if (bits == 6) f(std::integral_constant<int, 6>{});
  else throw std::invalid_argument("packq_a8: the code width must be 3, 4 or 6");
}

}  // namespace packq_a8
}  // namespace dgpp
