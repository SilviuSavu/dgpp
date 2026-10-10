// The DFlash2 drafter for GLM-5.3-Flash (the port of models/glm_dsa/
// dflash2_model.cpp — itself the Qwen port — to the Flash family,
// docs/performance_improvement_plan.md §7): block-diffusion drafting over
// the target's tap-layer mHC stream means. The walk (decode.cpp) captures
// each tap layer's boundary streams through the contraction the head uses
// (launch_mhc_final_mean — the reference's hc_contract, mean over the four
// streams; the mid-stack capture hazard the deployment recipe documents),
// accumulates them through the drafter's split fc, and lands each draft
// layer's K/V in the drafter's own window planes (DFlash2Planes).
//
// Deltas against the glm_dsa original, all forced by the target:
//   * the tap feature is the mHC stream mean at the tap boundary, not a
//     plain residual (the capture hook lives in the Flash walk);
//   * the embedding table is full on every rank (the loader never shards
//     it), so the block's [anchor, mask x drafts] gather is one kernel
//     with no boundary fold;
//   * this family caps at kDecodeRows = 8 rows — exactly one 8-row block
//     verify — so the stacked draft batch is a single slot (df_batch_ 1)
//     and the recorded draft exists only in its scalar form.
#include <chrono>
#include <cmath>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/qsa.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mma_gemv.hpp"
#include "models/glm/config.hpp"
#include "models/glm/dflash2.hpp"
#include "models/glm/forward.hpp"

namespace dgpp {

void GlmDiagnosticModel::dflash2_init(const std::string& dflash2_dir) {
  dfcfg_ = DFlash2Config::from_json_file(dflash2_dir + "/config.json");
  dflash2_validate_against(dfcfg_, cfg_);
  if (!dfcfg_.tp_divisible(loader_.world()))
    throw std::invalid_argument(
        "GlmDiagnosticModel: the DFlash2 drafter's heads / kv heads / MLP rows must divide across the world");
  df_tap_.assign(cfg_.num_hidden_layers, -1);
  for (size_t t = 0; t < dfcfg_.target_layer_ids.size(); ++t)
    df_tap_[static_cast<size_t>(dfcfg_.target_layer_ids[t])] = static_cast<int>(t);
  const int H = cfg_.hidden_size;
  const int M = max_tokens_;
  df_nh_ = dfcfg_.local_heads(loader_.world());
  df_kvh_ = dfcfg_.local_kv_heads(loader_.world());
  df_qw_ = dfcfg_.local_q_row(loader_.world());
  df_kvr_ = dfcfg_.local_kv_row(loader_.world());
  df_i_ = dfcfg_.local_intermediate(loader_.world());
  dfw_ = load_dflash2_weights(dfcfg_, dflash2_dir, stream_, loader_.rank(), loader_.world(), dflash2_fp8_);
  // The drafter's bf12 companions (the glm_dsa port's recipe): the five
  // block matrices per layer — under fp8 only the k|v rows keep a bf16 copy
  // (the feature path's GEMM reads them) — the fc taps and the hidden
  // projection, packed beside the model's own companions before
  // finish_companions() reserves (preconstruct_layers runs after this).
  if (Bf12Companions::enabled()) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
      if (w != nullptr) bf12_.pack(w, n, k, gemm_, stream_);
    };
    const int64_t dH = dfcfg_.hidden_size, dI = df_i_;
    const int64_t QW = df_qw_, KV = df_kvr_;
    for (const DFlash2LayerWeights& w : dfw_.layers) {
      if (w.qkv_fp8 != nullptr) {
        pack(w.kv_rows, 2 * KV, dH);
        continue;
      }
      pack(w.qkv, QW + 2 * KV, dH);
      pack(w.o, dH, QW);
      pack(w.gate, dI, dH);
      pack(w.up, dI, dH);
      pack(w.down, dH, dI);
    }
    for (size_t t = 0; t < dfcfg_.target_layer_ids.size(); ++t)
      pack(dfw_.fc + t * static_cast<size_t>(dH) * dH, dH, dH);
    pack(dfw_.hidden_projection, dfcfg_.selector_rank, dH);
    bf12_ms_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000.0;
  }
  {
    std::vector<float> host(dfcfg_.head_dim / 2);
    for (size_t i = 0; i < host.size(); ++i)
      host[i] = static_cast<float>(
          std::pow(dfcfg_.rope_theta, -static_cast<double>(2 * i) / static_cast<double>(dfcfg_.head_dim)));
    DGPP_CUDA_OK(cudaMalloc(&df_inv_freq_, host.size() * 4));
    DGPP_CUDA_OK(cudaMemcpy(df_inv_freq_, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  }
  df_planes_.init(dfcfg_.num_hidden_layers, df_kvr_, max_requests_, dfcfg_.sliding_window,
                  dfcfg_.query_rows(), max_context_, stream_);
  const int QR = dfcfg_.query_rows();
  const int D = QR - 1;
  // One 8-row block fits this family's whole decode batch: the stacked
  // scratch stays single-slot (the graph engine's fixed batch is
  // unavailable at rows_per_request == max_decode_rows, so the recorded
  // draft only ever runs its scalar form).
  df_batch_ = std::max(1, kDecodeRows / QR);
  const int BR = df_batch_ * QR, BD = df_batch_ * D, CR = df_rows_cap();
  DGPP_CUDA_OK(cudaMalloc(&df_acc_, static_cast<size_t>(M) * H * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_t32_, static_cast<size_t>(M) * H * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_collapsed_, static_cast<size_t>(M) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_norm_, static_cast<size_t>(CR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_kv_, static_cast<size_t>(CR) * 2 * df_kvr_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_resid_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_x_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_xc_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_qkv_, static_cast<size_t>(BR) * (df_qw_ + 2 * df_kvr_) * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_q_, static_cast<size_t>(BR) * df_qw_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_attn_, static_cast<size_t>(BR) * df_qw_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_attn_part_, dflash2_block_attn_partials_bytes(BR, df_nh_)));
  DGPP_CUDA_OK(cudaMalloc(&df_o_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_mlp_, static_cast<size_t>(BR) * df_i_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_gate_, static_cast<size_t>(BR) * df_i_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_up_, static_cast<size_t>(BR) * df_i_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_zero_, QR * 4));
  DGPP_CUDA_OK(cudaMemsetAsync(df_zero_, 0, QR * 4, stream_));
  DGPP_CUDA_OK(cudaMalloc(&df_delta_, static_cast<size_t>(BR) * 2 * dfcfg_.conv_taps * dfcfg_.conv_groups() * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_h_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_logits_, static_cast<size_t>(BR) * lm_vocab_count_ * 4));
  df_topk_ws_bytes_ = dflash2_topk_ws_bytes(lm_vocab_count_, BR, dfcfg_.selector_top_k);
  DGPP_CUDA_OK(cudaMalloc(&df_topk_ws_, df_topk_ws_bytes_));
  DGPP_CUDA_OK(cudaMalloc(&df_hidden32_, static_cast<size_t>(BR) * dfcfg_.selector_rank * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_ids_, static_cast<size_t>(BR) * dfcfg_.selector_top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_sc_, static_cast<size_t>(BR) * dfcfg_.selector_top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_tok_, static_cast<size_t>(BD) * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_pos_, static_cast<size_t>(BR) * 8));
  DGPP_CUDA_OK(cudaMalloc(&df_tokens_, static_cast<size_t>(BR) * 8));
  DGPP_CUDA_OK(cudaMallocHost(&df_tok_h_, static_cast<size_t>(BD) * 4));
  DGPP_CUDA_OK(cudaMallocHost(&df_io64_h_, static_cast<size_t>(2) * BR * 8));
  DGPP_CUDA_OK(cudaMalloc(&df_conf_, static_cast<size_t>(max_requests_) * D * 4));
  DGPP_CUDA_OK(cudaMemsetAsync(df_conf_, 0, static_cast<size_t>(max_requests_) * D * 4, stream_));
  if (loader_.world() > 1)
    DGPP_CUDA_OK(cudaMalloc(&df_table_, dflash2_topk_table_elems(BR, dfcfg_.selector_top_k, loader_.world()) * 2));
  // The walk's per-row request ids and positions (the prefill tap store's
  // device sources; decode walks read d_req_ids_ / d_step_pos_ instead).
  DGPP_CUDA_OK(cudaMalloc(&df_walk_req_, static_cast<size_t>(M) * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_walk_pos_, static_cast<size_t>(M) * 8));
  DGPP_CUDA_OK(cudaMallocHost(&df_walk_req_h_, static_cast<size_t>(M) * 4));
  DGPP_CUDA_OK(cudaMallocHost(&df_walk_pos_h_, static_cast<size_t>(M) * 8));
  // The recorded drafts' pinned mirror, one row per request slot, and the
  // mask rows' candidate tables the engine's diagnostics read.
  DGPP_CUDA_OK(cudaMallocHost(&df_mirror_h_, static_cast<size_t>(max_requests_) * D * 4));
  DGPP_CUDA_OK(
      cudaMallocHost(&df_cands_h_, static_cast<size_t>(max_requests_) * D * dfcfg_.selector_top_k * 4));
  dflash2_ = true;
}

void GlmDiagnosticModel::dflash2_capture_tap(int layer, const uint16_t* streams, int T) {
  // The tap layer's boundary streams through the head's contraction (the
  // reference's hc_post materialization + mean(dim=1)) into the fc slice,
  // fp32-accumulated (the first tap seeds the accumulator).
  const int H = cfg_.hidden_size;
  launch_mhc_final_mean(streams, df_collapsed_, mhc_cfg_, T, stream_);
  const uint16_t* fc_t = dfw_.fc + static_cast<size_t>(df_tap_[static_cast<size_t>(layer)]) * H * H;
  gemm_.matmul(df_collapsed_, fc_t, df_t32_, T, H, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  dflash2_acc_f32(df_acc_, df_t32_, static_cast<int64_t>(T) * H,
                  df_tap_[static_cast<size_t>(layer)] == 0 ? 0 : 1, stream_);
}

void GlmDiagnosticModel::dflash2_store_features(int T, const int32_t* d_req, const int64_t* d_pos) {
  const int H = cfg_.hidden_size;
  const int L = dfcfg_.num_hidden_layers;
  const int KV = df_kvr_, KVH = df_kvh_, HD = dfcfg_.head_dim;
  const int32_t* tables = df_planes_.device_tables();
  const int64_t bpr = df_planes_.table_stride();
  const int bt = df_planes_.block_tokens();
  for (int off = 0; off < T; off += df_rows_cap()) {
    const int rows = std::min(df_rows_cap(), T - off);
    dflash2_norm_f32_bf16(df_acc_ + static_cast<size_t>(off) * H, dfw_.hidden_norm, df_norm_, rows, H,
                          dfcfg_.rms_norm_eps, stream_);
    for (int l = 0; l < L; ++l) {
      const uint16_t* kv_w = dfw_.layers[static_cast<size_t>(l)].kv_rows;
      gemm_.matmul(df_norm_, kv_w, df_kv_, rows, 2 * KV, H, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      dflash2_norm_rope_bf16(df_kv_, 2 * KV, dfw_.layers[static_cast<size_t>(l)].k_norm, d_pos + off,
                             df_inv_freq_, df_kv_, 2 * KV, rows, KVH, HD, dfcfg_.rms_norm_eps, stream_);
      qsa_kv_append(df_kv_, 2 * KV, df_kv_ + KV, 2 * KV, d_req + off, d_pos + off, rows, tables, bpr,
                    bt, KVH, HD, df_planes_.k_cache(l), df_planes_.v_cache(l), stream_);
    }
  }
}

void GlmDiagnosticModel::dflash2_store_walk(int req, int64_t token_start, int T) {
  // The prefill/group walk's rows as the store's device sources (decode
  // walks pass d_req_ids_ / d_step_pos_ directly).
  if (group_num_spans_ > 0) {
    int64_t row0 = 0;
    for (int sp = 0; sp < group_num_spans_; ++sp) {
      const int len = group_span_lens_[sp];
      const int32_t sreq = group_span_reqs_[sp];
      for (int t = 0; t < len; ++t) {
        df_walk_req_h_[static_cast<size_t>(row0 + t)] = sreq;
        df_walk_pos_h_[static_cast<size_t>(row0 + t)] = t;
      }
      row0 += len;
    }
  } else {
    for (int t = 0; t < T; ++t) {
      df_walk_req_h_[static_cast<size_t>(t)] = req;
      df_walk_pos_h_[static_cast<size_t>(t)] = token_start + t;
    }
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df_walk_req_, df_walk_req_h_, static_cast<size_t>(T) * 4,
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(df_walk_pos_, df_walk_pos_h_, static_cast<size_t>(T) * 8,
                               cudaMemcpyHostToDevice, stream_));
  dflash2_store_features(T, df_walk_req_, df_walk_pos_);
}

bool GlmDiagnosticModel::dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts) {
  if (!dflash2_) throw std::logic_error("dflash2_draft: no drafter loaded");
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  const int64_t pos = session_position(req);
  if (pos < 0 || pos + QR > max_context()) return false;
  // The planes' per-request rings always cover the window; the block
  // writes at [pos, pos + QR) through the alias table by construction.
  for (int j = 0; j < QR; ++j) {
    df_io64_h_[j] = pos + j;
    df_io64_h_[QR + j] = j == 0 ? bonus : dfcfg_.mask_token_id;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df_pos_, df_io64_h_, static_cast<size_t>(QR) * 8, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tokens_, df_io64_h_ + QR, static_cast<size_t>(QR) * 8, cudaMemcpyHostToDevice, stream_));
  // The [bonus, mask x D] embed rows (the Flash table is full on every
  // rank — one gather, no fold).
  dflash2_embed_rows(globals_.embed, df_tokens_, df_resid_, cfg_.vocab_size, QR, cfg_.hidden_size,
                     stream_);
  gemm_.set_bf12_wide(true);
  const int reqs[1] = {req};
  df_block_layers(1, reqs, /*capture=*/false);
  df_block_select(1, reqs, /*capture=*/false);
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tok_h_, df_tok_, static_cast<size_t>(D) * 4, cudaMemcpyDeviceToHost, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  drafts->assign(df_tok_h_, df_tok_h_ + D);
  return true;
}

void GlmDiagnosticModel::df_block_layers(int slots, const int* reqs, bool capture) {
  const int H = cfg_.hidden_size;
  const int QR = dfcfg_.query_rows();
  const int L = dfcfg_.num_hidden_layers, QW = df_qw_, KV = df_kvr_;
  const int KVH = df_kvh_, HD = dfcfg_.head_dim, NH = df_nh_;
  const int G = dfcfg_.conv_groups(), TAPS = dfcfg_.conv_taps;
  const int I = df_i_;
  const float eps = dfcfg_.rms_norm_eps;
  const int R = slots * QR;
  const auto stage = [&](uint16_t* fallback, int width) -> uint16_t* {
    if (!boundary_) return fallback;
    uint16_t* s = boundary_->stage(R, width);
    if (s == nullptr && capture)
      throw std::runtime_error("dflash2 draft: a capture fold does not fit the recorder's staged buffer");
    return s ? s : fallback;
  };
  // The sharded drafter's boundary folds (the walk's contract): the o and
  // down projections' partial [R, H] outputs are summed across the ranks
  // before the output-side convs read them. The bus's staged reduce is
  // host-driven — the producing GEMM must be complete before the fold
  // reads the staged buffer (the eager draft syncs; a captured draft's
  // recorded node orders it).
  const auto fold = [&](uint16_t* buf, int width) {
    if (!boundary_) return;
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    boundary_->reduce(buf, R, width);
  };
  const int32_t* tables = df_planes_.device_tables();
  const int64_t bpr = df_planes_.table_stride();
  const int bt = df_planes_.block_tokens();
  const size_t qkv_stride = static_cast<size_t>(QW) + 2 * KV;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  for (int l = 0; l < L; ++l) {
    const DFlash2LayerWeights& w = dfw_.layers[static_cast<size_t>(l)];
    dflash2_rmsnorm_bf16(df_resid_, w.input_norm, df_x_, R, H, eps, stream_);
    gemm_.matmul(df_x_, w.attn_conv_kp, df_delta_, R, 2 * TAPS * G, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    dflash2_grouped_conv_bf16(df_x_, df_delta_, w.attn_conv_base, df_xc_, R, QR, H, TAPS,
                              dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    // engine.dflash_weights fp8: the five block matrices stream as E4M3
    // with fp32 block-128 scales; lossy for the proposals only, the
    // verify stays exact whatever the drafter proposes.
    if (w.qkv_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.qkv_fp8, w.qkv_scales, df_qkv_, R,
                               static_cast<int>(qkv_stride), H, qkv_stride, 7, 7, stream_, gemm_ws_,
                               gemm_ws_bytes_);
    else
      gemm_.matmul(df_xc_, w.qkv, df_qkv_, R, static_cast<int>(qkv_stride), H, DType::BF16,
                   GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    dflash2_norm_rope_bf16(df_qkv_, qkv_stride, w.q_norm, df_pos_, df_inv_freq_, df_q_, QW, R, NH, HD,
                           eps, stream_);
    dflash2_norm_rope_bf16(df_qkv_ + QW, qkv_stride, w.k_norm, df_pos_, df_inv_freq_, df_qkv_ + QW,
                           qkv_stride, R, KVH, HD, eps, stream_);
    for (int k = 0; k < slots; ++k) {
      const size_t ro = static_cast<size_t>(k) * QR;
      const int32_t* table = tables + static_cast<size_t>(reqs[k]) * bpr;
      const uint16_t* krow = df_qkv_ + ro * qkv_stride + QW;
      qsa_kv_append(krow, qkv_stride, krow + KV, qkv_stride, df_zero_, df_pos_ + ro, QR, table, 1, bt,
                    KVH, HD, df_planes_.k_cache(l), df_planes_.v_cache(l), stream_);
      dflash2_block_attn_split(df_q_ + ro * QW, QW, df_planes_.k_cache(l), df_planes_.v_cache(l), table,
                               bt, bpr, QR, dfcfg_.sliding_window, df_pos_ + ro, QR, NH, KVH, HD, scale,
                               df_attn_part_ + ro * static_cast<size_t>(NH) * dflash2_block_attn_splits() * 130,
                               df_attn_ + ro * QW, stream_);
    }
    uint16_t* ao = stage(df_o_, H);
    if (w.o_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_attn_, static_cast<size_t>(QW), w.o_fp8, w.o_scales, ao, R, H, QW,
                               static_cast<size_t>(H), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
    else
      gemm_.matmul(df_attn_, w.o, ao, R, H, QW, DType::BF16, GemmOut::BF16, QW, gemm_ws_,
                   gemm_ws_bytes_, stream_);
    fold(ao, H);
    dflash2_grouped_conv_bf16(ao, df_delta_ + TAPS * G, w.attn_conv_base + static_cast<size_t>(TAPS) * H,
                              df_xc_, R, QR, H, TAPS, dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    dflash2_add_rmsnorm_bf16(df_resid_, df_xc_, w.post_norm, df_x_, R, H, eps, stream_);
    gemm_.matmul(df_x_, w.mlp_conv_kp, df_delta_, R, 2 * TAPS * G, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    dflash2_grouped_conv_bf16(df_x_, df_delta_, w.mlp_conv_base, df_xc_, R, QR, H, TAPS,
                              dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    if (w.gate_fp8 != nullptr) {
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.gate_fp8, w.gate_scales, df_gate_, R, I,
                               H, static_cast<size_t>(I), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.up_fp8, w.up_scales, df_up_, R, I, H,
                               static_cast<size_t>(I), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
    } else {
      gemm_.matmul(df_xc_, w.gate, df_gate_, R, I, H, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      gemm_.matmul(df_xc_, w.up, df_up_, R, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H),
                   gemm_ws_, gemm_ws_bytes_, stream_);
    }
    swiglu_limit_bf16(df_gate_, df_up_, df_mlp_, static_cast<int64_t>(R) * I, INFINITY, stream_);  // the drafter is a plain-SiLU backbone
    uint16_t* mo = stage(df_o_, H);
    if (w.down_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_mlp_, static_cast<size_t>(I), w.down_fp8, w.down_scales, mo, R, H, I,
                               static_cast<size_t>(H), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
    else
      gemm_.matmul(df_mlp_, w.down, mo, R, H, I, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(I), gemm_ws_, gemm_ws_bytes_, stream_);
    fold(mo, H);
    dflash2_grouped_conv_bf16(mo, df_delta_ + TAPS * G, w.mlp_conv_base + static_cast<size_t>(TAPS) * H,
                              df_xc_, R, QR, H, TAPS, dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    add_inplace_bf16(df_resid_, df_xc_, static_cast<int64_t>(R) * H, stream_);
  }
  dflash2_rmsnorm_bf16(df_resid_, dfw_.norm, df_h_, R, H, eps, stream_);
}

void GlmDiagnosticModel::df_block_select(int slots, const int* reqs, bool capture) {
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  const int R = slots * QR;
  const int64_t V = lm_vocab_count_;
  const int topk = dfcfg_.selector_top_k, rank = dfcfg_.selector_rank;
  // The recorded walk draws for a stochastic request (the armed
  // proposals); the eager walk is the argmax. Every walk with a slot list
  // writes the slots' confidence (the scheduled verify depth's input).
  const bool sampled = capture && df_specs_ != nullptr && reqs != nullptr;
  gemm_.matmul(df_h_, globals_.lm_head, df_logits_, R, static_cast<int>(V), cfg_.hidden_size,
               DType::BF16, GemmOut::F32, static_cast<size_t>(cfg_.hidden_size), gemm_ws_,
               gemm_ws_bytes_, stream_);
  dflash2_topk_f32(df_logits_, df_ids_, df_sc_, V, R, topk, stream_, df_topk_ws_, df_topk_ws_bytes_);
  if (loader_.world() > 1) {
    const size_t elems = dflash2_topk_table_elems(R, topk, loader_.world());
    uint16_t* table = boundary_ ? boundary_->stage(1, static_cast<int>(elems)) : nullptr;
    if (table == nullptr) {
      if (capture)
        throw std::runtime_error("dflash2 draft: the top-K gather does not fit the recorder's staged buffer");
      table = df_table_;
    }
    dflash2_topk_stage(df_ids_, df_sc_, R, topk, lm_vocab_begin_, loader_.rank(), loader_.world(), table, stream_);
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));  // the host-driven fold's contract
    if (boundary_) boundary_->reduce(table, 1, static_cast<int>(elems));
    dflash2_topk_merge(table, R, topk, loader_.world(), df_ids_, df_sc_, stream_);
  }
  gemm_.matmul(df_h_, dfw_.hidden_projection, df_hidden32_, R, rank, cfg_.hidden_size, DType::BF16,
               GemmOut::F32, static_cast<size_t>(cfg_.hidden_size), gemm_ws_, gemm_ws_bytes_, stream_);
  for (int k = 0; k < slots; ++k) {
    const size_t ro = static_cast<size_t>(k) * QR, co = static_cast<size_t>(k) * D, mo = ro + 1;
    const size_t po = sampled ? static_cast<size_t>(reqs[k]) * static_cast<size_t>(df_props_stride_) : 0;
    dflash2_selector_walk(df_ids_ + mo * topk, df_sc_ + mo * topk, df_hidden32_ + mo * rank,
                          dfw_.pred_codebook, dfw_.succ_codebook, df_tokens_ + ro, df_tok_ + co, D, topk,
                          rank, stream_, sampled ? df_pos_ + ro : nullptr,
                          sampled ? df_specs_ + reqs[k] : nullptr, sampled ? df_props_ + po : nullptr,
                          sampled && df_props_h_ != nullptr ? df_props_h_ + po : nullptr,
                          reqs != nullptr ? df_conf_ + static_cast<size_t>(reqs[k]) * D : nullptr);
  }
}

void GlmDiagnosticModel::dflash2_draft_batch(const std::vector<int>& reqs,
                                             const std::vector<int64_t>& bonuses,
                                             std::vector<std::vector<int32_t>>* drafts) {
  if (!dflash2_) throw std::logic_error("dflash2_draft_batch: no drafter loaded");
  if (reqs.empty()) throw std::invalid_argument("dflash2_draft_batch: no requests");
  if (reqs.size() != bonuses.size() || drafts == nullptr)
    throw std::invalid_argument("dflash2_draft_batch: reqs/bonuses/drafts shape");
  const int S = static_cast<int>(reqs.size());
  if (S > df_batch_)
    throw std::invalid_argument("dflash2_draft_batch: batch exceeds the stacked scratch");
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  drafts->assign(static_cast<size_t>(S), {});
  // Admission first (no shared state touched): positions per slot,
  // compacting out the failures (the planes' rings are preallocated).
  std::vector<int> idx, creqs;
  idx.reserve(static_cast<size_t>(S));
  std::vector<int64_t> poss(static_cast<size_t>(S), -1);
  for (int s = 0; s < S; ++s) {
    const int req = reqs[static_cast<size_t>(s)];
    const int64_t pos = session_position(req);
    if (pos < 0 || pos + QR > max_context()) continue;
    poss[static_cast<size_t>(s)] = pos;
    idx.push_back(s);
    creqs.push_back(req);
  }
  const int NS = static_cast<int>(idx.size());
  if (NS == 0) return;
  const int R = NS * QR;
  // Row inputs: each stacked block [bonus, mask x D] at its slot's positions.
  for (int k = 0; k < NS; ++k) {
    const int s = idx[static_cast<size_t>(k)];
    const int64_t pos = poss[static_cast<size_t>(s)];
    for (int j = 0; j < QR; ++j) {
      df_io64_h_[static_cast<size_t>(k) * QR + j] = pos + j;
      df_io64_h_[static_cast<size_t>(df_batch_) * QR + static_cast<size_t>(k) * QR + j] =
          j == 0 ? bonuses[static_cast<size_t>(s)] : dfcfg_.mask_token_id;
    }
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df_pos_, df_io64_h_, static_cast<size_t>(R) * 8, cudaMemcpyHostToDevice,
                               stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tokens_, df_io64_h_ + static_cast<size_t>(df_batch_) * QR,
                               static_cast<size_t>(R) * 8, cudaMemcpyHostToDevice, stream_));
  dflash2_embed_rows(globals_.embed, df_tokens_, df_resid_, cfg_.vocab_size, R, cfg_.hidden_size,
                     stream_);
  gemm_.set_bf12_wide(true);
  df_block_layers(NS, creqs.data(), /*capture=*/false);
  df_block_select(NS, creqs.data(), /*capture=*/false);
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tok_h_, df_tok_, static_cast<size_t>(NS) * D * 4,
                               cudaMemcpyDeviceToHost, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  for (int k = 0; k < NS; ++k)
    (*drafts)[static_cast<size_t>(idx[static_cast<size_t>(k)])].assign(
        df_tok_h_ + static_cast<size_t>(k) * D, df_tok_h_ + static_cast<size_t>(k + 1) * D);
}

void GlmDiagnosticModel::session_graph_capture_block_draft(int req, const PickVerdict* verdict) {
  if (!dflash2_) throw std::logic_error("session_graph_capture_block_draft: no drafter loaded");
  if (req < 0 || req >= max_requests_) throw std::invalid_argument("session_graph_capture_block_draft: request");
  if (verdict == nullptr) throw std::invalid_argument("session_graph_capture_block_draft: null verdict");
  if (!graph_device_positions_ || !graph_device_tokens_)
    throw std::logic_error(
        "session_graph_capture_block_draft: the step must be captured with device positions and tokens");
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  if ((graph_feed_rows_ > 0 ? graph_feed_rows_ : decode_rows_) != QR)
    throw std::logic_error("session_graph_capture_block_draft: the token feed is [next, drafts] (T = query rows)");
  // The block's rows off the verdict: the anchor is the verify's next
  // token at the position the recorded commit advanced to; mask rows
  // follow (positions past the context -1: no state write).
  dflash2_stage_block(verdict, d_session_pos_ + req, dfcfg_.mask_token_id, QR, max_context(), df_pos_,
                      df_tokens_, stream_);
  dflash2_embed_rows(globals_.embed, df_tokens_, df_resid_, cfg_.vocab_size, QR, cfg_.hidden_size,
                     stream_);
  gemm_.set_bf12_wide(true);
  const int reqs[1] = {req};
  df_block_layers(1, reqs, /*capture=*/true);
  df_block_select(1, reqs, /*capture=*/true);
  // The next replay's feed and the host's copy of the drafts.
  dflash2_block_feed(verdict, df_tok_, D, step_tokens_, stream_);
  dflash2_publish_drafts(df_tok_, df_mirror_h_ + static_cast<size_t>(req) * D, D, stream_);
  const int K = dfcfg_.selector_top_k;
  dflash2_publish_words(df_ids_ + K, df_cands_h_ + static_cast<size_t>(req) * D * K, D * K, stream_);
}

}  // namespace dgpp
