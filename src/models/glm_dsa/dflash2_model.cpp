// The DFlash2 drafter for GLM-5.3 (the port of models/qwen/dflash2's model
// integration, docs/performance_improvement_plan.md §7): block-diffusion
// drafting over the target's tap-layer residual streams. Eager v1 scope —
// the scalar greedy draft; graph capture, stacked redrafts and sampled
// proposals follow the Qwen sequence later.
//
// Deltas against the Qwen original, all forced by the target:
//   * the drafter's KV planes are a dedicated GlmDFlash2Planes ring (GLM's
//     cache pool is DSA-latent and cannot host plain GQA planes);
//   * the shared head is the target's vocab-sharded lm_head GEMM (Qwen's
//     fused head_gemv);
//   * the embed gather goes through the sharded-embedding scratch path.
#include <cmath>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/qsa.hpp"
#include "kernels/kernels.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/dflash2.hpp"
#include "models/glm_dsa/model.hpp"

namespace dgpp {

void GlmDsaModel::dflash2_init(const std::string& dflash2_dir) {
  dfcfg_ = DFlash2Config::from_json_file(dflash2_dir + "/config.json");
  dfcfg_.validate_against(cfg_);
  if (!dfcfg_.tp_divisible(tp_world_))
    throw std::invalid_argument("GlmDsaModel: the DFlash2 drafter's heads / kv heads / MLP rows must divide across the world");
  df_tap_.assign(cfg_.num_hidden_layers, -1);
  for (size_t t = 0; t < dfcfg_.target_layer_ids.size(); ++t)
    df_tap_[dfcfg_.target_layer_ids[t]] = static_cast<int>(t);
  const int H = cfg_.hidden_size;
  const int M = max_tokens_;
  df_nh_ = dfcfg_.local_heads(tp_world_);
  df_kvh_ = dfcfg_.local_kv_heads(tp_world_);
  df_qw_ = dfcfg_.local_q_row(tp_world_);
  df_kvr_ = dfcfg_.local_kv_row(tp_world_);
  df_i_ = dfcfg_.local_intermediate(tp_world_);
  dfw_ = load_dflash2_weights(dfcfg_, dflash2_dir, stream_, tp_rank_, tp_world_, false);
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
  const int BR = QR, CR = df_rows_cap();
  DGPP_CUDA_OK(cudaMalloc(&df_acc_, static_cast<size_t>(M) * H * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_t32_, static_cast<size_t>(M) * H * 4));
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
  DGPP_CUDA_OK(cudaMalloc(&df_embed_rows_, static_cast<size_t>(QR) * H * 2));
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
  DGPP_CUDA_OK(cudaMalloc(&df_tok_, static_cast<size_t>(D) * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_pos_, static_cast<size_t>(BR) * 8));
  DGPP_CUDA_OK(cudaMalloc(&df_tokens_, static_cast<size_t>(BR) * 8));
  DGPP_CUDA_OK(cudaMallocHost(&df_tok_h_, static_cast<size_t>(D) * 4));
  DGPP_CUDA_OK(cudaMallocHost(&df_io64_h_, static_cast<size_t>(2) * BR * 8));
  DGPP_CUDA_OK(cudaMalloc(&df_conf_, static_cast<size_t>(max_requests_) * D * 4));
  if (tp_world_ > 1)
    DGPP_CUDA_OK(cudaMalloc(&df_table_, dflash2_topk_table_elems(BR, dfcfg_.selector_top_k, tp_world_) * 2));
  log_memory_ledger("glm_dsa: dflash2 drafter resident");
  dflash2_ = true;
}

void GlmDsaModel::dflash2_store_features(int T, const int32_t* d_req, const int64_t* d_pos) {
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
      const uint16_t* kv_w = dfw_.layers[l].kv_rows;
      gemm_.matmul(df_norm_, kv_w, df_kv_, rows, 2 * KV, H, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      dflash2_norm_rope_bf16(df_kv_, 2 * KV, dfw_.layers[l].k_norm, d_pos + off, df_inv_freq_, df_kv_,
                             2 * KV, rows, KVH, HD, dfcfg_.rms_norm_eps, stream_);
      qsa_kv_append(df_kv_, 2 * KV, df_kv_ + KV, 2 * KV, d_req + off, d_pos + off, rows, tables, bpr,
                    bt, KVH, HD, df_planes_.k_cache(l), df_planes_.v_cache(l), stream_);
    }
  }
}

bool GlmDsaModel::dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts) {
  if (!dflash2_) throw std::logic_error("dflash2_draft: no drafter loaded");
  const int H = cfg_.hidden_size;
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
  // The [bonus, mask x D] embed rows (gather_embedding handles the
  // vocab-sharded embedding's gather+fold at world > 1).
  gather_embedding(df_tokens_, df_resid_, QR, false);
  gemm_.set_bf12_wide(true);
  const int reqs[1] = {req};
  df_block_layers(1, reqs, /*capture=*/false);
  df_block_select(1, reqs, /*capture=*/false);
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tok_h_, df_tok_, static_cast<size_t>(D) * 4, cudaMemcpyDeviceToHost, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  drafts->assign(df_tok_h_, df_tok_h_ + D);
  return true;
}

void GlmDsaModel::df_block_layers(int slots, const int* reqs, bool capture) {
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
    return s ? s : fallback;
  };
  const auto fold = [&](uint16_t* buf, int width) {
    if (!boundary_) return;
    boundary_->reduce(buf, R, width);
  };
  const int32_t* tables = df_planes_.device_tables();
  const int64_t bpr = df_planes_.table_stride();
  const int bt = df_planes_.block_tokens();
  const size_t qkv_stride = static_cast<size_t>(QW) + 2 * KV;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  for (int l = 0; l < L; ++l) {
    const DFlash2LayerWeights& w = dfw_.layers[l];
    dflash2_rmsnorm_bf16(df_resid_, w.input_norm, df_x_, R, H, eps, stream_);
    gemm_.matmul(df_x_, w.attn_conv_kp, df_delta_, R, 2 * TAPS * G, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    dflash2_grouped_conv_bf16(df_x_, df_delta_, w.attn_conv_base, df_xc_, R, QR, H, TAPS,
                              dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
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
    gemm_.matmul(df_xc_, w.gate, df_gate_, R, I, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    gemm_.matmul(df_xc_, w.up, df_up_, R, I, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    swiglu_limit_bf16(df_gate_, df_up_, df_mlp_, static_cast<int64_t>(R) * I, INFINITY, stream_);
    uint16_t* mo = stage(df_o_, H);
    gemm_.matmul(df_mlp_, w.down, mo, R, H, I, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(I), gemm_ws_, gemm_ws_bytes_, stream_);
    fold(mo, H);
    dflash2_grouped_conv_bf16(mo, df_delta_ + TAPS * G, w.mlp_conv_base + static_cast<size_t>(TAPS) * H,
                              df_xc_, R, QR, H, TAPS, dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    add_inplace_bf16(df_resid_, df_xc_, static_cast<int64_t>(R) * H, stream_);
  }
  dflash2_rmsnorm_bf16(df_resid_, dfw_.norm, df_h_, R, H, eps, stream_);
}

void GlmDsaModel::df_block_select(int slots, const int* reqs, bool capture) {
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  const int R = slots * QR;
  const int64_t V = lm_vocab_count_;
  const int topk = dfcfg_.selector_top_k, rank = dfcfg_.selector_rank;
  gemm_.matmul(df_h_, globals_.lm_head, df_logits_, R, static_cast<int>(V), cfg_.hidden_size,
               DType::BF16, GemmOut::F32, static_cast<size_t>(cfg_.hidden_size), gemm_ws_,
               gemm_ws_bytes_, stream_);
  dflash2_topk_f32(df_logits_, df_ids_, df_sc_, V, R, topk, stream_, df_topk_ws_, df_topk_ws_bytes_);
  if (tp_world_ > 1) {
    const size_t elems = dflash2_topk_table_elems(R, topk, tp_world_);
    uint16_t* table = boundary_ ? boundary_->stage(1, static_cast<int>(elems)) : nullptr;
    if (table == nullptr) table = df_table_;
    dflash2_topk_stage(df_ids_, df_sc_, R, topk, lm_vocab_begin_, tp_rank_, tp_world_, table, stream_);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));  // the host-driven fold's contract
    if (boundary_) boundary_->reduce(table, 1, static_cast<int>(elems));
    dflash2_topk_merge(table, R, topk, tp_world_, df_ids_, df_sc_, stream_);
  }
  gemm_.matmul(df_h_, dfw_.hidden_projection, df_hidden32_, R, rank, cfg_.hidden_size, DType::BF16,
               GemmOut::F32, static_cast<size_t>(cfg_.hidden_size), gemm_ws_, gemm_ws_bytes_, stream_);
  for (int k = 0; k < slots; ++k) {
    const size_t ro = static_cast<size_t>(k) * QR, co = static_cast<size_t>(k) * D, mo = ro + 1;
    dflash2_selector_walk(df_ids_ + mo * topk, df_sc_ + mo * topk, df_hidden32_ + mo * rank,
                          dfw_.pred_codebook, dfw_.succ_codebook, df_tokens_ + ro, df_tok_ + co, D, topk,
                          rank, stream_, nullptr, nullptr, nullptr, nullptr,
                          reqs != nullptr ? df_conf_ + static_cast<size_t>(reqs[k]) * D : nullptr);
  }
}

}  // namespace dgpp
