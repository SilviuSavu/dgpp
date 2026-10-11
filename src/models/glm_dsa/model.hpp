#pragma once
// The full GLM-5.3 (GlmMoeDsaForCausalLM) on the shared session core
// (2026-09-12, docs/glm53_plan.md G5). The layer walk follows transformers'
// GlmMoeDsaDecoderLayer:
//
//   h = embed(x)
//   per layer: h += attn(input_norm(h))   (MLA with the decoupled interleaved
//                                          RoPE, the DSA indexer on the "full"
//                                          layers, the last full layer's
//                                          selection on the "shared" ones)
//              h += mlp(post_norm(h))     (the BF16 dense MLP on the first
//                                          layers, the sigmoid-routed MoE with
//                                          its packed-int experts after)
//   logits = lm_head(final_norm(h))
//
// One row walk (run_rows) serves every entry point — the cold diagnostic
// forward, a session's prefill chunks, the eager decode rows and the
// captured decode graphs — the engine/session_model.hpp core owns
// everything around it (positions, feeds, chunking, snapshots, the graph
// era, the draft block's plumbing).
//
// Attention state per request slot: its row of the DSA pool's block table
// (models/dsa_state.hpp: the latent rows with their rope keys on every
// layer, the fp8 index cache on the 21 indexed layers, 128-token blocks).
// Per-token selection (index_kpool 1) leaves no tail-ring state that any
// later row reads, so — as for GLM-4.7 — a rejected verify row leaves
// stale cache rows past the position that nothing reads: the speculative
// commit table is empty and the prefix snapshot is the block list plus the
// draft's hidden row.
//
// The shared selection (plan D4): the walk enqueues a "full" layer with
// its indexer view and each following "shared" layer with a view that
// carries no indexer tensors; DsaLayer then reuses the topk/counts its
// last indexed enqueue left in the (one, shared) scratch, and checks the
// call shape matches. Nothing else touches that scratch between the two.
//
// TP: `tp_world` > 1 loads this rank's slices (64/W heads, every expert's
// intermediate slice, the dense MLP's slice, the lm-head vocab slice) and
// folds the attention and MLP block outputs per layer through `boundary`.
// The indexer, the fused q_a/kv_a projection, the router and the norms
// are replicated; every rank's residual is bitwise the others'.
//
// The MTP draft block (plan §1.4, D5): the draft layer resident beside the
// stack (its BF16 experts requantized to the packed format at load), one
// more pool layer (with its own indexer), the hidden window per slot; its
// input is eh_proj([enorm(embed(tok_{q+1})) | hnorm(h_q)]) with h_q the
// main stack's output hidden at q (POST final norm — vLLM's DeepSeek MTP
// convention; the alternative is one flag away and acceptance decides).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/arena.hpp"
#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/l2_prefetch.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/sample_pick.hpp"
#include "models/dsa_geometry.hpp"
#include "models/dsa_layer.hpp"
#include "models/dsa_state.hpp"
#include "models/glm/moe_layer.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/dflash2.hpp"
#include "models/glm_dsa/loader.hpp"

namespace dgpp {

class GlmDsaModel : public SessionModel<GlmDsaModel> {
 public:
  // Several cold prompts as the spans of one walk (session_prefill_group,
  // 2026-09-14, the group prefill ported from DeepSeek): the DSA attention
  // runs per span (its selection state and cache are per request), the
  // dense, MoE and head sites over every row; every span within max_tokens.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using Base = SessionModel<GlmDsaModel>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  // max_tokens bounds a walk's rows (a prefill chunk, the diagnostic
  // forward); max_cache_tokens the pool's capacity in tokens (rounded up
  // to a block) — shared by every slot (max_context()). max_requests: the
  // session slots (at most 16: kPickMaxRequests' verdict bound, plan D9 — the fused
  // select runs its rows in groups of eight). mtp: the draft block.
  // decode_rows: the fixed decode batch's row ceiling (at most 32, the
  // same bound; 0 = kDecodeRows). latent_format:
  // the latent cache's storage (bf16, fp8 or fp4; the rope key and the
  // index cache are unaffected).
  GlmDsaModel(const GlmDsaTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
              int64_t max_cache_tokens, GlmDsaResidency residency = GlmDsaResidency::Streaming,
              BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
              int max_requests = 1, bool mtp = false, int decode_rows = 0,
              LatentFormat latent_format = LatentFormat::kBf16, bool serving_logits = false,
              const std::string& dflash2_dir = "");
  ~GlmDsaModel();
  GlmDsaModel(const GlmDsaModel&) = delete;
  GlmDsaModel& operator=(const GlmDsaModel&) = delete;
  // engine.dflash_weights (checkpoint | fp8): the drafter's five block
  // matrices stream as E4M3 payloads with fp32 block-128 scales instead of
  // bf16 (the Qwen lane's recipe; lossy for the proposals only — the
  // verify is exact whatever the drafter proposes). Set before
  // construction, like Qwen35Model::set_dflash2_weights_fp8.
  static void set_dflash2_weights_fp8(bool on) { dflash2_fp8_ = on; }

  // Every byte the constructor (and its layer objects) will allocate for
  // a shape, from the same formulas BEFORE anything is allocated.
  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const GlmDsaTextConfig& cfg, int max_tokens,
                                int64_t max_cache_tokens, int tp_rank = 0, int tp_world = 1,
                                GlmDsaResidency residency = GlmDsaResidency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0,
                                LatentFormat latent_format = LatentFormat::kBf16,
                                bool serving_logits = false,
                                const std::string& dflash2_dir = "");

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  // capture_layers: every layer's residual output [T, H] in layer_states.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);
  // The draft block over a prompt (the parity gate's surface): the cold
  // forward, then the draft rows q = 0 .. T-2 (token q+1, the hidden at q)
  // with the head on every row — logits [T-1, count], the draft's normed
  // final hidden [T-1, hidden].
  Outputs mtp_forward(const std::vector<int64_t>& token_ids);

  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  // The decode batch's cap (plan D9): eight rows until 2026-09-13, sixteen
  // until 2026-10-11, thirty-two for a day, sixty-four since (plan D9b's
  // rungs: the DFlash2 lane's 8-row blocks — eight slots a pass);
  // the fused select launches them in groups of eight, the pick kernels
  // verdict sixteen request slots (kDecodeRowsMax bounds the row arrays).
  static constexpr int decode_rows_cap() { return 64; }
  // The same number for a shape that is not built yet (the memory plan).
  static size_t session_snapshot_bytes(const GlmDsaTextConfig& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;
  // The DSA configuration the pool and the layer run (the geometry's
  // formulas for a shape).
  static DsaConfig dsa_config(const GlmDsaTextConfig& cfg, int tp_world, bool mtp, LatentFormat format);
  // One cache block's bytes across every pool plane (the NVMe cold tier's
  // record, issue #26) for a shape that is not built yet.
  static size_t kv_block_bytes_static(const GlmDsaTextConfig& cfg, int tp_world, bool mtp, LatentFormat format);

  const GlmDsaTextConfig& config() const { return cfg_; }
  const DsaStatePool& dsa_pool() const { return pool_; }
  const DsaConfig& dsa_cfg() const { return dsa_cfg_; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const { return kSnapshotStamp; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  DsaStatePool& pool() { return pool_; }
  const DsaStatePool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests);
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  // Depth >= 2: the chain rows take the block's output residual (pre
  // shared_head.norm) from mtp_r_; no state to snapshot around them.
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  const uint16_t* draft_hidden_rows() const { return mtp_r_; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

  // ---- the DFlash2 drafter (the GLM port, src/models/glm_dsa/dflash2.*) ----
  // Every run_rows pass also feeds the drafter: the tap layers' residual
  // streams accumulate through the split fc, get the hidden_norm, and land
  // as each draft layer's K/V in the drafter's own window planes at the
  // run's positions (rejected verify rows land too and are masked by
  // position — the planes need no rollback). dflash2_draft runs the block
  // [bonus, mask x drafts] through the six bidirectional layers over the
  // plane context; each mask row's top-K goes through the reference
  // chained selector walk, which proposes the following `drafts()` tokens.
  // The graph worlds record the same block forward inside the step's
  // replay (session_graph_capture_block_draft[_batch]) off the device
  // verdict, feeding the next replay's rows and the drafts' pinned mirror.
  // Public: the engines bind the drafter through requires-probes on these.
  bool dflash2_enabled() const { return dflash2_; }
  int dflash2_drafts() const { return dflash2_ ? dfcfg_.drafts() : 0; }
  bool dflash_verify_graph() const { return false; }
  bool dflash_graph_verify_available() const { return false; }
  bool dflash_draft_batch() const { return dflash2_ && df_batch_ > 1; }
  int dflash_depth() const { return -1; }
  // False without a draft (the context bound): the caller runs the step
  // without speculation.
  bool dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts);
  // The stacked redraft: one block forward for every slot (the row-wise
  // GEMMs/norms/convs over slots*query_rows rows; the per-slot calls at
  // row offsets). drafts[i] is slot i's proposals, empty when that slot
  // has no draft.
  void dflash2_draft_batch(const std::vector<int>& reqs, const std::vector<int64_t>& bonuses,
                           std::vector<std::vector<int32_t>>* drafts);
  int dflash2_batch_slots() const { return df_batch_; }
  // The recorded block draft (the graph engine's block proposal at every
  // world): off the verify's device verdict — the anchor is its next
  // token, the block's positions follow the committed device position —
  // the block forward, the mask rows' head rows and their top-K (world>1:
  // the ranks' slice lists merged through one boundary fold), the selector
  // walk, the slot's next feed [next, drafts] and the drafts' pinned
  // mirror. Kernel nodes only.
  void session_graph_capture_block_draft(int req, const PickVerdict* verdict);
  // The fixed batch's form: every slot's block off its verify verdict
  // (slot q == request q) in one stacked block forward; the feeds and
  // mirrors of all slots. An inactive slot's rows run at position -1 (no
  // state write) and its feed is zeroed.
  void session_graph_capture_block_draft_batch(const PickVerdict* verdicts, int requests);
  // The sampled requests' block proposals (the engine's draft rule): the
  // recorded walk draws each draft from the selector's softmax at the
  // request's draft temperature and writes the set it drew from to
  // proposals[req * stride + l] (and the pinned mirror), the next verify's
  // ratio rule; a greedy request keeps the argmax and n = 0. `specs` is
  // the engine's device SampleSpec table (one per slot). Null specs: the
  // argmax walk, no proposals. Before the first capture.
  void dflash2_arm_proposals(const SampleSpec* specs, DraftProposal* proposals,
                             DraftProposal* proposals_host, int stride) {
    if (specs != nullptr && (proposals == nullptr || stride < dfcfg_.drafts()))
      throw std::invalid_argument("dflash2_arm_proposals: proposals / stride");
    df_specs_ = specs;
    df_props_ = proposals;
    df_props_h_ = proposals_host;
    df_props_stride_ = stride;
  }
  bool dflash2_proposals_armed() const { return df_specs_ != nullptr; }
  // The selector's confidence per draft, [max_requests][drafts] written by
  // every walk (recorded or eager) at the slot's row — the scheduled
  // verify depth's input for the block drafter (engine.mtp_schedule).
  const float* dflash2_confidence() const { return df_conf_; }
  // The mirror the recorded draft of slot `req` publishes (readable once
  // the replay's end event has passed).
  const int32_t* block_drafts_host(int req) const {
    return df_mirror_h_ + static_cast<size_t>(req) * static_cast<size_t>(dfcfg_.drafts());
  }
  // The mask rows' top-k candidate ids the recorded draft walked
  // ([drafts][block_candidates()] per slot; the engine's acceptance
  // diagnostics rank the verify's correction among them).
  int block_candidates() const { return dfcfg_.selector_top_k; }
  const int32_t* block_candidates_host(int req) const {
    return df_cands_h_ + static_cast<size_t>(req) * static_cast<size_t>(dfcfg_.drafts()) *
                         static_cast<size_t>(dfcfg_.selector_top_k);
  }
 private:
  static constexpr int kBlockTokens = 128;
  static constexpr int kPrefillChunkTokens = 2048;
  inline static bool dflash2_fp8_ = false;  // set_dflash2_weights_fp8
  // No state families beside the pool's blocks: a 16-byte zero stamp keeps
  // the prefix arena's slots addressable (it refuses a zero-byte model).
  static constexpr size_t kSnapshotStamp = 16;
  static constexpr int kDsaDecodeSplit = 32;
  static constexpr size_t kDotBudget = 64ull << 20;

  void build_layer_objects(const GlmDsaLayerResident& r);
  DsaLayerWeights dsa_view(const GlmDsaAttnResident& a) const;
  static GlmMoeWeights moe_view(const GlmDsaMoeResident& m);
  int moe_ordinal(int layer) const { return layer - cfg_.first_k_dense_replace; }
  int table_slots() const;
  int pool_layers() const { return cfg_.num_hidden_layers + (mtp_ ? 1 : 0); }
  static std::vector<int> index_ordinals(const GlmDsaTextConfig& cfg, bool mtp);
  // The layer's two blocks over `resid` (the residual, updated in place):
  // the attention block and the MLP/MoE block with their folds.
  struct WalkRows {
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    const int32_t* spans = nullptr;
    int num_requests = 1;
    int req = 0;
    int64_t pos0 = 0;          // prefill: the chunk's first position
    bool decode = false;
    bool capture = false;
    int moe_table_slot = -1;   // >= 0: the capture's graph table
    MoeTraceStaging* trace = nullptr;
    // A group prefill (session_prefill_group, 2026-09-14): the walk's rows
    // are several requests' spans, span-major; the attention runs per span
    // (its selection state and cache are per request), everything else
    // over all rows. 0 spans: the one-request prefill above.
    const int32_t* span_reqs = nullptr;
    const int64_t* span_pos0 = nullptr;
    const int32_t* span_lens = nullptr;
    int num_spans = 0;
  };
  void enqueue_layer(const GlmDsaLayerResident& r, int pool_layer, uint16_t* resid, int T, const WalkRows& rows);
  void enqueue_dense(const GlmDsaDenseMlpResident& d, const uint16_t* x, int T, uint16_t* out);
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);

  // The vocab-sharded embedding (the loader's setting at world > 1):
  // each rank gathers the rows it holds and one fold sums them — bitwise
  // the replicated lookup. The draft's rows go through a scratch the
  // fused input kernel then reads by row (an iota index).
  bool embed_sharded_ = false;
  uint16_t* mtp_embed_rows_ = nullptr;  // [max_tokens, hidden]
  int64_t* d_iota_ = nullptr;           // 0..max_tokens-1
  void gather_embedding(const int64_t* tokens, uint16_t* dst, int T, bool capture);

  GlmDsaTextConfig cfg_;
  GlmDsaLayerStream loader_;
  CublasLtGemm gemm_;
  // The bf16 decode weights' 12-bit companions (kernels/bf12_companions.hpp),
  // their planned bytes, and the rows of the walk in flight (the prefetch
  // windows' view).
  static constexpr int kBf12ExpandSlots = 1;
  void pack_layer_companions(int layer, const GlmDsaLayerResident& r);
  void pack_head_companion();
  void finish_companions();
  void prefetch_bf16(const uint16_t* w, size_t bytes);
  static size_t bf12_plan_bytes(const GlmDsaTextConfig& cfg, int tp_rank, int tp_world, GlmDsaHeadSharding head,
                                bool mtp);
  static size_t bf12_slot_bytes(const GlmDsaTextConfig& cfg, int tp_rank, int tp_world, GlmDsaHeadSharding head,
                                bool mtp);
  Bf12Companions bf12_;
  double bf12_s_ = 0.0;
  int walk_rows_ = 1;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  GlmDsaGlobalsResident globals_;
  GlmMoeConfig moe_cfg_;
  DsaConfig dsa_cfg_;
  LatentFormat latent_format_ = LatentFormat::kBf16;

  // Layer objects (built at first use, rebound per layer).
  std::unique_ptr<DsaLayer> dsa_;
  std::unique_ptr<GlmMoeLayer> moe_;

  // The DSA pool (every layer's latent rows, the indexed layers' caches)
  // and the layer's shared scratch, both from one arena.
  Arena arena_;
  DsaStatePool pool_;
  void* dsa_scratch_ = nullptr;
  size_t dsa_scratch_bytes_ = 0;
  // The rotary table (bf16 [positions][2][rope/2]) the layer and the
  // oracle read; positions = the pool's token capacity.
  uint16_t* rope_table_ = nullptr;
  int64_t rope_positions_ = 0;

  // Activations [M rows] (the token rows, the head's outputs and the tail
  // mirrors are the core's).
  uint16_t* resid_ = nullptr;  // [M, H] the residual
  uint16_t* x_ = nullptr;      // [M, H] the normed block input
  uint16_t* y_ = nullptr;      // [M, H] the block output (the fold's fallback)
  // The BF16 dense MLP's scratch [M, I/W] (the first layers).
  uint16_t* dense_g_ = nullptr;
  uint16_t* dense_u_ = nullptr;
  uint16_t* dense_act_ = nullptr;
  // The prefill walk's route traces: one pinned staging slot per MoE layer
  // ([moe layers + draft][M][top_k]), materialized after the walk's final
  // sync.
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;

  // The L2 weight prefetcher (the boundary windows, bit-identical on or
  // off; engine.l2_prefetch false A/Bs).
  WeightPrefetcher prefetch_;
  size_t prefetch_window_bytes_ = 0;
  void prefetch_ffn_side(const GlmDsaLayerResident& r);
  void prefetch_attention_side(int layer);
  void prefetch_head();
  void prefetch_packed(const GlmPackedMatrix& m);

  // The draft block (mtp_): its fusion scratch and residual (the window,
  // the counters and the feeds are the core's).
  uint16_t* mtp_h_ = nullptr;   // [M, H] the gathered hidden rows (decode rows)
  uint16_t* mtp_in_ = nullptr;  // [M, 2H] [enorm(embed) | hnorm(h)]
  uint16_t* mtp_r_ = nullptr;   // [M, H] the draft's residual

  void dflash2_init(const std::string& dflash2_dir);
  void dflash2_store_features(int T, const int32_t* d_req, const int64_t* d_pos);
  void df_block_layers(int slots, const int* reqs, bool capture);
  void df_block_select(int slots, const int* reqs, bool capture);
  bool dflash2_ = false;
  DFlash2Config dfcfg_;
  DFlash2Weights dfw_;
  std::vector<int> df_tap_;
  GlmDFlash2Planes df_planes_;
  int df_nh_ = 0, df_kvh_ = 0, df_qw_ = 0, df_kvr_ = 0, df_i_ = 0;
  float* df_inv_freq_ = nullptr;
  float *df_acc_ = nullptr, *df_t32_ = nullptr;
  uint16_t *df_norm_ = nullptr, *df_kv_ = nullptr;
  uint16_t *df_resid_ = nullptr, *df_x_ = nullptr, *df_xc_ = nullptr, *df_qkv_ = nullptr;
  uint16_t *df_q_ = nullptr, *df_attn_ = nullptr, *df_o_ = nullptr, *df_mlp_ = nullptr;
  uint16_t *df_gate_ = nullptr, *df_up_ = nullptr, *df_h_ = nullptr;
  uint16_t* df_table_ = nullptr;
  float* df_attn_part_ = nullptr;
  int32_t* df_zero_ = nullptr;
  uint16_t* df_delta_ = nullptr;
  float* df_logits_ = nullptr;
  void* df_topk_ws_ = nullptr;
  size_t df_topk_ws_bytes_ = 0;
  float* df_hidden32_ = nullptr;
  int32_t *df_ids_ = nullptr, *df_tok_ = nullptr, *df_tok_h_ = nullptr;
  float* df_sc_ = nullptr;
  float* df_conf_ = nullptr;
  int64_t *df_pos_ = nullptr, *df_tokens_ = nullptr, *df_io64_h_ = nullptr;
  // The stacked draft batch (dflash2_draft_batch / the batched recorded
  // draft): slots per block forward at the decode batch's row ceiling.
  int df_batch_ = 1;
  // The recorded drafts' pinned mirrors (read after the replay's end
  // event) and the engine's armed sampled-proposal tables.
  int32_t* df_mirror_h_ = nullptr;
  int32_t* df_cands_h_ = nullptr;
  const SampleSpec* df_specs_ = nullptr;
  DraftProposal* df_props_ = nullptr;
  DraftProposal* df_props_h_ = nullptr;
  int df_props_stride_ = 0;
  static constexpr int df_rows_cap() { return 2048; }
  int tp_rank_ = 0, tp_world_ = 1;
};

}  // namespace dgpp
