# GLM-5.3 (HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64) on dgpp — architecture facts and implementation plan (2026-09-12)

Status: plan. Nothing is built. The checkpoint was downloading into rank 0's
Hub cache while this was written (78 per-layer shards plus
`passthrough.safetensors`, 427.4 GB / 398.1 GiB, 175,985 tensors); the peer
copies follow. The architecture study uses the repository's `config.json`,
`model.safetensors.index.json`, the headers of the shards already on disk,
transformers' `modeling_glm_moe_dsa.py` (main, 2026-09), vLLM's
`deepseek_v2.py` / `deepseek_mtp.py` for the draft-layer and index-sharing
conventions, and the checkpoint's own model card (written by this project's
quantization run, `docs/model_cards` will get a copy). Companion checkpoint
in the same format: `HawkBearPig/GLM-5.3-Int4-Int8Mix-AWQ-g64` (per-layer
equivalent; loads unchanged once this plan is built).

This is the fourth served family. Per PLAN.md it goes on the shared cores
(`SessionModel<Derived>`, `ResidentLayerStream<Family>`, `WeightBuilder`,
`DsaLayer`, `GlmMoeLayer`), never on copies of GLM-5.3-Flash's model class.

## 0. Summary

Full GLM-5.3 (`GlmMoeDsaForCausalLM`, `model_type glm_moe_dsa`) is a
78-layer plain pre-norm MoE with DeepSeek-V3-style MLA attention and a DSA
top-2048 indexer on every layer: 3 dense-MLP layers then 75 MoE layers of
256 routed experts (top-8, sigmoid + correction bias, `routed_scaling_factor`
2.5, `noaux_tc`) plus one shared expert of 2048; hidden 6144; 64 heads with
`qk_nope` 192 + `qk_rope` 64 and `v_head` 256; `q_lora` 2048, `kv_lora` 512;
indexer 32 heads × 128 with interleaved RoPE on its first 64 dims; one MTP
draft layer (`model.layers.78`); vocab 154,880; positions 1,048,576;
`rope_theta` 8e6 with no scaling. 754 B parameters.

Relative to GLM-5.3-Flash it **drops** KDA linear attention, the mHC
hyper-connection residual, the vision tower, `swiglu_limit` and the pooled
indexer (`index_kpool`), and **adds** four things the engine does not have:
decoupled RoPE inside MLA (a 64-wide rope key per token beside the 512
latent), interleaved RoPE inside the indexer, cross-layer top-k sharing
(only 21 of the 78 layers run an indexer; the other 57 attend with the last
full layer's selection), and per-token selection (`kpool` = 1, `select_k` =
2048). The checkpoint adds a **third weight format**: compressed-tensors
`pack-quantized` int4 (routed experts) and int8 (attention projections and
the shared expert), both symmetric, group 64 along K, bf16 scales, no zero
points.

Reuse is large: the DSA layer object and its three attention kernels, the
fused decode select and the prefill select, the latent cache formats
(bf16/fp8/fp4), the noaux_tc router and expert kernels, the two-rounding
norm, the MTP input kernel and draft plumbing, the GEMV row geometry, the
loader builder, the resident image, the session/graph/prefix/MTP core, the
GLM tokenizer (byte-identical `tokenizer.json`) and the tool-call format.
New: the family (config, binding, loader, layer walk, hooks), the int4/int8
g64 GEMV cores and their slot/grouped/tile forms, the rope-tail latent row,
the shared-selection walk, kpool 1 in the select networks, a chat-template
golden set, and the draft-layer requantization at load.

The numbers that frame the build (all derived, none measured; §2): resident
weights ≈ 99 GiB per rank at world 4, decode traffic ≈ 10 GB per rank per
token (a 42 ms floor at 240 GB/s; GLM-4.7 sits at 9.7 GB and 49 ms measured),
about 158 recorded collectives per step, and a context budget of roughly 2 to
4 GiB per rank after the weights, i.e. tens of thousands of tokens at four
request slots. The model fits four Sparks; it does not fit two.

## 1. The model

### 1.1 Configuration facts

The config is flat (no `text_config`); the fields below are the full model's
with the Flash value beside it where it differs.

| field | full GLM-5.3 | GLM-5.3-Flash (`text_config`) |
|---|---|---|
| `architectures` / `model_type` | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` | `Glm5NextForConditionalGeneration` / `glm5_next` |
| layers | 78 + draft layer 78 (`num_nextn_predict_layers` 1) | 45 + draft 45 |
| `layer_types` | absent — every layer is DSA/MLA | 34 KDA + 11 DSA |
| `mhc`, `hc_*`, `linear_attn_config`, `swiglu_limit`, `vision_config` | absent | present |
| hidden / vocab / eps | 6144 / 154,880 / 1e-5, silu, untied head | 4096 / same / same |
| attention | 64 heads; `q_lora_rank` 2048, `kv_lora_rank` 512; `qk_nope_head_dim` 192, `qk_rope_head_dim` 64 (`qk_head_dim` 256), `v_head_dim` 256; `attention_bias` false | 64 heads; 1536 / 512; nope 256, **rope 0** (`mla_use_nope`), v 256 |
| RoPE | `rope_theta` 8,000,000, `rope_type default`, `rope_interleave` true, `max_position_embeddings` 1,048,576 | none in MLA |
| indexer | `index_n_heads` 32, `index_head_dim` 128, `index_topk` 2048, `indexer_rope_interleave` true; **no `index_kpool`** (per-token selection) | 32 × 128, top-k 2048, `index_kpool` 4 with compress and always-select-tail |
| indexer schedule | `indexer_types` = `full` at layers 0, 1, 2, 6, 10, …, 74 (`index_skip_topk_offset` 3, `index_topk_freq` 4), `shared` elsewhere; `index_share_for_mtp_iteration` true | all 11 DSA layers `full` |
| dense MLP | `intermediate_size` 12,288 in layers 0–2 (`first_k_dense_replace` 3) | same |
| MoE | 256 experts, top-8, `n_group` = `topk_group` = 1, sigmoid + `e_score_correction_bias`, `norm_topk_prob`, × 2.5, `moe_intermediate_size` 2048, one shared expert, `moe_router_dtype` float32 | 288 experts, otherwise the same |
| tokens | eos [154820, 154827, 154829], pad 154820; `generation_config.json`: temperature 1.0, top_p 0.95 | same ids |
| quantization | compressed-tensors `pack-quantized`, two groups (§1.5) | `dgpp_mixed` (NVFP4 experts + FP8) |

The engine's current GLM-5.3 parser rejects this config on six counts
(`src/models/glm/config.cpp`): the missing `text_config` nesting, the missing
`layer_types` / `linear_attn_config` / `mhc`, `qk_rope_head_dim` ≠ 0
("the rope-free MLA path only"), `indexer_types` containing `shared`, and the
missing `index_kpool`. `loaders/architecture.cpp` does not recognise
`GlmMoeDsaForCausalLM` at all (its `glm_moe_dsa` fallback only fires when
`architectures` is absent, and maps to the Flash family). So the family is a
new parser and a new `ModelArchitecture` enumerator, not a variant of the
Flash one.

### 1.2 One layer (transformers `GlmMoeDsaDecoderLayer`)

```
h   = RMSNorm_6144(x) · w_in                                  # two roundings (GlmMoeDsaRMSNorm == Glm5's)
qr  = RMSNorm_2048(W_qa h) · w_qa                             # q_a_layernorm on the q latent
q_h = W_qb qr  → per head [q_nope 192 | q_rot 64]
c   = W_kva h  → [kv 512 | k_rot 64]                          # kv_a_proj_with_mqa, k_rot is NOT normed
lat = RMSNorm_512(kv) · w_kva                                 # kv_a_layernorm on the 512 only
q_rot, k_rot ← interleaved RoPE at position t (pairs (2i, 2i+1), inv_freq 8e6^(-2i/64), bf16 cos/sin)
sel = indexer(h, qr, t)            on "full" layers        # §1.3
    = selection carried from the last "full" layer         on "shared" layers
k_h = [W_uk,h lat | k_rot] ; v_h = W_uv,h lat                 # kv_b_proj rows per head: [192 nope | 256 v]
o_h = softmax_{j ∈ sel, j ≤ t}((q_nope,h · k_nope,h,j + q_rot,h · k_rot,j) / 16) · v_h,j   # 1/sqrt(256), fp32 softmax
x   = x + W_o [o_1 … o_64]                                     # bf16 add, plain residual
h   = RMSNorm_6144(x) · w_post
x   = x + MLP(h) | MoE(h)
```

RoPE detail: `cos`/`sin` are computed in fp32 from `inv_freq[i] = 1 /
8e6^(2i/64)`, i < 32, cast to bf16; pair i of the 64 rope dims is `(x[2i],
x[2i+1])` and rotates as `x[2i]·c − x[2i+1]·s`, `x[2i+1]·c + x[2i]·s`, each
product rounded to bf16 and the sum rounded once more — the QSA kernel's
three-rounding policy with the interleaved pairing (the GLM-4.7 lesson in
reverse: that family is half-split, this one is interleaved; both must be
read from transformers' code, not assumed).

The attention scale is `qk_head_dim^-0.5` = 1/16. The engine's `DsaLayer`
computes `attn_scale_ = 1/sqrt(qk_nope_head_dim)` (`dsa_layer.cu:219-222`),
which is only right when rope is 0; the family passes the sum.

MoE (`GlmMoeDsaTopkRouter`): fp32 logits, sigmoid, `+bias` for the choice
only, the uncorrected scores gathered, normalised (`+1e-20`), × 2.5, ties to
the lower id — the router the engine already runs for both GLM families.
Experts: `down(silu(gate) · up)`, no clamp (`swiglu_limit` absent; GLM-4.7's
no-clamp path). The reference accumulates each expert's contribution into a
bf16 buffer over ascending ids and adds the shared expert in bf16; the
engine's fp32 chain with one rounding is within the numerics budget as for
the other two families.

### 1.3 The indexer and cross-layer sharing (`GlmMoeDsaIndexer`)

```
q = W_wqb qr                → [32 heads, 128] = [q_rot 64 | q_pass 64]      # rope dims FIRST
k = LayerNorm_128(W_wk h)   → [128]           = [k_rot 64 | k_pass 64]      # LayerNorm with bias, eps 1e-6
q_rot, k_rot ← interleaved RoPE at t
s_h,j = relu((q_h · k_j) / sqrt(128))                                        # fp32
w     = fp32(W_wp h) / sqrt(32)                                              # weights_proj kept in fp32
score_j = Σ_h w_h s_h,j  over j ≤ t ; sel = top-2048 of score (all j when t < 2048)
```

Note the layout asymmetry: MLA heads are `[nope | rope]`, indexer heads are
`[rope | pass]`. The key is cached after norm and RoPE (a key's rotation
depends only on its own position); the query is rotated at selection time.

Sharing: transformers passes each layer's `topk_indices` to the next; a
`shared` layer takes `prev_topk_indices` verbatim and raises without one.
With `indexer_types` as above, layers 3–5 attend with layer 2's selection,
7–9 with 6's, …, 75–77 with 74's. So the selection is computed 21 times per
token, not 78, and only 21 layers (3 dense + 18 MoE) own an index cache. The engine's `DsaLayer`
runs the indexer unconditionally and keeps `topk_`/`counts_` in scratch
valid only until the next enqueue on any layer (`dsa_layer.hpp:22-27,
303-304`); the family's walk has to enqueue shared layers select-less and
guarantee nothing overwrites the selection between a full layer and its
three dependants.

Selection scope: DESIGN §7.2's pinned semantics apply with `kpool` = 1 —
visible entries `p + 1`, `select_k` = 2048 (a power of two, as the bitonic
networks require), pool id = token position (`kIdxBits` 21 bounds the
context at 2,097,151 tokens, above the model's 1,048,576), exact ties to the
lower position, ascending output, the dense regime for positions ≤ 2047
(one entry per token: `1 × (2048 + 1) − 2`), so a 2048-token chunk is dense
exactly as under Flash. The tail ring degenerates: no incomplete tail ever
exists.

Hadamard-128 and the fp8 index cache are this engine's device, not the
reference's (transformers scores bf16 keys in fp32); they are dot-product
preserving and the selection audit already certifies flips as measured
boundary near-ties, the same stance as under Flash. RoPE must be applied
before the transform on both sides.

### 1.4 The draft layer (`model.layers.78`, in `passthrough.safetensors`)

```
x_q = eh_proj([enorm(embed(tok_{q+1})) | hnorm(h_q)])      # eh_proj BF16 [6144, 12288]
→ one decoder layer with its OWN indexer and caches → shared_head.norm → lm_head (shared)
```

`h_q` convention: vLLM's DeepSeek-V3 MTP (which `GlmMoeDsaForCausalLM` is a
subclass of there) feeds the target model's returned hidden state, i.e.
**post-final-norm**; GLM-4.7 confirmed that convention by acceptance (79–98 %
post-norm vs 11–46 % pre-norm), while GLM-5.3-Flash's draft takes the
pre-norm stream mean. Default post-norm, pre-norm one flag away, acceptance
rate on the five-class corpus is the arbiter (§5, G5). `shared_head.norm`,
`enorm`, `hnorm` are the two-rounding norm.

`index_share_for_mtp_iteration` (vLLM `set_skip_topk`): the draft layer
computes its own selection on draft step 0 and reuses it on steps 1+. At
depth 1 there is no step 1; it matters only when depth 2 is built for this
family. The draft's experts, shared expert, attention and indexer are BF16
in the file (19.9 GB); D5 requantizes the experts and shared expert at load.

### 1.5 Tensor census and the weight-format contract

Shards: `layer-NNN.safetensors` for N = 0..77 (one layer each; MoE layers
5.35 GB, dense layers 0.80 GB), `passthrough.safetensors` (23.7 GB: embed,
final norm, `lm_head`, the whole draft layer). Verified from the headers on
disk:

| class | tensors | dtype / shape |
|---|---|---|
| norms | `input_layernorm`, `post_attention_layernorm` [6144]; `q_a_layernorm` [2048]; `kv_a_layernorm` [512] | BF16 |
| attention, layers 0–2 | `q_a_proj.weight` [2048, 6144], `q_b_proj` [16384, 2048], `kv_a_proj_with_mqa` [576, 6144], `kv_b_proj` [28672, 512], `o_proj` [6144, 16384] | BF16, verbatim |
| attention, layers 3–77 | the same five as triples: `weight_packed` I32 [N, K/4], `weight_scale` BF16 [N, K/64], `weight_shape` I64 [2] = [N, K] | int8 g64 |
| indexer (layers 0, 1, 2, 6, 10, …, 74 — 21 layers — and 78) | `wq_b.weight` [4096, 2048], `wk.weight` [128, 6144], `weights_proj.weight` [32, 6144], `k_norm.weight`/`.bias` [128] | BF16 |
| dense MLP (0–2) | `gate_proj`/`up_proj` [12288, 6144], `down_proj` [6144, 12288] | BF16 |
| router (3–77) | `gate.weight` BF16 [256, 6144], `gate.e_score_correction_bias` F32 [256] | |
| routed experts (3–77) | per expert `gate_proj`/`up_proj`: packed I32 [2048, 768], scale [2048, 96]; `down_proj`: packed [6144, 256], scale [6144, 32]; each with `weight_shape` | int4 g64 |
| shared expert (3–77) | `gate_proj`/`up_proj` packed [2048, 1536] + scale [2048, 96]; `down_proj` packed [6144, 512] + scale [6144, 32] | int8 g64 |
| globals | `model.embed_tokens.weight`, `lm_head.weight` [154880, 6144], `model.norm.weight` | BF16 |
| draft (78) | no `embed_tokens` duplicate (the shared embedding feeds it); `enorm`/`hnorm`/`shared_head.norm` [6144], `eh_proj.weight` [6144, 12288], the attention set as `.weight`, the indexer set, `mlp.gate.*`, 256 × 3 BF16 expert matrices [2048, 6144] / [6144, 2048], `shared_experts.*` | BF16 |

Packing contract (compressed-tensors 0.18 `pack-quantized`, `symmetric`,
`strategy group`, `group_size` 64, `observer memoryless_minmax`, no
`zp_dtype`): codes are stored **unsigned with an offset of 2^(bits−1)**
(int4: nibble − 8 ∈ [−8, 7]; int8: byte − 128 ∈ [−128, 127]); element j of a
row lives in int32 word `j / (32/bits)` at bit position `bits × (j mod
(32/bits))`, so the **low nibble / low byte is the first element**; the scale
for element j is `weight_scale[n, j / 64]`; the dequantized weight is `code ×
scale`. The nibble histogram of a real expert matrix is centred on 8 (the
signed-packing alternative would centre on 0/15), scales are ~1e-4 (the
fixture rule from GLM-4.7: release-like magnitudes, or the TP folds' bf16
rounding shows as hard disagreements). The quantizer's scale was
`max|w| / 7.5` (int4) and `/ 127.5` (int8), which the loader never needs.
`weight_shape` is read and checked, never trusted for allocation.

**Dequant policy (D2, revised 2026-09-12):** packed matrix products use
`code × bf16(scale)` exactly, without rounding individual weights to BF16.
GEMV factors the scale outside each integer-code partial dot; GEMM feeds
exact integer codes to BF16 tensor cores and scales each 64-element group
in FP32. Their summation orders differ, so they are tolerance-equal.
The independent oracle accumulates exact weights in FP64. The loader's
BF16 `kv_b` bridge is an explicit exception: it rounds `code × scale`
before the absorb/value-out kernels consume it. Bridge comparisons use
power-of-two scales when they require exact weight identity. This replaces
the original per-weight BF16-rounding proposal.

### 1.6 Tokenizer, template, generation defaults

`tokenizer.json` and `tokenizer_config.json` are byte-identical to the
Flash checkpoint's (sha256 19e77364…, 98b12715…): the GLM split regex, no
NFC, the `Sequence[ByteLevel]` post-processor; the GLM tokenizer goldens
apply as they are. `chat_template.jinja` differs (255 vs 210 lines): the
`clear_thinking` default flips to **false** (prior turns' `reasoning_content`
stays in the prompt unless the caller clears it — a prompt-length and
prefix-cache behaviour change worth a golden), `elif content is not none`,
and duplicate/unknown tool-call-id guards (`has_dup_tool_result_id`,
`tc_id_exists`, `can_sort` breaks). The interpreter already has `macro`,
`namespace`, `range`, `length` and `break`. Tool calls keep the GLM markers
(`ToolFormat::kGlmMarkers`). Generation defaults: temperature 1.0, top_p
0.95 (Flash: none).

## 2. Placement and cost by world size

TP only (D9); the per-rank figures are for W = 4. Placement follows the
engine's DSA rules (`tp.hpp:5-16`): q heads and `kv_b` head blocks sharded,
`o_proj` column-packed (one fold), `q_a`/`kv_a`/indexer/router/norms
replicated, experts sliced on the intermediate dim (gate/up rows, down
columns), `lm_head` vocab-sharded, embedding replicated, **latent and index
caches replicated on every rank** (every rank selects the same tokens).

### 2.1 Resident bytes per rank

Size model validated against the index's 427.4 GB total (matches to 0.1 %).
`docs/checkpoint_budget_glm53.md` (G0, `tools/checkpoint_audit.py` on the
headers, 2026-09-12) evaluates the same placement per tensor: **99.30 GiB at
W = 4** (97.84 without the draft), the number the loader's memory plan
measured; the only difference from the table's first draft is `kv_b`, held
BF16 at load (the absorbed-attention bridge), +0.4 GiB:

| class | per rank at W = 4 |
|---|---|
| routed experts, int4 g64 (0.53125 B/weight), 75 layers | 90.0 GiB |
| shared experts int8 + attention int8 (sharded part, `kv_b` BF16 at load) | 3.8 GiB |
| replicated per MoE layer: `q_a` + `kv_a` int8 (16.6 MB), router, norms | 1.4 GiB |
| indexers, 3 dense + 18 MoE layers, BF16 replicated | 0.37 GiB |
| dense layers 0–2 (BF16 attention + MLP) | 0.62 GiB |
| embedding (replicated) + `lm_head` slice + final norm | 2.2 GiB |
| draft layer 78: experts int4 + shared int8 at load (D5), attention/eh_proj/indexer BF16 | 1.35 GiB (4.65 with BF16 experts) |
| **weights** | **99.3 GiB** (97.8 without the draft; 102.4 with BF16 draft experts) |

Against the two-node Flash experience (94.71 GiB of weights left 6.5 GiB of
caches, a 2 GiB arena, scratch and the plan's 8 GiB headroom on a 119.5 GiB
ceiling), this leaves **2 to 4 GiB per rank for the caches**. Two levers if
the plan refuses: vocab-shard the embedding (−1.4 GiB, one more collective
per step) and run plain (−1.4 GiB and no draft cache).

### 2.2 Decode traffic and the floor

Per token per rank at W = 4:

| term | GB |
|---|---|
| routed experts (8 × 3 slices × 75 layers, int4) | 3.01 |
| attention sharded (`q_b`, `kv_b`, `o_proj` int8, 75 layers) | 2.88 |
| `q_a` + `kv_a` replicated (int8, 75 layers) | 1.25 |
| shared experts (int8) | 0.73 |
| dense layers 0–2 (BF16) | 0.66 |
| `lm_head` slice | 0.48 |
| indexers (BF16, 21 layers) | 0.39 |
| routers | 0.24 |
| draft layer (D5) | 0.33 |
| **total** | **≈ 9.95 GB → 41.5 ms at 240 GB/s** (the audit, with `kv_b` BF16: 10.23 GB, 44.5 ms at 230 GB/s) |

For scale: GLM-5.3-Flash hybrid reads 5.9 GB and measures 26 ms; GLM-4.7
reads 9.7 GB and measures 49 ms. Collectives: two folds per layer
(`o_proj` partials, MoE partials) plus the draft and the head ≈ 158 recorded
nodes per step, ~8 ms at the measured 47–60 µs per node. The honest
expectation is T = 1 near 50 ms per step and MTP near 60 ms per pass at
~1.8 tokens per pass (≈ 33 ms/token), to be replaced by measurements. The
levers after the port, in order of bytes: `q_a`/`kv_a` (1.25 GB, replicated
because `q_a_layernorm` needs the full latent — row-sharding costs an
all-gather per layer, to be measured, not assumed), `kv_b` and the indexers
in a narrower format (quality decisions), the dense layers in FP8 at load
(the Qwen `dense_weights` precedent, a quality decision), one fused
`q_a`+`kv_a` launch (bytes unchanged, launches fewer).

### 2.3 Context budget

Per context token per rank (78 latent rows of 512 in the cache format plus
the 64-wide rope key in bf16, D3; 21 index caches of 128 fp8 + 4; the draft
hidden cache at 12 KiB per token per slot):

| latent format | 1 slot | 2 slots | 4 slots | plain (no draft) |
|---|---|---|---|---|
| bf16 (1152 B/row) | 102.8 KiB | 114.8 | 138.8 | 90.8 |
| fp8 (644 B/row) | 64.1 | 76.1 | 100.1 | 52.1 |
| fp4 (420 B/row) | 47.1 | 59.1 | 83.1 | 35.1 |

With 3 GiB of cache: about 31k tokens at four slots fp8, 41k at two, 65k at
one slot fp4, 60k plain with fp8 and 90k plain with fp4. Compare Flash at world 4: 490k. The
structural limiter is the replicated MLA latent; a context-parallel cache
(positions sharded, a distributed top-k) is a large engine change and is
out of scope here — the plan ships the honest ceiling and the memory plan's
number, read from a boot refusal, not from the standalone run.

*Measured (2026-09-12):* the plan's pool is flat in the slot count (the draft
takes one hidden row, not a per-token cache — this table's per-slot column
overstated it) at 91.7 KiB per token bf16, 52.5 fp8, 35.3 fp4, and the
prefix arena costs 0.05 GiB whatever `prefix_cache_gib` says. On the 121.6
GiB nodes the ceilings under the 5 GiB headroom (the 8 GiB one revisited
the same night, `docs/measurements.md`) are 96K bf16 with MTP, 112K bf16
plain, 160K fp8 with MTP; fp4 would reach ~300K. Beyond that the levers are
the replicated weights (a vocab-sharded embedding −1.3 GiB, `kv_b` as int8
−0.3) and then context parallelism.

## 3. Design decisions

**D1 — A new family on the shared cores, named `glm_dsa`.** `src/models/glm_dsa/`
with `GlmDsaModel : SessionModel<GlmDsaModel>` and `GlmDsaLayerStream :
ResidentLayerStream<GlmDsaLoaderFamily>`, exactly as `Glm4Model` /
`Glm4LayerStream` were built; `ModelArchitecture::GlmMoeDsa` for
`GlmMoeDsaForCausalLM` (and the `glm_moe_dsa` fallback, which today wrongly
lands on the Flash family); serve family name `glm_moe_dsa`. The layer walk
is the Flash draft block's body (`mtp.cpp:88-122`: norm → DSA → fold →
residual add → norm → MoE/dense → fold → residual add) 78 times, then the
final norm, the head, and the draft. `DsaLayer`, `GlmMoeLayer`, the dense
MLP, the norms and the MTP input kernel are reused as objects; what the
walk adds is the shared-selection cadence (D4) and the rope plumbing (D3).
GLM-5.3-Flash's `GlmDiagnosticModel` is not touched.

**D2 — One packed-int core, templated on bits ∈ {4, 8}, with the exact
dequant.** `src/kernels/packq_gemv.cuh` (`namespace packq_gemv`) on the
shared row geometry (`gemv_common.cuh`): a lane's 16-byte chunk holds 32
int4 or 16 int8 codes (half or a quarter of a 64-group, so the group scale
is loaded once per two or four chunks), in-register unpack (`(word >> 4i)
& 0xF` − 8, or bytes − 128), an integer-code partial dot per chunk (`part
+= code × x`, an fp32 FMA chain over exact integers) then `acc += scale ×
part` — the weight is `code × bf16(scale)` exactly, never rounded to bf16
(revised 2026-09-12 from the bf16-rounded form: one scale multiply per
chunk instead of one per element, and the future tile kernel can feed the
integer codes to the tensor cores with the same per-group scaling and stay
bitwise per segment), the fixed xor-shuffle tree — the same chain whatever
the row count or launcher, so the slot path, the grouped path and the
sliced fold are bitwise twins (the pins `fp4_gemv_test` and `glm_moe_test`
carry for NVFP4). The bf16 bridge (`kv_b`, D6) is `bf16(code × scale)`, a
rounding away from the core; the tests that compare the two forms use
power-of-two scales, where they coincide. Compiled K set:
6144 (`q_a`, `kv_a`, gate/up at every world), 2048 (`q_b`), 512 (`kv_b`,
when D6's lever lands), 16384 / 8192 / 4096 (`o_proj` at W 1/2/4), 2048 / 1024 / 512 (expert and
shared down at W 1/2/4). A matrix struct `GlmPackedMatrix {packed, scales,
rows, cols, bits}` in `quant_matrix.hpp`; `MoeExpertView` gets the packed
interpretation and the slot kernels a format enumerator in place of the
`bool kSharedFp4` (routed int4 with an int8 shared expert is the production
shape). Row slices are free; **column slices start on a 64-element group
boundary** (which also satisfies the 8-per-word packing), true for every
production slice (`o_proj` K/W ≥ 4096, down K/W ≥ 512).

**D3 — The rope key rides the latent row as a bf16 tail.** The cache row
becomes `[512 latent in the cache format | 64 bf16 rope key]` (bf16 1152 B,
fp8 640, fp4 416, the fp8/fp4 row scale staying in the planar array beside
the rows — so the rope tail never shares a row scale with the latent),
`DsaGeometry` gains `rope_dim` and `score_width` = `kv_lora + rope`, and
the absorbed query becomes `[W_uk,hᵀ q_nope,h | q_rot,h]` (576 wide). The
score dot runs over 576 and the value accumulation over the first 512, in
all three attention kernels (the flash kernels are templated on the score
width beside the value width — `Geo<512, 576>`, 91 KB of shared memory,
one block per SM; the split kernel gives every thread a slice of the tail
beside its value window), the append kernel writes the tail unquantized,
and the absorb kernel emits the concatenation. Flash's geometry (rope 0)
is the same code with a zero-width tail, which is how the existing bitwise
gates stay green (verified 2026-09-12: `dsa_test` 33/33 unchanged after
the change). `attn_scale_` = 1/sqrt(nope + rope). RoPE for MLA q/k and for
the indexer q/k is one small kernel (`dsa_rope_interleave`: interleaved
pairs kept in place, three bf16 roundings, contraction-proof) over a bf16
cos/sin table the model builds once on the host (`dsa_rope_table_host`:
fp32 inv_freq and position product as transformers, the trig in double
then rounded to fp32 and bf16 — host, device and the python reference read
one table, so the rotation is bitwise across them), applied to `q_rot`
before absorb, to `k_rot` at append, to the indexer key after its
LayerNorm and to the indexer query after `wq_b`, both before the
Hadamard/quantization.

**D4 — Select once, attend four times.** The family's walk enqueues a full
layer as today and a shared layer with `reuse_selection = true`, which
skips `project_common`'s indexer half and the select and reads `topk_` /
`counts_` left by the last full layer; the pool keeps latents for 78 layers
and index caches for the 21 that own an indexer (`DsaStatePool` gets a
latent layer count and an index-cache ordinal map). `DsaLayer` asserts that
nothing between a full layer and its dependants touches the selection
scratch (the prefill dot buffer and q/k scratch are separate regions). The
graph recorder sees 21 selects per step. Snapshots are unaffected: the
selection is recomputed every step and is not state.

**D5 — The draft layer's experts and shared expert are requantized at load
with the checkpoint's own recipe** (RTN, symmetric, group 64: int4 for the
routed experts, int8 for the shared expert; a host encoder
`loaders/packq_quant.hpp` pinned by a unit test that round-trips the
checkpoint's scale rule). The draft only proposes; its numerics never reach
a transcript (GLM-4.7's D1). This is what turns 4.65 GiB per rank into 1.35
and 1.2 GB of BF16 expert traffic per pass into 0.3. The draft's attention,
`eh_proj` and indexer stay BF16 (0.3 GiB).

**D6 — `kv_b` and the indexer projections are consumed in BF16.** `kv_b`
(int8 in the file) is dequantized at load with the D2 rounding — the
reference's own weights — because the absorb and value-out kernels are bf16
kernels; +0.26 GiB per rank and +0.28 GB per step, listed as a lever. The
indexer, router, norms and dense layers are BF16 in the file and stay so.

**D7 — Packed tensor-core prefill for attention and experts.**
`packq_gemm.cu` consumes the int4/int8 representation directly. A 32 × 64
output tile stages two packed 64-deep groups, decodes exact integer codes
into BF16 fragments, and applies each group's BF16 scale to its FP32
partial. Dense attention, routed experts and the int8 shared expert share
this arithmetic; no dequantized weight bridge is allocated. Grouped gate/up
reads hidden rows through the device row map, with FP32 down outputs
feeding the existing ordered accumulation. Automatic dispatch starts at
128 prompt rows. Short prompts and every current decode/verification
family retain GEMV arithmetic. Dense/grouped GEMM outputs are bitwise
equal for the same inputs; GEMM/GEMV use the numerical gates in
[numerics](numerics.md).

**D8 — `kpool` = 1 is a real template instance, not a special case.**
`index_kpool ∈ {1, 2, 4, 8}` in the geometry check and the kernel dispatch;
`select_k` 2048 in the bitonic networks (8 KB of keys; the merge and the
running top-k at 2048 need their own smem budget check on the 99 KB block
limit); the tail ring is allocated at one slot for the snapshot machinery
and never scored; the prefill select's materialized dots are four times
Flash's per context length and are bounded by the existing dot budget
(smaller query tiles, more K re-reads — a measured prefill item, not a
design change). `index_kpool_compress` and `always_select_tail` are
irrelevant at kpool 1 and the parser requires them absent.

**D9 — TP only, text only, decode rows ≤ 16.** No expert parallelism (the
Qwen plan's argument), no vision (the checkpoint has none). The decode batch
was capped at 8 rows by the fused decode select's shared memory
(`kSelectMaxRows` = 8: four request slots at depth 1, eight plain, two at
depth 3); since 2026-09-13 the family's `decode_rows_cap` is 16 — the
select launches its rows in groups of eight (a group is the same work at
any grouping: every block scores a stripe of every row's own context), the
pick kernels verdict sixteen request slots, the DSA layer's attention
tiles and (row, split) workspace scale with the count, and the absorb/vout
projections keep their warp kernels for every decode shape (the
tensor-core forms are tolerance-equal, not bitwise). Eight slots at depth
1, five at depth 2, four at depth 3; the bus's 32 graph variants bound a
plain deployment at twelve slots. The graph engine records 4- and 6-slot
batch families beside the 2-, 3- and every-slot ones for recipes wider
than four slots, so four live requests on an eight-slot world ride an
eight-row batch (41–42 tok/s aggregate, the four-slot template's own)
rather than the sixteen-row one (26–28). Every recipe up to eight rows runs the
launches it always did (T=1 51.3 ms/step and depth 1 66.9 ms/pass
unchanged on the fabric). One consequence of the GEMM interface's decode
lowering: a sixteen-row deployment prefills prompts of 9–16 tokens through
the GEMV chain where an eight-row one used cuBLASLt — last-bit
differences, deterministic within a deployment.

**D9b — the 32-row decode batch: the multi-request block (planned 2026-10-11).**
The DFlash2 lane's verify is 8 rows a slot, so the sixteen-row batch holds ONE
other request's block: a 2-slot family replays, C4 takes two passes and C8 four
(the recorded C2–C8 flatness). Doubling the decode batch to 32 rows puts FOUR
8-row blocks in one pass and an eight-slot capped family at 4 rows each under
the schedule. The seams, audited 2026-10-11:

- `GlmDsaModel::decode_rows_cap()` 16 → 32, and every `kDecodeRows`-shaped
  default that keys off it (the diagnostic wrapper's shape helpers, the serve
  app's `max(kDecodeRows, max_concurrency × rows_per_request)`).
- The fused select already launches rows in groups of `kSelectMaxRows` = 8
  (dsa.cu) — four groups is the same work per group; no kernel change.
- The pick kernels verdict `kPickMaxRequests` = 16 request slots — eight
  serving slots stay inside; only >16 concurrent slots would need it wider.
- The DSA attention tiles and (row, split) workspace scale with the count
  (the memory plan must re-fit; rank 0 sits at ~2.9 GiB free with the 1 GiB
  prefix cache — trim `prefix_cache_gib` before shrinking the pool).
- The MoE slot kernel takes the rows (the grouped chain stays off: the
  2026-10-09 A/B lost at 8 rows, 160 vs 148 ms; re-measure at 32 only if the
  slot kernel's expert re-streams dominate).
- Dispatch seam: `dense_gemv_rows()` defaults to 4 (no env override on this
  lane), so decode GEMMs already mix GEMV and cuBLASLt per row count — the
  batch-vs-scalar gate is the near-tie rule, not bitwise (D9's precedent at
  9–16-token prefills). No row-invariance campaign needed.
- The bus's graph-variant budget (`kBusMaxGraphVariants` = 128; variants per
  depth = 2 × (slots + families)) binds first: 32 rows add the 3- and 4-slot
  families beside 2 and every-slot, and the schedule's depth options multiply.
  Budget the options explicitly (drop the scalar-slot options the recipe never
  replays, or trim depth options for the wide families) before the first boot.
- The drafter side already parameterizes: `dflash2_draft_batch` stacks
  `dflash2_batch_slots()` blocks a forward and the recorded batch draft takes
  the family's slots; the planes and feature path scale per slot.

Implementation (2026-10-11, the same day): `decode_rows_cap()` 32, the DSA
layer's `[1, 16]` validation to `[1, 32]` (the select workspace already
bounded rows at 32; the groups-of-eight select and the scaled scratch needed
nothing), and one real discovery — the first 3-slot capture died on 24 memset
graph nodes: cuBLASLt's heuristic hands the new 24-row GEMM shapes a split-K
algorithm whose workspace zeroing is a memset inside the capture, and decode
graphs must stay kernels-only. The plan builder now prefers workspace-free
algorithms whenever it runs on a capturing stream (graph_check logs any
memset node's pointer and width for the next triage). The boot then captured
the whole family tree: 2/3/4-slot x 8-row blocks with every scheduled-depth
variant, kernels-only.

Gates: the unit suites (glm_moe_test, glm_dsa_dflash2_test, the DSA forward
fixture) at 32-row shapes; the fabric lane's transcripts under the 4-slot
family pass the near-tie rule against the scalar path; the C1 walls are
unchanged (the 2-slot family's replays keep their shape); the C2/C4/C8 suite
(the 2026-10-11 baseline in
benchmarks/results/2026-10-11-glm53-dflash2-depth/) shows the aggregate lift —
the exit gate is C4 ≥ the C2 aggregate (a second block pays for its rows) and
no C1 regression.

**D10 — Tokenizer, template, tools.** The Flash tokenizer goldens are
registered against this checkpoint unchanged (identical bytes; the suite is
hash-keyed and skips if the cached file differs); a fourth chat-template
golden set is generated for the new template, including the `clear_thinking`
default, the `content is not none` branch, the duplicate-id guards, and the
tool-call round trips and grammars. `ToolFormat::kGlmMarkers` applies.

## 4. Layout

| path | contents |
|---|---|
| `src/models/glm_dsa/config.{hpp,cpp}` | `GlmDsaTextConfig` (the flat config, the indexer schedule from `indexer_types` cross-checked against freq/offset, `quantization_config` with the two `pack-quantized` groups: targets, `num_bits` 4/8, `group_size` 64, `symmetric`, the `ignore` list), `ModelArchitecture::GlmMoeDsa` in `loaders/architecture` |
| `src/models/glm_dsa/binding.{hpp,cpp}` | the expected-tensor table (dense layers verbatim, MoE layers as triples, indexer-owning layers, the draft, globals), roles `IntPacked`/`IntScale`/`IntShape`, the validator, the TP geometry check |
| `src/kernels/packq_gemv.{cuh,cu,hpp}`, `src/kernels/packq_gemm.{cu,hpp}` | D2, D7 |
| `src/models/quant_matrix.hpp` | `GlmPackedMatrix` |
| `src/kernels/glm_moe.cu`, `glm_moe_launch.hpp`, `models/glm/moe.hpp`, `moe_layer.cpp` | the packed slot/grouped/tile forms beside the fp8 and fp4 ones; the shared-expert format enumerator |
| `src/loaders/packq_quant.hpp` | the host RTN encoder (D5) |
| `src/models/glm_dsa/loader.{hpp,cpp}` | `GlmDsaLayerStream` on `ResidentLayerStream`: `load_packq_rows/cols` in `weight_build.hpp`, the draft requant, `kv_b` dequant, byte formulas, `loader_format()` |
| `src/models/dsa_geometry.hpp`, `dsa_layer.{hpp,cu}`, `dsa_state.{hpp,cpp}`, `src/kernels/dsa.{cu,hpp}` | D3 (rope tail, RoPE kernel, scale), D4 (`reuse_selection`, layer/index-cache ordinals), D8 (kpool 1, select_k 2048) — Flash's geometry must stay bitwise |
| `src/models/glm_dsa/{layers,forward}.{hpp,cpp}` | the plain pre-norm walk, `GlmDsaModel` (the `SessionModel` hooks, the draft on post-norm hidden, the memory plan) |
| `src/models/dsa_reference.*`, `tools/dsa_reference_dump.py`, `tools/glm_dsa_reference_dump.py` | the oracle extended for rope/kpool 1/sharing; the pure-python full-model reference on the fixture |
| `tools/glm_dsa_torch_reference.py` | transformers' own layer code on the real weights (the GLM-4.7 rule) |
| `tests/unit/glm_dsa_{config,binding}_test.cpp`, `tests/unit/packq_quant_test.cpp`, `tests/cuda/packq_gemv_test.cu` + `packq_gemv_checkpoint.cpp`, `tests/cuda/glm_dsa_*` | the gates (§5) |
| `apps/glm_dsa_load_check.cpp`, `apps/glm_dsa_forward_check.cpp`, `apps/dgpp_serve.cpp` (`GlmDsaFamily`) | the apps; the forward check carries `--teacher-file` for `fabric_logprob.py --prefill` |
| `deploy/cluster_glm-5.3_int4-int8_w4.example.json` (since 2026-09-14 one template: eight slots, MTP depth 1; the plain and fp8-cache shapes are knobs) | the deployments (model name `glm-5.3`, quant `int4-int8`) |
| `scripts/fabric_glm_dsa_serve.sh`, `scripts/fabric_glm_dsa_load.sh`, `scripts/fabric_glm_dsa_forward.sh` | the fabric procedures (the GLM-4.7 scripts' shape) |
| `tests/data/glm_dsa_chat_template_goldens.jsonl` | D10 |
| `docs/model_cards/GLM-5.3-Int4-Int8Mix-RTN-g64.md`, `docs/checkpoint_budget_glm53.md` | the card and the audit's budget |

## 5. Implementation stages and acceptance gates

Order: G0 → G1 → G2 → G3 → G4 → G5 → G6; nothing merges without its gate,
and every stage ends with the full build (`cmake --build build-ci -j --
-k`, never overlapped with another build) and an unfiltered `ctest` from
`build-ci` — the existing 62 tests are a gate of every stage because the
port touches shared code (the DSA layer, the MoE layer, the loader builder,
the architecture detector).

**G0 — the checkpoint on the fabric.** Rank 0's download completes; the
peers get the snapshot (the running `hf-download.sh … --copy-parallel`, or
`scripts/download_model.py --sync-only --config …`); `--verify-only` passes on all four nodes; each node
has ≥ 430 GB free before the copy and ≥ 100 GiB more for its resident image.
`tools/checkpoint_audit.py` learns the packed triple and emits
`docs/checkpoint_budget_glm53.md` (the authoritative byte breakdown; the
size model in §2.1 must agree with it to 1 %).

**G1 — config + binding.** `glm_dsa_config_test`: the real `config.json`
parses; every rejection names its field (a `shared` entry where freq/offset
predicts `full`, `qk_rope_head_dim` ≠ 64 with a 64-wide `kv_a` row,
`group_size` ≠ 64, a zero-point dtype, a target regex that does not cover
layers 3–77 exactly, `tie_word_embeddings`, `n_group` ≠ 1, a present
`index_kpool`); the indexer schedule reproduces `[0, 1, 2, 6, 10, …, 74]`.
`glm_dsa_binding_test` + `glm_dsa_load_check --bind`: the table reproduces
the real checkpoint's tensor set exactly (175,985 names, dtypes and shapes;
`unexpected` = 0; `weight_shape` == the declared [N, K] on every triple;
the 21 + 1 indexer-owning layers and no others), headers only. The
architecture detector selects the family; the Flash checkpoint still
selects Flash (`glm_config_test` untouched).

**G2 — the packed-int cores.** `packq_quant_test`: the encoder reproduces
the checkpoint's scale rule and packing on synthetic rows and round-trips
within the format's error. `packq_gemv_test` (the `fp4_gemv_test` shape):
bits 4 and 8 at every compiled K against the host oracle (`bf16(code ×
scale)`, fp64 accumulation, the dot rounded to bf16; 2 bf16 ulps with the
1e-3 cancellation floor, zero mismatches), rows independent of the row
count (bitwise), the f32 epilogue the unrounded bf16, NaN scales propagated,
geometry outside the contract rejected. `packq_gemv_checkpoint.cpp` (label
`checkpoint`): one routed expert triple and one attention matrix mmapped
from the real shard, single rows bitwise the batched rows, and the decoded
weights bitwise a reference decode of the same bytes (the packing contract
of §1.5, pinned on real data — the one test that catches a nibble-order or
offset mistake). `glm_moe_test` gains the `_packq` twin of every fp8/nvfp4
gate: slot == grouped == sliced fold bitwise, the int8 shared expert in the
slot kernels on every path, the tile kernel bitwise its dense form per
segment and within the mma-order budget of the GEMV chain, near-tie
certification of route flips. `compute-sanitizer` memcheck/racecheck on the
new kernels' shapes.

**G3 — loader.** `glm_dsa_loader_test`: fixture build == resident-image
restore byte for byte at worlds 1, 2, 4; byte formula == bump usage ==
source plan on every layer class (dense, MoE, indexer-owning, draft,
globals); packed column slices refused off a 64-boundary by name; the draft
requant round-trips within the format's error and its bytes are in the
image; `kv_b`'s dequant equals the oracle's. `glm_dsa_load_check --world 4
--rank 0 --streaming --mtp` on the real checkpoint: every layer placed, the
resident formula reconciled against the bump, the per-rank total recorded
(the §2.1 estimate is replaced here), the first image built and restored.
Real loads run in the foreground with the `drop_caches` loop beside them.

**G4 — the DSA changes, gated on Flash first.** Every existing DSA gate
(`dsa_test`, `dsa_dump_parity`, `glm_forward_test`, `glm_tp_test`,
`glm_gen_check` on the fabric) stays bitwise on GLM-5.3-Flash after D3/D4/D8
land — the zero-width rope tail and kpool 4 are the same code paths. Then
at the full geometry: the select fuzz at kpool 1 / select_k 2048 (the
1,100-case bitwise fuzz plus exact ties and the 2048th-boundary
construction), the three attention kernels with the 576-wide row against
the host oracle at TP1/TP4, empty rows and head groups, all three cache
formats (the fp8/fp4 rows dequantized, the tail exact), the RoPE kernel
against a double oracle with the three-rounding policy (interleaved pairs,
θ 8e6, positions across 2^16), the shared-selection walk on a fixture
(shared layers' selection bitwise the full layer's; a layer enqueued between
them does not disturb it; graph replay records 21 selects), the dense
regime boundary at position 2047/2048, and `dsa_reference_dump.py` extended
so the oracle's latent is bitwise, its index cache within one e4m3 ulp and
its top-k exact at the new geometry.

**G5 — model and engine.** A synthetic mini-checkpoint written on disk by
the fixture writer (the F,F,F,S,S,S,F schedule, 8 experts, 64 heads at the
real head dims because the listed kernel needs `local_heads % 16` at W = 4,
a reduced `index_topk` for a small dense regime, release-like scales) and
`tools/glm_dsa_reference_dump.py`'s pure-python full-stack reference over
the same weights: `glm_dsa_forward_test` compares per-layer hidden states
(l2 within the fp8/int budgets the other families hold, ≤ 0.004),
final hidden, top-1, route ids and weights, the draft's rows;
`glm_dsa_decode_test`: prefill == forward bitwise, interleaved slots
bitwise, chunked prefill across the dense/sparse boundary, snapshots at any
position hot == cold, 60 steps vs the re-forward, the speculator's
transcript through the draft; `glm_dsa_tp_test`: loopback worlds 2 and 4
rank-identical and within budget of world 1 on fresh ports (29952–29955 are
free; 29900–29951 and 29970 are taken); `glm_dsa_engine_test`: the graph
engine's scalar, batched and MTP replays == the eager engine's transcripts
over 60 steps, world 2 == world 1. Then the rule that found GLM-4.7's RoPE
bug: `tools/glm_dsa_torch_reference.py` runs transformers' own
`GlmMoeDsaForCausalLM` layer code on the real weights (layers 0–3 plus the
embedding, decompressed by compressed-tensors, on the 5090 box's
`~/.venvs/glm53q312` — transformers 5.14.1 has `glm_moe_dsa`; the 4-layer
truncation is set on the config before `from_pretrained`) against
`glm_dsa_forward_check --dump-states` at 32 and ~2,100 tokens (the second
crosses the top-k horizon), relative l2 per layer at the GLM-4.7 level
(≤ 0.003) and top-k selection flips certified as near ties. The draft's
hidden convention is settled here by acceptance on the fabric (post-norm
expected ≥ 75 %).

**G6 — serving, benchmarks, records.** `dgpp-serve` boots the family from
the deployment templates; the memory plan at the shipped shapes is read
from a boot refusal (the standalone plan reads 0.6 GiB more); tokenizer
goldens (55 cases) and the new template goldens pass; the fabric world of 4
runs `fabric_glm_dsa_serve.sh` (greedy transcripts, client timing,
`serve_api_check.py`, eval with thinking off) and the four op-stream md5s
match at `dgpp-cluster down`. The published numbers follow
`docs/benchmarks.md` §9 exactly and land as new rows per section for
"GLM-5.3 Int4/Int8 RTN g64, world 4": §9.1 `serve_bench.py` (T=1 and MTP
ms/step, ms/token, rank 0's stats lines), §9.2 the five-class corpus
through the service (`serve_load.py --classes`, since `glm_gen_check` is
Flash-only) with `mtp_depth_check.py` + `diff -r` byte-identical between
plain and MTP, §9.3 `serve_load.py --concurrency 1,2,4 [--classes all]
[--temperature 1]` and `--isolation 4`, §9.4 `serve_prefill_probe.py 512
2048 8192 --repeat 3` (and `fabric_prefill_repeat.sh` once the family's
gen app takes ids), §9.5 `serve_eval.py --tasks gsm8k,extract` on the
fabric and HumanEval on the isolated evaluation machine with the
denominators recorded, §9.6 the three determinism checks. Quality has two
more items: the teacher-forced NLL on the three shipped texts through the
gen app's `--teacher-file` (`fabric_logprob.py`; the absolute perplexity is
recorded — there is no reference build of this model — and a rerun of the
same binary must show a delta of exactly 0), and the AWQ companion loaded
unchanged and scored the same way once disk allows, which is the end-to-end
AWQ-vs-RTN decision the model cards defer. Operations: `serve_soak_run.sh
60`, `serve_failure_drill.sh` on one rank, `serve_stop_check.sh`,
`serve_prefix_curve_sweep.sh` at one point. Regression of the other
families on the same build: `fabric_glm_regression.sh` bitwise its
baseline, Qwen and GLM-4.7 transcripts identical to their last records,
`ctest` all green. Records: the rows in `docs/benchmarks.md` §3–§7, a
"GLM-5.3 on four nodes" section in `docs/measurements.md`, a dated entry in
`benchmarks/results/`, the CHANGELOG line, the PLAN.md serving table row and
"Additional model work" link, `deploy/README.md`'s template table, the
model card copied into `docs/model_cards/` with the measured numbers and
the memory plan, and §6 of this document filled in.

**Optimization gate, after G6 measures.** The recorded T=1 step is compared
with the §2.2 floor (10.0 GB per rank per token plus ~158 collectives); the
port is accepted at or under GLM-4.7's ratio to its floor (49 / 40 ≈ 1.25×)
and the nsys kernel breakdown (`fabric_qwen_profile.sh`'s shape) names the
remainder. Prefill is accepted with the tile kernel in place: 2,048 tokens
under 3x the Flash hybrid's 1.3 s is the expectation from the byte ratio,
and the nsys split (expert tiles, attention, select, bus) is recorded
either way. Anything that moves a published number after that reruns its
§9 procedure and dates the row. Levers are taken one at a time with an A/B
at `--kv-capacity 16384` and never by booting an older binary at the
production `kv_capacity`.

Rules that apply throughout (the project's, restated because each has cost
a day): rebuild everything before any `ctest` verdict; one fabric ritual at
a time, long ones in the background with a Monitor; never `pkill -f` a
pattern the command itself contains; fixture scales release-like; the
python reference mirrors the kernels' tile chains where a tolerance would
otherwise hide a real gap; a passing suite with skipped checkpoint cases
validates nothing; and when a graph/feed contract changes, the family's gen
app is a second client of it (the 0 %-acceptance lesson).

## 6. Status

2026-09-12. G0 complete: the rank-0 download (427.47 GB, 175,985 tensors,
78 per-layer shards + passthrough), the peer copies rsynced (399 GB on each
of the three peers, no partial blobs) and `scripts/download_model.py
--verify-only --config deploy/cluster_glm-5.3_int4-int8_w4.example.json`
green on all four nodes (revision `147684fb`, 398.1 GiB of indexed weights);
`tools/checkpoint_audit.py` learned the pack-quantized triple (every one of
the 58,200 checked against the config's quantization groups) and wrote
`docs/checkpoint_budget_glm53.md`: 99.30 GiB per rank at W = 4, the memory
plan's number, and a 10.23 GB / 44.5 ms decode floor at 230 GB/s. G1–G5
implemented and green, G6 in progress, nothing committed yet:

- **G1** (`src/models/glm_dsa/config.*`, `binding.*`, `loaders/
  architecture.*`): the parser with its named rejections, the indexer
  schedule rule (21 full / 57 shared), the packed-shape derivation from the
  `config_groups` targets and `ignore` rules, the binding table (175,985
  tensors: 57,600 int4 and 600 int8 triples, 771 bf16 draft experts), the
  architecture detector; `unit_tests` 163/163 (8 new), `glm_dsa_bind_check`
  on the checkpoint.
- **G2** (`kernels/packq_gemv.{cuh,cu,hpp}`, `quant_matrix.hpp`,
  `glm_moe.cu` slot/grouped forms, `moe_layer.cpp`, `moe_reference.*`):
  the exact-dequant core (D2, revised), `packq_gemv_test` 5/5 plus four
  real checkpoint slices with packing pins, `glm_moe_test` 25/25 (4 new
  packq gates). The tensor-core tile kernel (D7) is deferred to the
  optimization stage; the grouped GEMV chain is the prefill until then.
- **G3** (`src/models/glm_dsa/loader.*`, `loaders/packq_quant.hpp`,
  `tests/cuda/glm_dsa_fixture.hpp`): the family on the shared stream —
  fused `[q_a | kv_a]` (packed or bf16), the `kv_b` bridge, the draft's
  requantization (D5), resident images; `glm_dsa_loader_test` 4/4 on the
  fixture; `glm_dsa_load_check` streaming layers 0–10 and 77–78 at world
  4: resident 99.30 GiB per rank (globals 2.22; dense 0.222 GiB, MoE 1.262,
  MoE+indexer 1.279, draft 1.465 after a 13 s requant of 4.78 GiB), the
  formulas and the source-byte plans reconciling.
- **G4** (`dsa_geometry.hpp`, `kernels/dsa.{hpp,cu}`, `topk_select.cuh`,
  `dsa_state.*`, `dsa_layer.*`, `dsa_reference.*`): `DsaConfig` gains
  `qk_rope_head_dim` ∈ {0, 64}, `index_kpool` 1, `index_relu`,
  `num_index_layers`; the rope tail in every cache format (D3), the RoPE
  kernel and table, kpool 1 through the existing compress/ring machinery
  with null gate/APE, the relu in both select key functors, select_k 2048
  (a 4096-key prefill tile, an eight-round expansion), the packed
  projections through the GEMV core in `DsaLayer`, the selection-reusing
  view (`DsaLayerWeights` without indexer tensors: the layer skips the
  indexer projections, the index-cache writes and the select, checks the
  reuse follows the indexed enqueue of the same shape, D4), the pool's
  layer → index-cache ordinal table, the host oracle extended the same
  way. `dsa_test` 43/43: the Flash 33 unchanged and bitwise where they
  were, plus RoPE bitwise host/device, the pool ordinals and byte formulas,
  full-geometry prefill / chunked prefill + decode on bf16, fp8 and fp4
  caches (0 selection flips, kept-row drift ≤ 4.2e-3), packed projections
  vs the bridge form, shared selection across two layers (prefill and
  decode, the contract's throws), TP2 head slicing, select_k 2048 bitwise
  in both selects, and a real-geometry smoke (64 heads, 576-wide rows,
  select_k 2048, decode repeat bitwise). `unit_tests` 6/6 geometry cases.

- **G5** (`src/models/glm_dsa/model.{hpp,cpp}`, `tools/glm_dsa_reference_dump.py`,
  `tests/cuda/glm_dsa_{forward,decode,tp,engine}_test.cpp`,
  `apps/glm_dsa_forward_check.cpp`): `GlmDsaModel` on the session core
  (the GLM-4.7 walk's shape with `DsaLayer` rebound per layer, the shared
  layers enqueued with an indexer-less view right after their full layer,
  a bf16 dense MLP for layers 0–2, the packed MoE, the draft block with
  its own indexer; per-token selection leaves no ring state any later
  row reads, so the speculative table is empty and snapshots are the
  block list plus the draft row, as for GLM-4.7). The fixture grew to 16
  heads (4 per rank at world 4, the split attention kernel's floor) and a
  32-head indexer; the pure-python reference (13 s for 32 tokens) mirrors
  the engine's rounding points and dumps every indexed layer's selection
  with its boundary margin. Gates, all green on 2026-09-12: the forward
  parity (layer 0 — pristine input — matches every selection including
  margins of 4e-4 and 0.2 % l2; the shared layer 1 0.5 %; deeper layers
  0.7–1.0 % on kept rows, their flips certified as near ties: the
  fixture's random indexer leaves the boundaries a few 1e-3 apart and a
  ~1 % hidden drift plus fp8 code flips move them — and a flipped
  token's cached key moves every later query's boundary, so flips
  cascade; the gates therefore score the rows before the first flip),
  the decode gate (prefill last row bitwise the forward, 110 steps
  across the 128-token block with kept rows within budget, per-step
  rolling snapshots bitwise, two interleaved slots bitwise, chunked
  prefill with a boundary cut, close/reopen, the prefix cache hot ==
  cold at a cut and mid-decode, the greedy speculator through the
  draft), the TP gate (worlds 2 and 4 on ports 29952/29953: ranks
  bitwise identical including the selections, the clean prefix within
  0.02 of world 1), the engine gate (world 2 on ports 29954–29956: the
  scalar, batched, MTP and depth-2 graph replays reproduce the eager
  transcripts exactly; depth 2 runs two slots; the sixteen-row gate on
  port 29960 runs eight slots at depth 1 against the eager engine).
  `glm_dsa_forward_check --layers 4` on the real checkpoint (streaming,
  boot 7 s, 27 tokens in 8.5 s) dumps the states the transformers
  cross-check (`tools/glm_dsa_torch_reference.py` on the 5090 box)
  compares; a `--diag` localizer in the decode test shows the first
  decode steps bitwise the cold forward at every layer. **The transformers
  cross-check (5090 box, 2026-09-12)**, transformers' own `GlmMoeDsa`
  layers on the real layers 0–3 in bf16 against the engine's dumps at 27
  and 2,112 tokens (the second crosses the 2,048 horizon): per layer,
  each fed the engine's own input (no accumulation), relative l2 0.0033 /
  0.0026 / 0.0023 / 0.0035 (short) and 0.0028 / 0.0021 / 0.0019 / 0.0025
  (long), max |d| one bf16 ulp; the final post-norm hidden 0.0039 / 0.0031;
  chained down the stack (each layer on transformers' own input, the
  errors accumulating) 0.0033 → 0.0062 and 0.0028 → 0.0049. Against a
  fp32 transformers reference of the dense layers 0–2 the engine's own
  error is 0.0031 / 0.0027 / 0.0025 (short) and 0.0030 / 0.0028 / 0.0026
  (long) — the bf16 pipeline's floor, the same level as GLM-4.7's 0.0025 /
  0.0026. No selection differs (all rows past the horizon match at the
  same level as the rest). The MoE layer's few 1–2 % rows are the routing
  near-ties: the tool prints each row's router margin (the 8th minus the
  9th score) and the worst rows hold the smallest margins of the prompt
  (4e-6 to 5e-5 against a median of 1.4e-4), one expert of eight
  differing.
- **G6, so far**: the `glm_moe_dsa` serving family in `dgpp-serve`
  (the DSA pool in the `--kv-dtype` format, the decode cap — eight rows
  then, sixteen since 2026-09-13 —,
  the pool-id check at one token per pool); the memory plan at world 4
  measures 99.30 GiB of weights + 1.43 GiB of caches at 16K bf16 (91.8
  KiB per context token; 52.6 KiB at fp8). The templates
  `deploy/cluster_glm-5.3_int4-int8_w4_{plain,mtp1,mtp1_large-cache}`
  carry 64K bf16 (105.00 GiB) / 64K bf16 (106.68) / 120K fp8 with a 1.5
  GiB prefix arena (107.12) — the ceiling under the 8 GiB headroom on the
  121.6 GiB nodes (128K fp8 leaves 0.4 GiB, 144K is refused; the plan's
  earlier 160K figure was arithmetic, not a plan run). The first boot on
  2026-09-12 found rank 2's node at 119.67 GiB — the NVIDIA DGX Spark on
  its launch firmware, whose display reservation is 4 GB instead of the
  2 GB of every later firmware — refusing the 64K plan (106.68 + 8
  against 114.43 GiB free), its peers' lanes then dying with "transport
  retry counter exceeded"; the campaign ran at 48K, and the same night the
  SoC firmware update (fwupd, two passes: the first applied only the EC)
  brought the node to 121.7 GiB and the templates back to these shapes;
  the tokenizer goldens (55 cases,
  byte-exact) and the chat-template goldens (26 cases; the template's
  `range(n)` needed the interpreter's one-argument form) registered as
  `glm_dsa_{tokenizer,chat_template}_test`. **The fabric (2026-09-12
  evening, `scripts/fabric_glm_dsa_serve.sh`, record
  `benchmarks/results/2026-09-12-glm53-full.md`):** boot 405 s the first
  time (the resident image captured on every rank), 30 s after; T=1 51.1
  ms/step; MTP depth 1 68–76 ms/pass at 1.77–1.97 tokens/pass (acceptance
  chat 77 %, code 97 %, prose 92 %, json 96 %, math 93 %), 36–42 ms/token
  — the post-final-norm draft convention (D6) confirmed; MTP == T=1
  transcripts 4/4; op streams identical across the ranks; four live
  requests 180–185 ms per eight-row step (aggregate 38–43 tok/s); prefill
  7.3 / 7.1 / 8.4 / 9.5 ms per token at 520 / 2.1K / 8.4K / 16.8K (the D7
  cost, measured); eval gsm8k 59/60, HumanEval 40/40, extraction 30/30
  with thinking on (the template has no `enable_thinking` knob); the API
  check green. Rows in docs/benchmarks.md §2–§9 and docs/measurements.md.
  Left for the optimization stage: D7 (the packed prefill tile kernel),
  the decode step against its 44.5 ms floor, a soak and a failure drill.
- **The memory headroom revisited (2026-09-12, late):** the growth after
  the plan check measured flat at 5.5–6.0 GiB (all four ranks, boot / 32K
  prefill / four live), 2.22 GiB of it the loader's pinned staging mirror;
  the shared stream's `release_sources()` now frees it with the mappings,
  the model materializes its stack in its constructor and releases there
  (before its caches, before any collective),
  the staging is a plan item (zero beyond the caches), the headroom is 5
  GiB; templates 96K bf16 MTP / 112K plain / 160K fp8, the 160K world
  booted on the four nodes.
- **The vocab-sharded embedding (2026-09-13, `engine.embed_sharding`):**
  each rank holds its lm-head slice of the rows (`GlmDsaLocalGeometry::
  embed_vocab_*`, the loader's `set_embed_vocab_sharded`), gathers its rows
  with zeros elsewhere and one fold sums them — bitwise the replicated
  lookup, on the main path and the draft's (a row scratch the fused input
  kernel indexes by an iota); −1.33 GiB per rank at world 4, +1 collective
  per T=1 step, +2 per MTP pass. Gates: the loader's slice, worlds 2 and 4
  bitwise the replicated worlds (transcripts, states, selections), the
  recorded graph world bitwise the replicated one. Fabric: weights 97.97
  GiB per rank, transcripts identical, pace unchanged. The headroom went
  to 4 GiB after a one-hour soak at the 120K MTP shape (flat memory, no
  reclaim on any node); templates 144K bf16 plain / 120K bf16 MTP / 208K
  fp8. The peers run headless (worth ~0.2 GiB, not a lever).

The packed prefill tile kernel (D7) is implemented for bulk prompts;
short prompts retain the grouped GEMV chain. The dated performance record
tracks its service, numerical and C1 promotion gates.

The dense lowering by rows and the group prefill (2026-09-14, carried
from the DeepSeek work): the DSA layer's fp8 projections take the
streaming tensor-core GEMM from five rows to 256 (`kernels/scale_gemm.hpp
mma_from_rows`; at sixteen rows o_proj 320 → 109 µs, q_a 155 → 59, q_b
104 → 38, kv_a 57 → 37 on one rank of four), the bf16 dense MLP, head and
indexer projections cuBLASLt's algorithm above four rows; the GEMV chunks
keep rows one to four (T = 1 unchanged). The group prefill runs the DSA
layer per span with a selection-scratch base per span
(`DsaLayer::enqueue_prefill(row_base)`; the selection-reuse contract per
base), so a selection-reusing layer finds every span's selection where its
indexed layer left it. Gates: `glm_dsa_decode_test` (a group's rows within
6.5e-3 relative l2 of the prefills alone, top-1 equal), the engine gates'
near-tie rule (docs/testing.md). Fabric numbers: docs/measurements.md.

## 7. Risks and open questions

- **Memory.** 99 GiB of weights per rank leaves 2–4 GiB for caches. If the
  plan refuses at four slots, the order of concessions is: fp4 latent, two
  slots, vocab-sharded embedding, plain decode. A model that serves 30k
  tokens of context at four slots is the likely first result; the
  replicated latent is the reason and context parallelism the only cure.
- **The 576-wide attention row** — resolved in G4: the flash kernels take
  the score width as a second template width (91 KB of shared memory, one
  block per SM for the 576 variant), the split kernel a rope window; Flash
  bitwise, the oracle at 576 green.
- **select_k 2048 in shared memory** — resolved in G4: the prefill select's
  tile doubles to 4096 keys (56 KB, opted in), the decode select's radix
  path needs no tile (39 KB at one row, 69 KB at eight); the expansion's
  rank sort at 2048 ids costs ~n²/threads compares and is a measured item
  for the optimization stage.
- **Draft hidden convention** (post- vs pre-norm) and the draft's own
  indexer at depth 1: settled by acceptance, both variants cheap.
- **Prefill select cost at kpool 1**: four times the materialized dots per
  context length; measured, and the dot budget bounds the footprint.
- **The template's `clear_thinking` default** changes what a multi-turn
  prompt contains; the API's handling of prior `reasoning_content`
  (`reasoning_in_content`, the Hermes client's stripping) needs one look at
  G6 so the prefix cache behaves.
- **AWQ vs RTN** is unresolved end to end; the format is shared, so the
  choice costs one more download and one evaluation pass, not code.
- **Two nodes** do not fit (198 GiB of weights per rank); no `w2` template.
