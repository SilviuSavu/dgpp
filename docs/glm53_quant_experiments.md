# GLM-5.3 quantization experiments

Completed experiment, 2026-10-08. A custom four-bit integer codebook is the
strongest quantization family tested for full GLM-5.3 on four GB10 nodes.
Group 64 leads on quality; group 128 is the better memory/performance
candidate. The results support developing this format, but do not establish
a faster, equally accurate production replacement yet.

## Constraints

| Item | Constraint / measured setup |
|---|---|
| Deployment hardware | Four GB10 nodes, tensor parallelism 4, MTP1 |
| Current quant | `HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64`, revision `147684fbad20c1e283ddff46fd07cb9d4ccbb3da` |
| Original weights | `zai-org/GLM-5.3-BF16`, revision `9d2398f478cab2de883137db3a36ad2c96205e24` |
| Current per-node weights | Approximately 97.59 GiB; routed experts account for 90.84375 GiB, including the draft layer |
| Experiment GPU | RTX PRO 6000 Blackwell Server Edition, SM120, 97,887 MiB reported VRAM |
| Experiment host memory | Container limit 187,999,997,952 bytes; the host's reported memory is not the container budget |
| Memory discipline | No swapping by experiment or DGPP processes; leave OS swap enabled |
| Accuracy measures | Local expert-output error and full-model teacher-forced scores; generation accuracy remains untested |
| Performance measures | Runpod screening; GB10 measurements required for deployment claims |
| Storage cleanup | Session-downloaded Runpod models and large captures removed after completion; results and provenance preserved. Pre-existing cluster models retained. |

## Recommended strategy

Develop the **NF4-like integer-codebook family**, with group 64 as the
quality reference and rotated group 128 as the hardware candidate. Keep
BF16 decode activations. Native INT8 prefill is promising at larger batches,
but its current 2K regression needs work before it becomes a default.

| Choice | Weight memory vs current | Quality evidence | Performance evidence / limitation |
|---|---:|---|---|
| Codebook g64, BF16, no rotation | Same | Better likelihood and fidelity on all three final documents | Decode slightly slower; BF16 prefill not benchmarked |
| Codebook g64, H32, INT8 prefill | Same | Best aggregate fidelity; all three document NLLs improve | Native prefill wins at 4K but regresses at smaller sizes; rotated BF16 decode is slower |
| Codebook g128, H32, INT8 prefill / BF16 decode | **−2.64 GiB/node** | About 22% lower KL; aggregate likelihood tied, small math regression | About 7% faster routed prefill at 4K, one 2K case slower; roughly 2–4% faster routed decode at 2–8 rows |

The group-128 recipe stores four-bit indices and one BF16 scale per 128
weights. H32 rotation and GPTQ calibration are applied offline. Prefill
unpacks indices directly into INT8 registers; it does not keep an expanded
INT8 copy of the model. The residual stream and nonlinear operations remain
BF16, and the KV format is unchanged. The MTP draft is held fixed.

Before replacing the deployed checkpoint:

1. Add DGPP loader/kernel support for the checkpoint's explicit format tag and
   codebook tensors. The existing INT4 loader cannot interpret these indices.
2. Verify DGPP runtime numerics, generation quality, longer contexts, and MTP
   acceptance against the current quant. These teacher-forced scores are a
   screening result, not a task-accuracy guarantee.
3. Measure full serving prefill/decode and memory use at the current chunk
   budgets. Fix the regressing prefill cases before enabling that path by
   default. Standalone expert-block gains do not establish serving gains.

The saved `candidate-recipe.json` describes group 128 precisely; it is a
research recipe. The subsequent checkpoint build is recorded below; DGPP
runtime integration remains separate.

## Checkpoint production

The group-128 checkpoint was produced exclusively on Runpod on 2026-10-08.
All 79 shards completed export and serialization checks at 16:21 UTC. Publication
and the independent public-repository audit finished at 16:43 UTC:
[HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128](https://huggingface.co/HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128).
The repository includes the complete tensor index, verified shard manifests,
and [completion audit](https://huggingface.co/HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128/blob/main/audit-complete.json).

| Item | Production behavior |
|---|---|
| Checkpoint size | 416,111,319,432 file bytes (416.1 GB), including 175,985 tensors in 79 shards |
| Routed experts | Recalibrate the 75 main MoE layers with the evaluated H32/GPTQ/group-128 recipe |
| Other tensors | Preserve the pinned baseline exactly, including the complete MTP draft |
| Converted payload | INT32 `weight_indices`, BF16 `weight_scale`, INT64 `weight_shape`; a distinct key prevents accidental INT4 interpretation |
| Numerical checks | Reconstruct every encoded matrix bit for bit; compare all three expert-0 matrices per layer to the evaluated quantizer |
| Serialization checks | Read back and compare every tensor in each newly serialized layer |
| Publication checks | Verify every uploaded shard's size and SHA-256 against Hub metadata |
| Completion | All 79 public shard hashes match; index, recipe, tensor shapes/dtypes, and completed model card verified anonymously |
| Swap | Zero bytes throughout the build |
| Checkpoint revision | `365d2797d2252a739951560d44c6b8cd65733432` |
| Audit publication revision | `c389123289248fa5d3bc7c18502183821ac91a67` |

The exporter, producer, and uploader are `tools/glm53_quant_checkpoint.py`,
`tools/glm53_quant_produce.py`, and `tools/glm53_quant_publish.py`. They do not
add a runtime loader or change the serving engine. The source GLM-5.3 license
is preserved verbatim in the model repository.

Hugging Face's hourly commit limit delayed publication after 63 shards; the
remaining 16 were staged and committed together when the window cleared.
The uploader now includes progress in shard commits and supports staging a
batch before a scheduled commit. This delay did not change the checkpoint.

The completed checkpoint remains at
`/workspace/dgpp-fp8-experiment/production/checkpoint` on Runpod. Small manifests,
logs, and source provenance are preserved under
`benchmarks/results/2026-10-08-glm53-nf4i8-production/`; no model weights were
produced or downloaded into the local checkout.

## Full-model results

All eleven recipes/references completed all 78 main layers. Each of the
three held-out documents contributes 8,191 scored next-token positions.
Negative log likelihood (NLL) is lower when the observed text receives
higher probability. KL divergence measures how far the complete predicted
vocabulary distribution moves from original BF16; lower is closer. Top-token
agreement measures fidelity to BF16, not whether generated answers are correct.

| Recipe | Code NLL | Prose NLL | Math NLL | Mean NLL | Mean KL to BF16 | Top-token agreement |
|---|---:|---:|---:|---:|---:|---:|
| Original BF16 | 1.46674 | 1.60069 | 0.64501 | 1.23748 | 0.00000 | 100.00% |
| Current INT4/INT8 | 1.47250 | 1.60219 | 0.65604 | 1.24358 | 0.03577 | 93.57% |
| Calibrated INT4 / BF16 | 1.47363 | 1.60490 | 0.65985 | 1.24613 | 0.03374 | 93.92% |
| H32 INT4 / BF16 | 1.47181 | 1.59861 | 0.65779 | 1.24274 | 0.03307 | 94.03% |
| H32 INT4 / INT8 | 1.47112 | 1.60053 | 0.65602 | 1.24256 | 0.03289 | 94.10% |
| H32 MXFP4 / MXFP8 | 1.46658 | 1.59524 | 0.65312 | 1.23831 | 0.03578 | 93.68% |
| Codebook g64 / BF16, no rotation | 1.46662 | 1.59878 | 0.64456 | 1.23665 | 0.02755 | 94.32% |
| H32 codebook g64 / BF16 | 1.46924 | 1.60188 | 0.65486 | 1.24199 | 0.02659 | 94.53% |
| H32 codebook g64 / INT8 | 1.46984 | 1.60060 | 0.65446 | 1.24164 | 0.02597 | 94.62% |
| H32 codebook g128 / BF16 | 1.46859 | 1.60408 | 0.65864 | 1.24377 | 0.02776 | 94.23% |
| H32 codebook g128 / INT8 | 1.47080 | 1.59864 | 0.65903 | 1.24282 | 0.02805 | 94.14% |

The codebook family gives the strongest combined likelihood/fidelity results:

- **Group 64, BF16, no rotation:** improves NLL and KL on all three documents.
  Mean NLL falls from 1.24358 to 1.23665; aggregate KL falls 23.0%.
- **Group 64, H32, INT8:** improves NLL and KL on all three documents and
  has the lowest aggregate KL, 27.4% below the current quant. Its NLL gain
  is smaller than the unrotated variant's.
- **Group 128, H32:** reduces aggregate KL by 22.4% with BF16 activations and
  21.6% with INT8. Aggregate NLL is effectively tied: changes are +0.00020
  and −0.00075. Math perplexity increases about 0.26% and 0.30%, respectively.
  The INT8 variant improves code and prose NLL; the BF16 variant improves
  code but slightly worsens prose.

A paired 512-token block bootstrap, stratified within the same three
documents, gives these conditional 95% intervals. Negative changes favor
the candidate.

| Recipe | NLL change vs current | KL change vs current |
|---|---:|---:|
| Codebook g64 / BF16, no rotation | -0.01179 to -0.00190 | -0.01015 to -0.00629 |
| H32 codebook g64 / INT8 | -0.00663 to +0.00281 | -0.01167 to -0.00804 |
| H32 codebook g128 / BF16 | -0.00409 to +0.00433 | -0.00995 to -0.00611 |
| H32 codebook g128 / INT8 | -0.00471 to +0.00320 | -0.00934 to -0.00611 |

The codebooks' fidelity improvement is consistent in this sample. Among the
codebooks, only unrotated group 64 shows a clear NLL improvement in this conditional
comparison. These intervals do not estimate uncertainty across a broader
task population. They do not prove equal or better generation accuracy,
long-context behavior, or MTP acceptance.

The NLL and fidelity rankings differ. MXFP4/MXFP8 improves NLL in these
documents while leaving aggregate KL essentially unchanged. Unrotated
calibrated INT4 raises NLL on all three documents. Choosing a format solely
from local expert error or one aggregate language-model metric would miss
these differences.

## GB10 performance

These measurements cover one routed expert block on GB10, using TP4 matrix
shapes, captured original-model inputs/routes, and synthetic resident weights.
They include gate/up, SwiGLU, down, and the candidate's activation transforms.
They exclude attention, shared experts, routing itself, collectives, and MTP
acceptance. They are not full-model serving measurements.

The serving model was unloaded during timing and restored afterward. All
four DGPP processes reported zero swap after restoration. The benchmarks
check their own swap usage and available system memory.

### Prefill

Median milliseconds across six counterbalanced trials; each trial measures
100 calls in CUDA graphs. Every timed allocation uses `cudaMalloc`. Gate/up
quantize each original token once and reuse it across projections/experts.
Gate/up outputs are BF16 and down outputs are FP32. All calibrated candidates
in this table include H32 rotation.

| Captured routes | Input tokens | Current INT4 / BF16 | INT4 / BF16 | INT4 / INT8 | MXFP4 / MXFP8 | Codebook g64 / INT8 | Codebook g128 / INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Layer 63 | 256 | 4.985 | 5.018 | 5.162 | 5.143 | 5.151 | 4.978 |
| Layer 63 | 512 | 6.129 | 6.197 | 6.350 | 6.323 | 6.334 | 6.121 |
| Layer 63 | 2,048 | 9.636 | 9.981 | 9.817 | 9.731 | 9.772 | 9.590 |
| Layer 63 | 4,096 | 15.462 | 16.172 | 14.740 | 14.478 | 14.546 | 14.430 |
| Layer 3 | 2,048 | 9.366 | 9.658 | 9.880 | 9.845 | 9.865 | 9.663 |
| Layer 3 | 4,096 | 15.254 | 15.950 | 14.633 | 14.431 | 14.393 | 14.308 |

For group 128, paired trial throughput changes are approximately zero in
most 256–2,048-token cases, **−3.2%** at layer 3 / 2,048 tokens, and
**+6.7–7.0%** at 4,096 tokens. The current idle prefill budget is 2,048 tokens
and the loaded budget is 256; the 4K advantage does not establish faster
cold prefill with current settings. Any larger chunk budget must also account
for attention work, latency, and workspace.

### Decode

Median microseconds across six counterbalanced trials, 200 calls per trial.
Inputs/routes come from layer 63. Eight rotating weight banks limit repeated
L2 reuse. The baseline uses the existing fused packed slot kernels. Codebook
kernels use the same packed-core geometry with a different code lookup; group
128 also halves scale storage and the number of scaled group reductions.

| Input rows | Current INT4 / BF16 | H32 INT4 / BF16 | Codebook g64 / BF16 | H32 codebook g64 / BF16 | H32 codebook g128 / BF16 |
|---|---:|---:|---:|---:|---:|
| 1 | 177.09 | 178.35 | 180.01 | 183.56 | 175.74 |
| 2 | 345.84 | 347.27 | 349.17 | 350.89 | 338.85 |
| 4 | 679.56 | 682.59 | 684.38 | 687.77 | 660.16 |
| 8 | 1253.91 | 1256.69 | 1263.09 | 1266.49 | 1210.50 |

Group 128 is effectively tied at one row: paired trial throughput changes
range from −2.9% to +1.6%. At two, four, and eight rows, paired median gains
are **2.4%, 2.9%, and 3.7%**. End-to-end decode gains will also depend on the
unchanged work and any change in MTP acceptance.

### Kernel validation and discarded timing methods

- Native SM121a INT8 and mixed FP8 × FP4 kernels pass independent CPU
  double-precision output checks and activation-code checks.
- Timing uses explicit CUDA streams and includes rotation/quantization.
  The native prefill kernel adapts to 16-, 32-, and 64-row expert tiles and
  prefetches weights two K tiles ahead.
- The device-allocation conversion preserves the checked mixed-kernel output
  bit for bit. The codebook decode checks include fused gate/up nonlinear
  operations and down-projection output.
- Earlier managed-memory, regular-batch, fixed-tile, and short GEMV timings
  are diagnostic only. Use `gb10-final-prefill.jsonl` and
  `gb10-final-decode.jsonl` for the tables above.

## Candidate weight storage

The changes below apply to all main and draft routed-expert weights. Other
weights are held fixed. They exclude transient kernel workspace. Activation
quantization does not change the KV format in these experiments.

| Routed format | Effective bits/weight | Weight change per node | Execution consideration |
|---|---:|---:|---|
| Current INT4, BF16 scales / 64 | 4.25 | Baseline | BF16 activations; software code unpacking and FP32 group scaling |
| Calibrated INT4, same layout | 4.25 | 0 | Can retain BF16 decode; test INT8 or FP8 activations for prefill |
| INT4, BF16 scales / 128 | 4.125 | −2.672 GiB | Larger groups trade quality for space; not the current loader format |
| NF4-like INT8 codebook, four-bit indices, BF16 scales / 64 | 4.25 | 0 | Decode indices to INT8 levels; native INT8 multiplication is possible with quantized activations |
| NF4-like INT8 codebook, BF16 scales / 128 | 4.125 | −2.672 GiB | Larger groups; requires a new explicit format tag and kernels |
| MXFP4, E8M0 scales / 32 | 4.25 | 0 | Native mixed FP8 × FP4 math; BF16 activation kernels are also possible |
| E2M1, BF16 scales / 64 | 4.25 | 0 | Flexible scales; requires software scaling |
| E2M1, E4M3 scales / 32 | 4.25 + global scalar | Approximately 0 | Not the native MXFP4 scaling layout |
| NVFP4, E4M3 scales / 16 | 4.5 + global scalar | +5.344 GiB | Native FP4 × FP4; consider selective use or an offsetting storage reduction |
| MXFP6, E8M0 scales / 32 | 6.25 | +42.75 GiB | Selective precision only under this memory budget |
| INT8 or MXFP8, scales / 64 or / 32 | 8.25 | +85.50 GiB | Quality reference or selective precision, not a full routed-weight replacement |

The storage arithmetic is 21.375 GiB per additional routed-weight bit per
node. A block rotation has no extra per-weight storage; its activation
transform and scratch must be measured. A mixture of formats must satisfy
the same total budget.

For the current experiments only the 75 main MoE layers change. Leaving
the existing MTP draft quant unchanged, group 128 saves **2.637 GiB per
node**, rather than the 2.672 GiB that also converts the draft. Estimated
resident weights would fall from 97.59 to about 94.95 GiB per node.

The native INT8 prototype quantizes gate/up inputs once per token and down
inputs once per selected expert. With top-8 routing and TP4 shapes, its
additional code-and-scale buffers require approximately
`tokens × (6144 + 8 × 512) × (1 + 4/128)` bytes: **20.625 MiB per node at
2,048 prefill tokens**. Rotation is fused into that quantization. Separate
BF16 decode rotation buffers require about 0.313 MiB for 16 input rows.
These are workspace estimates, not a measured full-engine memory ledger;
the existing KV format and capacity are held fixed.

## Method and validation

- Capture activations using original BF16 weights and routing. Preserve
  FP32 router correction biases.
- Stream one original layer at a time through the experiment GPU. This
  matches the resident-prefix captures bit for bit at layers 3 and 6.
  Calibration captures now cover every main MoE layer, 3 through 77.
- Initial calibration used 2K prefixes of three documents; evaluation used
  their disjoint 8K suffix positions. These are **not independent documents**.
- The independent experiment instead calibrates on three 8K documents and
  evaluates on three separate 8K documents: different repository code,
  GSM8K test rather than training examples, and different technical docs.
- Compare unchanged INT4, newly calibrated INT4, several FP4 grids, FP8 and
  INT8 activation grids, and a 32-element Hadamard rotation with GPTQ-style
  error compensation. NVFP4 also has an FP4-activation check.
- GPTQ identity-Hessian tests must reproduce ordinary rounding exactly.
  Rotation followed by inverse rotation must reproduce FP32 inputs within
  numerical tolerance.
- Later comparisons keep represented code-times-scale weights in FP32,
  avoiding an extra BF16 weight-rounding step in the initial screen.
  GEMM summation is a mathematical reference, not bitwise DGPP emulation.
  Linear outputs/nonlinear operations are BF16 in the layer-local screen.
- Aggregate only the routed contribution using original routing weights.
  Shared experts and the residual do not dilute the error measure.

Relative output error is not a task accuracy loss. Original routing also
does not establish behavior after quantization errors accumulate through
the full model. The full-model scores below address accumulation and routing. Generation
checks and actual serving measurements are still required before replacing
the quant.

## Full-model quality protocol

The completed full-model runs use three new 8K documents: unseen
repository code, unseen documentation, and GSM8K test records 200–399.
Calibration and damping selection use only the earlier data. Sources and
hashes are saved in the run manifest.

- Compare original BF16, existing INT4/INT8, calibrated INT4, rotated INT4
  with BF16 or INT8 activations, and rotated MXFP4 with MXFP8 activations.
- Propagate each candidate's own hidden states, expert routing, and sparse
  attention choices through all 78 main layers.
- Hold non-routed weights fixed to the actual current checkpoint for every
  quantized candidate, including its INT8 attention and shared experts.
- Score next-token negative log likelihood, perplexity, KL divergence from
  original BF16, and top-token agreement. Save per-token values for paired
  comparisons. These are teacher-forced scores, not generation task accuracy.
- Stream one expert's weights at a time. The original expert computation
  matches a separately instantiated resident Hugging Face expert module
  bit for bit in both the short and 8K validation checks.
- Retain HF's BF16 expert-output rounding and accumulation in this numerical
  comparison. DGPP's FP32 routed accumulation and BF16 expansion of `kv_b`
  differ from this arithmetic; MTP behavior and serving throughput also
  require runtime checks. DGPP's BF12 storage is lossless for the affected
  BF16 weights and does not introduce a separate quantization error.

No expert is silently omitted from full-model scoring. If an expert has
fewer than 16 routed calibration tokens, calibration uses a fixed subset
of the calibration pool; the fallback count is recorded per layer.

A follow-up pass evaluated the integer codebook with BF16 activations,
with rotation, and with rotation plus INT8 activations. It reuses the first
pass's final original/baseline states for scoring, after checking the prompt
hash and token identity. It still propagates every candidate independently
through all layers.

The final candidate pass evaluated group 128 with rotation, with either
BF16 or INT8 activations. No further formats are being added to this
comparison. Both integer-codebook group sizes match the unfused Torch
calibration implementation bit for bit on a full expert's gate, up, and
down matrices in the saved parity check.

## Broader validation

All-expert checks use damping 0.01 at layers 3, 6, and 21, and 1.0 at
layers 42, 63, and 77. Relative L2 error below measures the combined routed
output against original BF16, on the separate validation documents.

| Recipe | Layer 3 | Layer 6 | Layer 21 | Layer 42 | Layer 63 | Layer 77 |
|---|---:|---:|---:|---:|---:|---:|
| Existing INT4 / BF16 | 8.091% | 10.024% | 12.433% | 14.136% | 16.182% | 12.428% |
| Calibrated INT4 / BF16 | 4.560% | 8.410% | 11.443% | 12.292% | 15.118% | 9.388% |
| Rotated, calibrated INT4 / BF16 | 3.186% | 7.030% | 10.890% | 12.025% | 14.772% | 8.655% |
| Rotated, calibrated INT4 / INT8 | 3.220% | 7.056% | 10.921% | 12.062% | 14.810% | 8.691% |
| Rotated, calibrated MXFP4 / BF16 | 3.488% | 7.623% | 11.631% | 12.866% | 15.792% | 9.429% |
| Rotated, calibrated MXFP4 / MXFP8 | 4.052% | 8.069% | 12.071% | 13.374% | 16.323% | 9.951% |

Coverage is all 256 experts except layer 21: one expert had fewer than 16
calibration tokens and was skipped. That layer covers 99.9898% of validation
routing assignments. The baseline is evaluated on the same covered experts.
INT8 activations use dynamic FP32 scales per 128 values; residuals and
nonlinear operations remain BF16. All these weight recipes occupy 4.25 bits
per weight.

Rotated INT4 is the strongest of these same-size candidates on this local
measure. MXFP4 with FP8 activations has a small layer-63 regression against
the current quant, so it has not met even this preliminary quality gate.

A wider datatype screen covers 16 uniformly spaced experts at three depths.
Its numbers are not directly comparable to the all-expert table above.

| Calibrated weight format / BF16 activations | Bits/weight | Layer 3 | Layer 21 | Layer 63 |
|---|---:|---:|---:|---:|
| Existing INT4, uncalibrated baseline | 4.25 | 8.678% | 11.909% | 15.784% |
| INT4 group 64, rotated | 4.25 | 3.058% | 8.608% | 13.973% |
| INT4 group 128, rotated | 4.125 | 3.347% | 9.389% | 15.230% |
| MXFP4, rotated | 4.25 | 3.356% | 9.201% | 14.940% |
| NVFP4 | 4.5 | 3.963% | 8.178% | 12.661% |
| E2M1, E4M3 scales / 32 | 4.25 | 4.342% | 8.741% | 13.527% |
| E2M1, BF16 scales / 64 | 4.25 | 4.584% | 9.076% | 14.058% |
| INT4 group 128 gate/up + NVFP4 down | 4.25 weighted average | 4.177% | 9.088% | 14.348% |
| INT8 group 64, quality reference | 8.25 | 0.336% | 0.632% | 0.917% |

Full NVFP4 and INT8 would consume too much of the per-node memory budget.
The mixed INT4/NVFP4 recipe offsets NVFP4's extra down-projection storage
with larger INT4 groups in gate/up. Native NVFP4 activation quantization
did not preserve the baseline's local quality at every tested depth; its
layer-63 error was 18.992%. A smaller weight error alone is not sufficient.

### NF4 and an integer codebook

The NF4 follow-up uses the published QLoRA/bitsandbytes 16-level grid,
BF16 scales per 64 values, and the same calibration and rotation procedure.
It does not use double quantization or train adapters. A second grid rounds
those levels to integers after multiplying by 127:

`−127, −88, −67, −50, −36, −23, −12, 0, 10, 20, 31, 43, 56, 71, 92, 127`.

Both store a four-bit index per weight. The second grid can decode those
indices to exact INT8 operands for native integer multiplication; the
BF16-activation decode path can also represent each integer exactly.
This is a custom format, not a checkpoint the existing INT4 loader can read.

The same 16-expert validation sample gives:

| Calibrated recipe | Layer 3 | Layer 21 | Layer 63 |
|---|---:|---:|---:|
| Existing INT4 / BF16 | 8.678% | 11.909% | 15.784% |
| Rotated INT4 / BF16 | 3.058% | 8.608% | 13.973% |
| NF4 / BF16 | 3.975% | 7.915% | 12.267% |
| Rotated NF4 / BF16 | 2.652% | 7.306% | 11.861% |
| Integer codebook / BF16 | 3.989% | 7.929% | 12.296% |
| Rotated integer codebook / BF16 | 2.653% | 7.322% | 11.894% |
| Rotated integer codebook / INT8 | 2.695% | 7.360% | 11.935% |
| Rotated integer codebook, group 128 / BF16 | 2.763% | 7.587% | 12.350% |
| Rotated integer codebook, group 128 / INT8 | 2.804% | 7.625% | 12.390% |

The integer approximation preserves most of NF4's advantage in this screen.
Its activation emulation retains the FP32 group scales, matching the native
integer representation without reconstructing BF16 inputs first. The earlier
INT8 activation screen rounded that reconstruction to BF16; a paired check
changed error by less than 0.004 percentage points at these three depths.
These are local screening results; the separate full-model scores are reported above.

INT3 was also screened. With rotation it measured 7.346%, 20.917%, and
33.463% at layers 3, 21, and 63. It fails the quality requirement as a
whole-model replacement despite the attractive storage reduction.

## Completed initial checks

All-expert results below use the initial same-document suffix evaluation.
All 256 experts were covered in each layer.

| Weight / activation recipe | Layer 3 relative L2 | Layer 6 relative L2 |
|---|---:|---:|
| Existing INT4 / BF16 | 8.490% | 10.053% |
| Plain MXFP4 / BF16 | 10.443% | 11.642% |
| Calibrated MXFP4 / BF16 | 6.681% | 9.964% |
| Calibrated MXFP4 / MXFP8 | 7.165% | 10.330% |

Calibration helps, but these results do not support plain MXFP4 as an
automatic replacement. Layer 6 also shows why the early-layer gain alone
is insufficient.

The separate-document layer-21 pilot covers 16 uniformly spaced expert
IDs; these are combined outputs from those experts, not the whole layer.

| Recipe | Relative L2 |
|---|---:|
| Existing INT4 / BF16 | 11.909% |
| Calibrated INT4 / BF16 | 9.324% |
| Rotated, calibrated MXFP4 / BF16 | 9.201% |
| Rotated, calibrated MXFP4 / MXFP8 | 9.710% |
| Rotated MXFP4 + MXFP8 gate/up; calibrated INT4 + BF16 down | 9.765% |

The full layer-3 separate-document validation subsequently covered all 256
experts: existing INT4 measured 8.091%, calibrated INT4 4.560%, and rotated
MXFP4 3.488% with BF16 or 4.052% with MXFP8 activations.

The initial small-sample calibration regressed at deeper layers. Increasing
the diagonal Hessian regularization from 0.01 to 1.0 corrected the regression
in the 16-expert validation pilots below. These validation documents now
inform hyperparameter selection; they must not be presented as untouched
final test data.

| Recipe | Layer 42 | Layer 63 |
|---|---:|---:|
| Existing INT4 / BF16 | 14.981% | 15.784% |
| Calibrated INT4 / BF16, damping 0.01 | 16.633% | 19.358% |
| Calibrated INT4 / BF16, damping 1.0 | 12.736% | 14.445% |
| Rotated MXFP4 / BF16, damping 1.0 | 13.329% | 14.940% |
| Rotated MXFP4 / MXFP8, damping 1.0 | 13.871% | 15.472% |

Using all calibration tokens rather than only an expert's routed tokens
also reduced the regression, but damping 1.0 performed better in these two
pilots.

## Reproduction and evidence

Experimental tools (no production dispatch changes):

- `tools/glm53_fp8_experiment.py`: original prefix captures and initial grid screen.
- `tools/glm53_mxfp4_calibrate.py`: initial sequential Hessian compensation.
- `tools/glm53_stream_capture.py`: bounded-memory original-model captures.
- `tools/glm53_quant_candidates.py`: broader grids, independent documents,
  rotations, and mixed gate/up versus down recipes.
- `tools/glm53_quant_gptq_triton.py`: fused calibration block updates.
  It matches the Torch reference bit for bit across all three full expert
  matrices in INT4, MXFP4, and NVFP4 in the saved parity check. The reference
  stays available; this speeds up the experiment, not serving.
- `tools/glm53_mxfp4_mxfp8_bench.py`: pinned CUTLASS native mixed-math check on Runpod.
- `tools/glm53_quant_microbench.cu` and `tools/glm53_mxfp4_mxfp8.cuh`: GB10 prototype comparison.
- `tools/glm53_quant_full_eval.py`: streaming full-model quality comparison.
- `tools/glm53_quant_ragged_bench.cu`: GB10 test with captured routing and
  correct activation reuse across gate/up and experts.
- `tools/glm53_quant_decode_bench.cu`: fused decode paths with rotating
  weight banks; codebook decode uses an isolated copy of the packed core.
- `tools/glm53_make_codebook_header.py`: generate that copy, changing the
  four-bit code lookup, namespace, and optionally scale group; records the source hash.
- `tools/glm53_quant_summary.py`: validate and aggregate the final score and
  timing artifacts without loading model weights.
- `tools/glm53_quant_uncertainty.py`: paired block-bootstrap sensitivity
  within the three scored documents; this does not measure uncertainty
  across independent tasks or a broader document population.

Result artifacts are under
`benchmarks/results/2026-10-08-glm53-w4a8-pilot/`. The canonical evidence is:

- `full-final`, `full-nf4-int8`, and `full-nf4-g128`: full-model manifests,
  scores, and per-token metrics.
- `gb10-final-prefill.jsonl` and `gb10-final-decode.jsonl`: all timing trials.
- `summary.json` and `full-quality-uncertainty.json`: derived comparisons.
- `candidate-recipe.json`, `experiment-sources.json`, and the prompt/token
  manifests: calibration, hardware, source hashes, and evaluated inputs.
- `quant-evidence-final.tar.gz`: 289 small Runpod artifacts and source files,
  verified against the included SHA-256 manifest. It contains no model weights.
- `cleanup.json`: approximately 2.03 TB of session-generated weights and
  large captures removed from Runpod after the evidence was verified locally.

Runpod's peak cgroup swap counter remained zero. The four-node Flash serving
deployment was restored, with zero process swap on all ranks. Runpod has no
remaining experiment GPU jobs.

Rebuild the derived summary with:

```sh
python3 tools/glm53_quant_summary.py \
  benchmarks/results/2026-10-08-glm53-w4a8-pilot \
  --output benchmarks/results/2026-10-08-glm53-w4a8-pilot/summary.json
```

The full-model manifests contain each run's arguments and input hashes;
the evidence archive includes the executed Runpod sources. Repeating model
scoring requires downloading the pinned checkpoints and regenerating the
calibration captures. Saved per-token metrics can be analyzed without them.
The native CUDA build must explicitly select
`-gencode arch=compute_121a,code=sm_121a`; the generic SM121 fallback does
not accept the architecture-specific block-scale instruction.

Methods and hardware references:

- [GPTQ](https://arxiv.org/abs/2210.17323): sequential Hessian error compensation.
- [MR-GPTQ](https://arxiv.org/abs/2509.23202): motivates format-specific calibration
  and block rotations. This experiment is a smaller implementation, not a
  reproduction of every component in that paper.
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma): supported matrix operand and scaling formats.
- [NVIDIA NVFP4 description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): group-16 E4M3 scales and a tensor-wide scale.
- [QLoRA](https://arxiv.org/abs/2305.14314) and the
  [bitsandbytes NF4 grid](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py): the nonuniform four-bit grid used in the follow-up.
