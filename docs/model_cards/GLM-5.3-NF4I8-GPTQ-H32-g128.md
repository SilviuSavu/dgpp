---
license: other
license_name: glm-5.3
base_model: zai-org/GLM-5.3-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags: [glm, moe, quantization, gptq, nf4i8, custom-format, dgpp]
---

# GLM-5.3 NF4I8 GPTQ H32, group 128

A quantization of the full **GLM-5.3** (754B-parameter MoE, model type `glm_moe_dsa`) from the
original BF16 release, for the DGPP engine on four DGX Sparks (TP=4). It is the
[Int4/Int8 RTN checkpoint](GLM-5.3-Int4-Int8Mix-RTN-g64.md) with only the routed experts of
layers 3–77 re-encoded: four-bit indices into a fixed sixteen-level INT8 codebook, GPTQ
calibration, and a block-32 Hadamard rotation of the expert inputs.

| | |
|---|---|
| Quantized components | gate, up and down projections of the 256 routed experts, main MoE layers 3–77 |
| Weight representation | 4-bit indices into the codebook `[-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127]` (round(127 × NF4)); weight = level × scale |
| Scales | one BF16 scale per 128 input channels per output row (4.125 bits/weight with scales) |
| Rotation | normalized 32-wide Sylvester Hadamard on the expert input (gate/up) and on the SwiGLU output (down); never on the residual, router, shared expert or draft |
| Calibration | GPTQ, 128-column blocks, three 8K documents (code, math, technical prose) |
| Everything else | byte for byte the Int4/Int8 checkpoint: int8 g64 attention and shared expert, BF16 dense layers 0–2, indexers, routers, norms, embedding, `lm_head`, MTP block |
| Tensors | `weight_indices` I32 `[N, K/8]` (low nibble first), `weight_scale` BF16 `[N, K/128]`, `weight_shape`; `config.json` `quant_method: dgpp_nf4i8`, `format: mixed-codebook-packed`, version 1 (`FORMAT.md` in the repository) |
| Size | 416.1 GB in 79 shards; tensor payload 387.51 GiB (the Int4/Int8 checkpoint: 398.06), 2.64 GiB less per rank at TP=4 |

## The publisher's accuracy table

Teacher-forced through all 78 main layers on three held-out 8K documents (24,573 scored
tokens), against the original BF16 model:

| Model | Mean NLL | Mean KL to BF16 | Top-token agreement |
|---|---:|---:|---:|
| Original BF16 | 1.23748 | 0 | 100 % |
| Int4/Int8 RTN g64 | 1.24358 | 0.03577 | 93.57 % |
| **NF4I8 H32 g128** (BF16 activations) | 1.24377 | 0.02776 | 94.23 % |

22 % less divergence from BF16 at an NLL change whose 95 % interval spans zero. The card
also measured an INT8-expert-activation prefill variant (KL 0.02805); the engine serves the
BF16-activation form.

## DGPP serving notes (2026-10-09)

**The format in the engine:** packed scale format 2 (`kPackedScaleBf16G128Nf4i8`,
`models/quant_matrix.hpp`) next to the int4/int8 formats, detected from `config.json` and
refused on any drift of the codebook, group, rotation or layer range. The GEMV core multiplies
the exact integer level by the bf16 group scale in fp32 (the codebook from two byte-permute
lookups); the tensor-core prefill decodes the levels to their exact bf16 patterns through the
same kind of lookup (`kernels/packq_gemm.cu`). The H32 rotation is applied once per row by
`kernels/hadamard32.cu` (fp32 xor butterflies at strides 1..16, × fp32(1/√32), one bf16
rounding — the one op order the host oracle and the Python reference share): the decode step
rotates the token rows into a side buffer the routed slots read and the routed SwiGLU rows in
place; the prefill chains rotate the gathered routed rows and the SwiGLU rows the same way.
Nothing else in the model changes; the draft layer is requantized at load exactly as for the
Int4/Int8 checkpoint.

**Memory plan, world 4 (rank 0):** model weights resident 93.68 GiB per rank (the Int4/Int8
checkpoint: 96.32), 52.6 KiB per context token with the fp8 latent cache. The template
`deploy/cluster_glm-5.3_nf4i8_w4.example.json` carries a 272K-token fp8 K/V pool at eight
slots, MTP depth 1 (plan 110.43 GiB; the 280K shape, plan 110.85, boots only on a freshly
settled node — the process holds ~3.5 GiB beyond the plan at listening, so the 4 GiB check
is the floor, not a margin). Boot 324 s the first time (the resident image is
written), 30 s after.

**Gates (2026-10-08/09):** the packed GEMV, GEMM and MoE-layer tests against the host oracles
(0 ulps at every row count, decode and prefill), the synthetic-checkpoint forward/decode/TP/
engine gates, the binding of the landed checkpoint (every tensor), op streams identical
across the four ranks; on the real weights the teacher-forced score of
`benchmarks/teacher_text_hard.txt` (7,308 tokens): mean NLL 2.3276 nat/token (perplexity
10.25, top-1 50.3 %) against the Int4/Int8 checkpoint's 2.3295 (10.27, 50.3 %) through the
same kernels — a 0.0019 nat difference inside its 0.0043 standard error; gsm8k 59/60,
HumanEval 40/40, schema extraction 30/30 at `reasoning_effort` low (the Int4/Int8 record's
numbers exactly); MTP acceptance 76–97 % by class at 1.76–1.97 tokens per pass.

**Speed on the four nodes (`docs/benchmarks.md` has the rows; the Int4/Int8 figures here are
the same binary's A/B at the same fp8 K/V form):** decode steps the same as
the Int4/Int8 checkpoint's at one request and at every context bucket (0 / 32K / 64K / 128K:
64.3 / 76.4 / 81.0 / 90.2 ms per pass; 256K 109.2), and at eight requests (sixteen rows: 246–260 ms per step against the Int4/Int8
checkpoint's 245–262, after the rotation was hoisted out of the slot kernels); cold prefill
3.87 / 17.9 / 76.5 s at 2K / 8K / 32K prompts; C1 27.1–30.6 engine tokens/s, C8 52.6–56.4
wall tokens/s over the five prompt classes. Full-size evals: HumanEval 157/164, GSM8K
294/300, schema extraction 100/100 (the Int4/Int8 rows: 159–160, 292–293, 100).

**The draft head in block FP8** (`engine.mtp_head: fp8`, opt-in): the draft's GEMV 1.36 → 1.05 ms
per pass (0.5 % of the step), acceptance unchanged, transcripts identical (the verifier keeps
the main head), +0.22 GiB per rank; the template leaves it off in favour of the context.

**Where the pass goes and what moves it (nsys, one request, MTP depth 1, 12.8 GB read per rank
per pass, a 55 ms floor at the 233.6 GB/s device ceiling, 63–66 ms measured):** routed experts
25 ms at the ceiling, o_proj 7.5, the collectives 7.9 (162 rounds), bf12 GEMVs ~7, attention
compute ~4; the tensors the boundary prefetcher hides (q_a/kv_a, q_b, kv_b, the indexers,
~1.8 GB a pass) cost nothing to read, so narrowing them buys nothing (an int8 read of kv_b was
built, measured slower, and removed). Measured levers: MTP depth 2 +9–16 % single stream on
code/JSON/prose/math (`--mtp-depth 2 --max-concurrency 5`; a loss at C4 and C8 on the
eight-slot recipe), the FP8 draft head 0.5 % (fusing the boundary norms with their residual
adds was built, measured level — a graph node costs ~0.45 µs and the one-block-per-row form
lost the add's parallelism — and removed), and
`engine.attention_weights: int4` as the attention-bytes instrument: −7.5 % per pass for +0.030
nat on the hard text, with o_proj the exposed read.
