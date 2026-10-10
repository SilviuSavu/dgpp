# GLM-5.3-Flash + DFlash2 — first deployment (2026-10-10)

Target: HawkBearPig/GLM-5.3-Flash-NVFP4-FP8, four Sparks (w4, bf16 latent
786K, bf12+bf16, 20 GiB prefix). Drafter: incoai/GLM-5.3-Flash-DFlash2
(5 bidirectional layers, taps [5,14,24,33,42] captured through the mHC
stream mean at each tap boundary, block 8 -> 7 drafts, selector top-16
rank-256). engine.dflash_weights checkpoint; the drafter weighs 0.83 GiB
per rank with 0.47 GiB companions + 0.14 GiB planes/scratch.

## Gates

- Exactness (greedy, 4 classes, 256 tokens, reasoning_effort low): the
  dflash transcripts are IDENTICAL to the plain path 4/4 at the shipped
  recipe (engine.dflash_batch_rows 4 — the verify caps at 4 rows).
  The whole-block (8-row) verify reads DIFFERENT from plain: this
  family's decode kernels are not row-invariant past 4 rows (the old
  kSpecRows=4 GEMV bound; the Qwen lane's 2026-10-05 row-invariance
  campaign never landed here — PLAN's "GLM-Flash row expansion" item).
  MTP-1 (2-row verify) is identical to plain, which isolates the cause.
- Acceptance at depth 3: p1 94 % / p2 88 % / p3 79 %, 3.59 tok/step at
  43.1 ms/step (~83 tok/s instantaneous C1) — the recipe's healthy band
  (the vLLM 2x-Spark lane measured 74 % aggregate; ~15 % would mean the
  tap capture or the mHC contraction is wrong).
- End-to-end wall per class (256 tokens, single stream): dflash@4 vs
  plain 1.35x (chat) / 1.52x (code) / 2.00x (json) / 1.86x (math); vs
  the MTP-1 template 1.00x / 1.05x / 1.31x / 1.26x — the §7 exit gate
  (beat the best native-MTP configuration) holds on code/json/math,
  level on chat.
- Sampled requests serve (the ratio rule with the armed proposals).

## Files

ab_plain.json / ab_mtp1.json / ab_d4.json / ab_final.json — the gate
transcripts and walls per class.

## Depth 7 (the same day)

The whole-block divergence was dispatch, not kernels: the GEMV chunks and
the fp8 core are row-invariant through eight rows by design, and the Flash
model routed rows 5-8 into cuBLASLt/MMA (`set_decode_rows(4)`,
`gemm_mma_from_rows 5`). With the decode bounds past the whole batch:

- whole block (`--dflash-batch-rows 0`): transcripts IDENTICAL 4/4,
  5.17 tok/step at 62.4 ms, accept p1 93 / p2 80 / p3 64 / p4 58 /
  p5 52 / p6 45 / p7 34
- the four-row recipe stays the shipped default (same ~83 tok/s
  instantaneous; its shorter step wins chat 4.8 s vs 6.4 s and code
  4.2 vs 4.5, the block wins json 2.8 vs 3.2 and math 3.4 vs 3.5)
- ab_block.json / ab_plain2.json / ab_final2.json: the depth-7 gates

## Serving suite (2026-10-10, the shipped dbr 4 lane)

`scripts/serve_load.py` five classes x 3 reps, max_tokens 320, greedy,
plus `scripts/serve_prefill_probe.py` cold prefill (3 probes per length)
against the live deployment — raw/serving/{dflash2-dbr4.json,prefill.json}.

- C1 engine decode: median 66.9 tok/s (49.9-76.5 across class windows) —
  the plain 4-node template's 56.8-62.2 (same kv/slots)
- C1 wall per class: prose 56.9, chat 55.8 (level with plain's 54.8-59.7);
  code 73.2, json 74.6, math 73.7 (+22-33%)
- C2/C4/C8 walls stay 53.5-81.4: the whole 8-row batch is one request's
  block, so concurrent requests serialize across steps instead of sharing
  decode rows — plain reaches 79.5-117.7 by interleaving its 4 slots in
  one batch. The multi-request block (several requests' rows in one
  recorded block draft) is the lane's concurrency follow-up.
- Cold prefill 1.408 / 5.818 / 24.215 s at ~2K/~8K/~32K (plain:
  1.182 / 5.075 / 21.843): the drafter's tap capture and plane stores
  ride every prefill chunk (~16-19%).
