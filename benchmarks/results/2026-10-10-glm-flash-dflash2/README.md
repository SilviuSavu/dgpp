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
