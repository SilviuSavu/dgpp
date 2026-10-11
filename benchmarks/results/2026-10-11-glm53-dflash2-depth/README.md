# Full GLM-5.3 + DFlash2 — the verify-depth A/B (2026-10-11)

Target: HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64, four Sparks (w4, 212,992-token
FP8 KV pool, 8 slots, bf12 weights, decode graph). Drafter:
incoai/GLM-5.3-DFlash2 (`engine.dflash_model`, block 8 -> 7 drafts/step,
`dflash_weights checkpoint` + bf12 companions). The lane's standing recipe
before this A/B verified the whole block every step (`dflash_batch_rows` 0);
the pass is ~148 ms with ~75 ms of int4 expert reads at 8 verify rows, and the
per-position acceptance tail (chat p5-p7 at 29/14/0 %) wastes those reads.

## Method

`scripts/serve_load.py` against the live lane (C1, greedy, thinking off,
max_tokens 320, five classes, median of 3 reps), one lane cycle per variant:
config edit -> lane-stop-all -> glm53i8-dgpp-start.sh -> boot verified
(`dbr=`/`mss=` in the rank-0 settings line) -> suite. Variants:
`engine.dflash_batch_rows` 4 / 5 / 6 and `engine.mtp_schedule` (the selector
-confidence depth schedule; exclusive with the rows budget). Baseline = the
standing whole-block recipe measured on the same day before any restart.
Raw suites: `raw/serving/{dbr0,dbr4,dbr5,dbr6,sched}.json`.

## C1 wall per class (tok/s, prefill included)

| variant | prose | code | json | math | chat | mean |
|---|---|---|---|---|---|---|
| dbr 0 (standing) | 26.4 | 29.4 | 27.3 | 30.4 | 14.5 | 25.6 |
| dbr 4 | 26.3 | 26.7 | 26.2 | 25.4 | 22.2 | 25.4 |
| dbr 5 | 26.2 | 26.7 | 26.9 | 26.2 | 22.7 | 25.7 |
| dbr 6 | 30.3 | 31.2 | 31.5 | 30.6 | 22.3 | 29.2 |
| mtp_schedule | 31.4 | 35.5 | 32.3 | 34.1 | 19.9 | 30.6 |

Rep noise observed at +-0.2 tok/s within a variant (three reps per class), so
the class deltas above are outside noise except json dbr0-vs-dbr4/5.

## Reading

- Every depth cap lifts chat ~53 % (14.5 -> 22+): the thin-tail classes pay
  full expert reads for rows 5-8 that almost never survive.
- Caps 4 and 5 lose code/math: the structured classes' p4/p5 acceptance
  (43 % / 29 %) is worth more than the rows it costs. The knee is at five
  verified rows (dbr 6): +14 % mean over the whole block.
- The confidence schedule beats every fixed cap on four of five classes and
  the mean (+19.5 % over the standing recipe; code +21 %, math +12 %):
  it verifies deep exactly where the block is confident (json/math/code) and
  short elsewhere. Chat is the exception (19.9 vs dbr 6's 22.3): the schedule
  still takes too-deep steps on chat's confident-but-thin blocks — the
  `msmin`/`msbase` schedule knobs are the follow-up for a shallower floor.
- Schedule acceptance across the suite (cumulative /metrics): per-position
  accepted [1111, 772, 555, 378, 277, 205, 161] of 1,400 steps — 79 % at p1
  with real mass through p7 on the structured classes.


## The schedule's cost model (the chat fix, second sweep the same day)

The whole sweep above ran `engine.mtp_schedule` with its DEFAULT cost model
(row 8 ms / base 28 ms) — never profiled for this lane (~9.4 ms/row of expert
reads alone). The depth rule is S_i > lambda* x row_ms (base only floors the
warm-up lambda), so a too-cheap row makes deep steps look free: chat's
confident-but-thin blocks verified deep and lost. Re-sweeping the model:

| variant | prose | code | json | math | chat | mean |
|---|---|---|---|---|---|---|
| sched defaults (8/28) | 31.4 | 35.5 | 32.3 | 34.1 | 19.9 | 30.6 |
| sched row 14 | 32.5 | 36.6 | 32.2 | 34.0 | 21.0 | 31.3 |
| sched row 24 | 32.6 | 37.2 | 33.5 | 34.1 | 22.5 | 32.0 |
| sched row 14 + base 90 | 29.8 | 34.2 | 31.0 | 32.5 | 17.3 | 29.0 |
| sched row 32 | 31.4 | 37.2 | 33.8 | 35.2 | 22.9 | 32.1 |
| sched row 32 (pin confirm) | 31.8 | 37.8 | 33.7 | 35.0 | 22.8 | 32.2 |

row 24 fixes chat past even the dbr-6 cap (22.5 vs 22.3) while keeping every
structured class; row 32 plateaus (32.1/32.2 vs 32.0, inside noise, chat best
at 22.9) and wins the pin. Raising base_ms hurts everywhere (it only lowers
the reservation lambda -> more cold speculation). Raw suites:
`schedRow{14,24,14B90,32}.json` + `final-32.json`.

## Outcome

The standing lane config (`deploy/cluster_glm-5.3_int4-int8_w4.json`) ships
`engine.mtp_schedule: true` + `engine.mtp_schedule_row_ms: 32` from 2026-10-11
(the measured-best recipe after both sweeps; confirm run 32.2 tok/s mean).
The dbr-6 fallback row is obsolete — row 32 beats it on chat too.

## Follow-ups

- DONE the same day: the cost-model sweep (row 32) recovered chat to 22.8+
  while lifting every other class; no floor tuning needed.
- The C2-C8 lever is the multi-request block (this suite is C1; concurrent
  requests still serialize on one eight-row block a step).
- Re-run this A/B after the expert-read work (DGPP_MOE_DECODE_GROUPED, the
  upstream GLM-5.3 GPTQ quants) — the knee moves with the pass floor.

## D9b: the 32-row decode batch — the multi-request block (the same evening)

`GlmDsaModel::decode_rows_cap()` 16 -> 32 (the DSA layer's matching [1,16]
validation; the select already grouped rows by eight and its workspace
already bounded 32). Two real fixes on the way:

- The first 3-slot capture died on 24 memset graph nodes: cuBLASLt's
  heuristic handed the new 24-row GEMM shapes a split-K algorithm whose
  workspace zeroing is a memset inside the capture (decode graphs must stay
  kernels-only). The plan builder now prefers workspace-free algorithms
  whenever it runs on a capturing stream; `graph_check` logs any memset
  node's pointer/width for triage. The boot then captured the whole family
  tree: 2/3/4-slot x 8-row blocks with every scheduled-depth variant,
  kernels-only, 1.36 GiB device headroom left on rank 0.
- Unit gates: dsa_test, glm_moe_test, the glm_dsa forward chain and
  glm_dsa_dflash2_test all green on the 32-row build.

### C1-C8 aggregate walls (tok/s; the 16-row baseline above vs the 32-row lane)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose 16-row | 31.6 | 31.9 | 27.7 | 25.4 |
| prose 32-row | 32.2 | 31.7 | **38.8** | 26.9 |
| code 16-row | 33.8 | 36.9 | 32.6 | 30.8 |
| code 32-row | 36.4 | 36.7 | **43.5** | 29.9 |
| json 16-row | 33.7 | 36.5 | 39.3 | 34.3 |
| json 32-row | 32.5 | 35.4 | **49.1** | 33.0 |
| math 16-row | 34.8 | 41.6 | 36.0 | 37.2 |
| math 32-row | 33.7 | 41.8 | **45.5** | 37.0 |
| chat 16-row | 22.8 | 31.8 | 28.5 | 28.8 |
| chat 32-row | 22.4 | 32.1 | **41.0** | 27.9 |

- The exit gate holds everywhere: C4 beats the C2 aggregate in every class
  (a second/third/fourth block in the pass pays), +25-44 % over the 16-row
  C4 walls.
- C2 is flat (the same 2-slot family serves it before and after — the
  sanity anchor) and C1 is level within day-noise (mean 31.4 vs 32.2; the
  scalar path is untouched).
- C8 is flat as expected: eight live slots still take two 4-slot passes;
  the 64-row batch (eight blocks a pass) is the next rung, already inside
  kDecodeRowsMax.
- Acceptance inside the 4-slot passes is deep and healthy: 5.0-5.8
  tok/step/req with real mass through p7 (56-89 %) — the schedule takes
  deeper steps when four blocks amortize the pass.
- Transcripts: batched-vs-scalar prefix divergence matches the documented
  near-tie class (a C2 output diverged from its C1 twin at char 18-41 on
  the 16-row lane too — cross-dispatch, tolerance-equal, pre-existing);
  outputs stay coherent, no degenerate loops, all requests finish clean.

Raw suites: `rows32-c1c2c4c8.json` (and the 16-row baselines
`final-32-c1c2c4c8.json` + `final-32-c2c4c8-baseline-b.json`).

## D9b's last rung: the 64-row batch (the same evening)

`kSelectLayoutRows` 32 -> 64 (the select workspace layout; the groups-of-
eight launcher and everything else already scaled), the layer/model caps
matched, `prefix_cache_gib` 1.0 -> 0.5 for headroom. The boot captured the
whole tree — 2/3/4/6/8-slot x 8-row blocks, kernels-only, 0.70 GiB free on
rank 0 — and eight blocks ride one verify pass:

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 32.1 | 31.0 | 39.5 | **47.8** |
| code | 36.4 | 37.1 | 43.4 | **49.1** |
| json | 32.6 | 34.5 | 49.8 | **52.1** |
| math | 34.5 | 41.3 | 45.0 | **53.6** |
| chat | 22.0 | 31.9 | 41.1 | **48.1** |

C8 now beats C4 in every class (+45-88 % over the 16-row baseline's C8;
the staircase is closed: C1 ~32 -> C8 ~50 aggregate); C1/C2/C4 level with
the 32-row run; per-request acceptance at C8 is 2.2-2.7 tok/step (the
eight-block pass amortizes the step). Unit gates green on the 64-row build
(dsa, glm_moe, the forward chain, glm_dsa_dflash2, dflash2 kernels).
Raw: `rows64-c1c2c4c8.json`.

## The upstream merge (the same evening)

Merged origin/master (38d3af69: the GLM-5.3 GPTQ quants, llama.cpp timings,
HTTP binding setup) into the line — two real conflicts: CHANGELOG (both
sides' entries kept) and moe_layer's grouped-expert chain (upstream's
Mixed346/A8 branches under our grouped-decode `views` parameter threading;
upstream's structure, our rename — behavior-identical for prefill, which
passes `d_views_prefill_` as `views`). Post-merge: full rebuild clean, the
unit gates green except portability_test's
`test_localhost_and_deployment_http_override`, which fails BYTE-IDENTICALLY
on pristine origin/master (their test expects a localhost config to resolve
127.0.0.1 while their own DEFAULTS pin DGPP_HTTP_BIND=0.0.0.0) — pre-existing
upstream breakage, left as-is and documented here. The merged lane boots the
64-row world unchanged (5 families, kernels-only, 0.75 GiB free) and the
C1/C4 smoke is level (prose 31.9/37.0, code 35.7/43.7).
