# GLM-5.3 quant A/B — NF4I8 GPTQ H32 g128 vs the Int4/Int8Mix RTN lane (2026-10-12)

Question: does HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128 (4.125 bits/weight,
GPTQ-calibrated, NF4 codebook, block-32 Hadamard) lose quality or speed
against the production HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64 lane?

## Setup

Identical recipe on both lanes (deploy/cluster_glm-5.3_nf4i8_w4.json is the
int8 config with only the model swapped): world 4, dflash2 drafter
incoai/GLM-5.3-DFlash2, mtp_schedule on, msrow 32, decode rows 64, kv 212992
fp8, prefix cache 0.5 GiB, decode graphs on. Same binary (68a5d063).
Suite: scripts/serve_load.py --concurrency 1,2,4,8 --repeat 3
--max-tokens 320 --classes prose,code,json,math,chat (nf4i8-suite.json).
Int8 reference: the 64-row record of 2026-10-11-glm53-dflash2-depth.

## Throughput (aggregate wall tok/s, prefill included)

| class | int8 C1 | nf4i8 C1 | int8 C8 | nf4i8 C8 |
|---|---|---|---|---|
| prose | 32.1 | **28.0** | 47.8 | 47.9 |
| code  | 36.4 | 36.8 | 49.1 | 48.5 |
| json  | 32.6 | 32.6 | 52.1 | 51.0 |
| math  | 34.5 | 34.5 | 53.6 | 53.5 |
| chat  | 22.0 | 23.3 | 48.1 | 48.1 |

Class means: C1 -1.5%, C8 -0.7% — parity within run noise, with one real
regression: prose C1 28.1 tok/s over 5 confirmation repeats (int8 32.1),
decomposing into TTFT 573 vs 472 ms (+21%, the H32/codebook prefill path)
and 34.0 vs 32.0 ms/step (+6% at the latency-bound extreme).

## Acceptance and health

Draft acceptance p1 ~72%, p2 ~60%, p3 ~50%, p4/p5 100% — the same
acceptance class as the int8 lane (target fidelity inside the drafter's
window). Boot ledger: device free 102.7 GiB vs ~100.2 on int8 — the
-2.6 GiB/rank payload claim confirmed.

## Quality (external, verified 2026-10-11)

The publisher's teacher-forced tables (HF cards, byte-identical to the
in-tree docs/model_cards/): KL to BF16 0.0278 vs int8's 0.0358 (-22%),
top-token agreement 94.2% vs 93.6%, NLL-change interval spanning zero,
perplexity 10.25 vs 10.27 on held-out text; gsm8k 59/60, HumanEval 40/40,
schema 30/30 identical.

## Teacher-forced A/B on the fabric: not measured, and why

glm_dsa_forward_check --teacher-file at 7331 tokens (world 4) wedged
mid-forward on every variant tried (staged/bulk, 90 MB/8 MB slices,
paced/unpaced, pinned/unpinned; 6 runs): bulk-consumer deadlines then
lane transport retries exhausted, wc status=12 vendor_err=129, a
different peer each time. The publisher's fidelity tables come from a
python FP32-represented-weight harness, not this app; the engine never
folds a whole unwindowed prefill, which is why serving is immune.
Fixed on the way and committed (68a5d063): bulk folds sliced at 8 MB
(bitwise-exact, bus_test), the diagnostic now pins like the engine, and
dgpp-cluster got DGPP_BOOT_WAIT_S for cold resident-image builds
(first NF4I8 boot: 2192s; subsequent boots warm).

## Verdict

NF4I8 is quality-parity-or-better at throughput parity (-0.7 to -1.5%
aggregate) with +2.6 GiB/rank headroom. The single prose-C1 regression
(-12%) is real. Not a must-switch; the headroom matters only if the KV
pool or prefix cache grows. Mixed346 GPTQ A8 (staged, farm-verified,
lane config ready) remains the untested candidate — its int8 expert
activations promise a decode-side win (kernel-level -16% expert-block
median) at one more fidelity caveat (unmeasured task accuracy,
2x one-ulp input sensitivity).

## Addendum (same day): Mixed346 GPTQ H32-A8 g128 — the C4-C8 win

Same procedure, deploy/cluster_glm-5.3_mixed346_w4.json (identical recipe,
model swapped), cold boot 2795s (image 98.1 GiB, boots warm after), suite
in m346-suite.json.

| class | int8 C1/C8 | nf4i8 C1/C8 | m346 C1/C8 |
|---|---|---|---|
| prose | 32.1 / 47.8 | 28.0 / 47.9 | 30.7 / 50.4 |
| code  | 36.4 / 49.1 | 36.8 / 48.5 | 36.2 / 49.5 |
| json  | 32.6 / 52.1 | 32.6 / 51.0 | 33.0 / 55.0 |
| math  | 34.5 / 53.6 | 34.5 / 53.5 | 33.3 / 57.1 |
| chat  | 22.0 / 48.1 | 23.3 / 48.1 | 24.4 / 51.4 |

Class-mean deltas vs int8: C1 +0.0%, C2 +5.0%, C4 +3.2%, C8 +5.1%.
The throughput-bound tiers win everywhere; C1 is parity (prose C1 dips
to 30.7 — half of NF4I8's dip; every other C1 class is at or above int8).

Draft acceptance is the surprise: p1 75%, p2 75%, p3 69%, p4/p5 100%
(int8: 72/60/50/100/100) — the A8 target agrees with the unchanged
DFlash2 drafter MORE often, and the deeper acceptance compounds the
kernel-side speedup into the C4-C8 wins. Transcripts spot-checked
(prose, math): coherent reasoning, correct problem setup, no drift.
Boot ledger: device free 98.4 GiB after globals — the least headroom of
the three lanes (NF4I8 102.7, int8 ~100.2), consistent with the A8
activation buffers.

Remaining caveat, unchanged from the research phase: the publisher
shipped no generated-task evals for this quant (NF4I8's gsm8k/HumanEval/
schema parity does not transfer automatically), and A8 doubles one-ulp
input sensitivity at the kernel level. The higher measured acceptance
and coherent transcripts are the evidence we have; a task-level eval
suite on the lane would close it.

## Three-way verdict

- Stay int8 for: single-stream latency-identical serving, max headroom
  certainty, known task behavior.
- NF4I8: quality-parity-or-better, throughput parity, +2.6 GiB/rank —
  switch only if KV/prefix pools grow into the space.
- Mixed346: +5% at C4-C8 with higher acceptance, C1 parity — the
  serving-side winner IF a task-level eval confirms no generated-quality
  regression. Recommended next gate: 50-100 prompt task eval
  (gsm8k-style + code + schema) on the m346 lane vs int8 transcripts.

## Task-level eval gate (the switch condition): PASSED — winner = Mixed346

60 gradeable tasks (30 gsm8k-style math with computed keys, 15 code with
executable asserts, 15 strict-JSON schema checks), greedy, max_tokens 2048,
identical prompts per lane; scripts/task_eval.py (grader self-tested: 15
reference code solutions + math/schema probes green before any model run).

| lane | math | code | schema | total |
|---|---|---|---|---|
| int8 RTN (reference) | 30/30 | 15/15 | 15/15 | **60/60** |
| Mixed346 A8 | 30/30 | 15/15 | 15/15 | **60/60** |

Perfect parity. Two key bugs found and fixed during grading were authoring
errors (round-down vs round; a float truncation), both cases where the
model was right and the key was wrong. One task (code_flatten) needed
max_tokens 2048 — the model reasons at length before emitting code; at
1024 it burned the whole budget on reasoning with empty content.

Winner: Mixed346 — task quality at parity, serving +5% C2/C8, +3.2% C4,
higher draft acceptance. The m346 lane stays up as the new default
(swap back anytime: ./glm53i8-dgpp-start.sh).

## llama-benchy on the winner (eugr/llama-benchy 0.4.0, PyPI)

OpenAI-endpoint pp/tg benchmark, MTP-aware, 3 runs each, book corpus
(Gutenberg), tokenizer from the local m346 checkpoint. benchy/*.json.

| test | pp tok/s | tg tok/s | e2e TTFT ms |
|---|---|---|---|
| pp512/tg128/c1 | 461.4 ± 3.9 | 30.3 ± 1.5 | 1112 |
| pp512/tg256/c1 | 465.0 | 23.5 | 1104 |
| pp2048/tg128/c1 | 537.0 | 25.1 | 3817 |
| pp2048/tg256/c1 | 541.6 | 24.0 | 3784 |
| pp2048/tg256/c8 | 491.3 | 23.3 (per stream) | 18203 |

Notes: first invocation 404'd — the endpoint needs --base-url .../v1
(the tool appends /chat/completions itself). At c8 the tool logs "No
complete token_ids in response, using stream usage token count": the
engine streams usage but not token ids; throughput accounting unaffected.

## tool-eval-bench on the winner (SeraphimSerapis/tool-eval-bench)

69 deterministic scenarios, mock tools, pass/partial/fail scoring,
parallel 4. tool-eval/results.json + report md.

Score 126/138 points (91%), rating ★★★★★ Excellent, deployability 66,
median turn 19.4 s, 516,887 tokens total. Categories: A-F and H,J,L,N
perfect (tool selection, parameter precision, chains, restraint, error
recovery, localization); I Context & State 85%, K Safety 96%, M
Autonomous Planning 67% (1 fail), O Structured Output 50% (worst — raw
JSON-schema compliance; this is the engine's structured-output/grammar
path, quant-independent, and matches the schema-drift seen in category O
across serving stacks per the tool's own docs). Safety gate: 1 warning
(TC-51 batched a dependent tool call instead of sequencing).

## Summary

1. NF4I8: parity quality/throughput, prose-C1 regression — not adopted.
2. Mixed346: +3-5% serving at C2-C8, acceptance up, task eval 60/60,
   tool-eval Excellent, llama-benchy baseline recorded — adopted as the
   default lane.
3. Open engine follow-ups: structured-output compliance (cat O) and
   streaming token_ids at concurrency.
