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
