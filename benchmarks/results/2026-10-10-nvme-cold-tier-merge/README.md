# The NVMe cold tier on the merged dflash2-glm line (2026-10-10)

PR #53 (issue #26) merged into the dflash2-glm branch at bf28b39 and
validated on the four-Spark gx10 fabric (10.28.1.1–.4), GLM-5.3
Int4-Int8Mix RTN g64 (`glm_moe_dsa`), KV fp8, MTP depth 2, the tier at
16 GiB per rank (`deploy/cluster_glm-5.3_int4-int8_w4_nvme.json`, a
5-slot arena so eviction is quick — this model's snapshot is 12 304
bytes, so `prefix_cache_gib` 0.00006).

| Step | Cached tokens | TTFT |
| --- | --- | --- |
| document A + Q0, cold | 0 | 61 020 ms |
| document A + Q1, changed question (memory) | 22 528 | 7 085 ms |
| three other documents, each cold | 0 | ~61 000 ms |
| document A + Q1 again, **restored from disk** | 25 148 | **156 ms** |
| document A + Q0 again (memory) | 25 148 | 1 341 ms |

`serve_nvme_cache_check.py` PASS: the in-memory and restored transcripts
identical, 8 spills and 1 restore committed with no failures (10.6 GiB
written, 1.36 GiB read), the op streams byte-identical across the four
ranks (`md5 42676581…` on every `serve_rank*.ops`). The cold prefill is
slower than the flash model's 18.8 s because the Int4-Int8Mix recipe
prefills this document at ~410 tok/s; the restore is the same order as
the PR's (156 ms against its 407 ms on flash).

Gates on the merged tree: `unit_tests` 322/322 (disk index, config
block; the three arena cases need an idle GPU — they OOM beside a
serving world), `scheduler_test` 88/88 (the seven nvmeCache cases),
`fabric_serve_test` 10/10 over real sockets (ok 10: spill, evict,
restore, attach through the journal's return path),
`qwen_decode_test --fixture` including the tier block (spill/restore
bitwise, attach parity, a corrupted page detected), the Python resolver
20/20. Newer families joined the tier's contract in the merge:
`Dsv4StatePool`/`Qwen35KvPool` planes and block identities, and
`Dsv4Model`/`Qwen35Model` `kv_block_bytes_static` (the Qwen35 one
includes the DFlash2 drafter's own planes over the shared block table).
