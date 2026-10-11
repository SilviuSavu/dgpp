# Completion response timings compatibility — 2026-10-08

PR #94 adds a top-level `timings` object for timing-aware proxies such as
llama-swap. The follow-up uses llama.cpp revision
[`9c2e0e491a822adae1f0b1c831adb4160057d24f`](https://github.com/ggml-org/llama.cpp/tree/9c2e0e491a822adae1f0b1c831adb4160057d24f)
as the reference for token counters, rate arithmetic, and final chat stream
placement. The [API contract](../../docs/openai-compatibility.md#completion-timings-dgpp-extension)
describes DGPP's timing boundaries and multiple-choice aggregation.

## Findings and changes

The original PR counted the full prompt in `timings.prompt_n`, used arrival
to first token for prefill time, counted the prefill pick in decode throughput,
and emitted stream timings only with `stream_options.include_usage: true`.
That gave cache hits a different prefill-rate meaning from llama.cpp and left
ordinary streams without timing data.

The follow-up:

- Counts only uncached prompt tokens in response prefill rates.
- Uses choice 0's admission-to-first-token interval, excluding its initial queue
  wait and including waits between resumed prefill chunks.
- Counts all generated tokens in `predicted_n`, while decode rates exclude one
  prefill pick per choice that generated any tokens. A one-token completion
  has zero decode throughput. The generated-token window has llama.cpp's
  one-microsecond minimum.
- Appends timings to the final usage event when enabled, otherwise to the last
  choice's terminal event. Existing usage opt-in, chunk types and choice lifecycle
  remain intact on both completion routes.
- Records group timing state separately and attaches the object after existing
  response builders have serialized their original fields.
- Merges current master into the PR, preserving both sides of the changelog
  conflict and the contributor's original commits.

## Existing metric contracts

No existing metric is redefined. `usage.prompt_tokens` includes cached tokens;
completion usage includes prefill picks. JSON scheduler counters still distinguish
total, computed and cached prompt work. TTFT still includes the queue; queue,
prefill, decode and TPOT histograms retain their existing windows. Engine execution
time and per-request elapsed time remain distinct.

A source comparison against master `f7394108445afa8889409a5f078e5aa3fe5f9608`
confirmed that `append_usage`, all three usage-bearing response builders,
`route_metrics`, `route_metrics_prometheus`, `on_admit`, `on_retire`, and
`observe_token_pass` are identical. Scheduler, throughput-log and Prometheus
implementation files are unchanged. The token callback adds response-specific
state alongside its existing first-token measurements.

## Validation

The four `serve_timings_` host cases cover cached chat and legacy requests,
unchanged JSON/Prometheus/usage counters, queued and resumed prefill with
unchanged latency histogram definitions, both streaming routes with usage
omitted/false/true, six choices sharing four slots, one-token completions, and
three-token speculative passes. The socket helper reads complete HTTP framing
so responses may span multiple reads without adding idle waits.

Validation on the merged branch with the follow-up applied:

- `cmake --preset ci`: configured with `DGPP_WERROR=ON`.
- Rebuilt `serve_test`, `unit_tests`, `http_server_test`, `fabric_serve_test`,
  `scheduler_test`, `shutdown_watchdog_test`, `engine_watchdog_test`,
  `roster_check`, `dgpp_serve_app`, `qwen_forward_test` and
  `qwen_yarn_fixture_test` successfully before running the host suite.
- `serve_test`: **101 cases passed**, including the four new timing cases.
- `ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure`:
  **29/29 passed**, including the existing scheduler, metric, script and
  server-startup checks (217.28 seconds).
- `git diff --check` and the source comparisons described above passed.

CMake reported that clang-format is unavailable; touched code was reviewed
against the repository's existing formatting.

This is a response-format and timing-accounting change on the HTTP service.
It does not change scheduler decisions, generation, model kernels or the
multi-rank journal. Validation uses host fakes and the server's startup checks;
no live model/fabric benchmark is claimed.
