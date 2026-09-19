# Performance watchdog failure and versioned retry

2026-09-19, authorized .22. v2 is FAILED and must not be overwritten or merged
into v3. Static0-0 collected short23.4411s/medium49.6810s; these single-cell
numbers are incomplete, not a comparative performance conclusion.

Dynamic0-1 warmup and short request returnedHTTP200. During the medium256/64
request, layerwise preparation advanced from layer0 at18:44:24 to layer34 at
18:50:33, approximately8–12s weight preparation per layer. At18:50:35 scheduler
watchdog300 triggered SIGQUIT; the client then reported RemoteDisconnected.
The owned container was stopped by its controller, OOMKilled=false. No other
container was stopped. The timeout is a forward-progress watchdog, not a
performance acceptance bound. We do not remove this cost from the timed region.

Official cross-check: SGLang server arguments document watchdog default300 and
termination for overlong forward batches:
https://docs.sglang.io/docs/advanced_features/server_arguments
(checked2026-09-19). The actual runtime log explicitly confirms300. Both frozen
performance configs omit the flag; the already successful heldout-v3 explicitly
uses1800. This identifies an orchestration mismatch; the same performance
request must still reproduce successfully before elimination is claimed.

## Minimal retry

New adapter run_qwen_dynamic_perf_retry.py creates qwen-dynamic-perf-v3, with
Static AND Dynamic watchdog1800, still bounded. All12cells restart; no imported
partial timings. Same model/code/Q32/C64/threshold256, frozen inputs64/32 and
256/64,3ABBAblocks,hook-free,timed start/end, resource admission and monitoring.
The original frozen source hashes remain mandatory; explicit transformations
only change watchdog, output identity, entrypoints and approved heldout-v3
release path. Effective configs and runner hashes are frozen under
qwen-dynamic-perf-retry-v3 before execution; local commit acknowledged remotely.
Round9 download remains PAUSED for entire retry. Ordinary historical failure
is retained, no quality gate or underlying implementation changed.

Two new regression tests were observed RED, then GREEN: bounded watchdog is
added without mutation of inputs/other config, conflicting timeout rejected.
Local tooling suite31testsPASS. Actual remote frozen controller/client hashes
verified and transformed sources compiled successfully. Full pytest unavailable
locally (No module named pytest); no full native/device regression claimed.

Failure archive copied to local raw/ and retained on .22:
qwen-dynamic-perf-v2-failure.tar.gz
SHA256 db908e78a326f8c9c0f0ff090089cddd58784eaf83e5e6e051426b1682f886a2.
Device retry outcome PENDING. Never describe raising timeout as a speedup.
