# Performance observer parity — A3 verified

2026-09-16. This is an execution prerequisite, **not a speedup result**.

Frozen workload: `performance-fast.json`, 192 input IDs / 256 generated IDs,
batch1. Four independent serving processes, in order:

| Profile | Reference with metadata observer | Observer-OFF streaming repeats | Exact within profile |
|---|---:|---:|---:|
| D1_NODEFER | 1 | 5 | 5/5 |
| D1 | 1 | 5 | 5/5 |

All 12 requests completed. The same D1 workload recorded **16 deferred CPU
route hits** before formal timing, so exposure is nonzero. It is sparse and
does not establish performance across arbitrary prompts or heavier deferral.
OFF and ON reference token IDs also happen to agree here; that is diagnostic,
not the cross-mode acceptance requirement.

Report: `evidence/r64-minimal-perf-parity-v1/parity-report.json`.
SHA256: `be8708b9c2bd0db1b9378b6c72c65b5885b6d65ad7cbe26642f85e9087f220d0`.
Raw requests, process logs, resolved commands, source provenance and metadata
traces are retained alongside the report. No full tensor capture was enabled.

Shutdown terminology: the harness records `PLANNED_STOP` after sending its
owned guard SIGTERM. Guard records can show `CANCELLED`, SIGTERM/SIGKILL and
returncode -9 during bounded teardown; this is **not an EXIT_ZERO graceful
server shutdown claim**. All requests finished before teardown, and the
subreaper's `OWNED_DESCENDANTS_REAPED` result is required. No other process
group is targeted.

The next gate is the already-frozen three-block ABBA campaign, not additional
workload selection. At this document's creation that campaign is RUNNING;
`SPEEDUP_VERIFIED` has **not** been established.

Offline tooling verification at the same checkpoint: `python -m unittest
discover -s migration_bridge/round6_4/tools -p 'test_*.py'` — **70 tests PASS**.
This is LOCAL_VERIFIED, distinct from the A3 parity result above. An attempted
pytest invocation found pytest absent from the analysis venv; no tests ran in
that invocation and no dependency was installed. The suite uses unittest.
