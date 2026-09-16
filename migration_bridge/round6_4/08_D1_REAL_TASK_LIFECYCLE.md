# D1 real-task lifecycle qualification

State: **A3_VERIFIED / D1_REAL_TASK_LIFECYCLE_PASS**.
Production code unchanged; original batch1/context512, production overlap.
Metadata task tracing is enabled; this is not a performance measurement.

Six frozen S6 prompts × 64 tokens × six cycles produce 36 requests. Every
trajectory equals the corresponding observer-OFF D1 runtime reference. The
trace covers all 2,340 physical forwards, including lookahead.

| Measured event | Count |
|---|---:|
| Actual CPU tasks with selected routes | 3,042 |
| Of those, deferred tasks | 258 |
| Worker synchronization completions | 4,680 |
| CPU experts exercised | 8 / 8 |
| Unconsumed pending after terminal | 0 |

The ≥1000 CPU-expert exposure gate uses positive-route task submissions, not
generated token count or empty/padding submissions. Each physical forward has
ordered L7/L8 transitions validated by the retained transition audit. This
does **not** claim 1000 deferred-only tasks: the measured deferred count is258.

Summed RSS for owned processes: start31,430,770,688 bytes,
peak/end31,953,891,328 bytes. Most growth occurs during the first two requests;
an additional8192 bytes appears by request10. Samples10–36 are identical.
This run shows an observed plateau, not proof of zero memory leak. Summed RSS
can double-count shared pages and is not unique physical RAM usage. Separate
two-campaign stability/HBM measurements remain required later in Round6.4.

Server guard ended by planned project-owned shutdown, lifecycle PLANNED_STOP.
No crash, request error, output mismatch, or non-finite output was observed.

Evidence: `evidence/r64-d1-lifecycle-v1/lifecycle-report.json`;
SHA256 `ed873dd782053d8f9006e58bd4401af425214525b43fbb2b57d0bad32d184e35`.
Actual source provenance is in the same evidence directory. Raw trace remains
at `/workspace/results/r64-d1-lifecycle-v1/deferred-events.jsonl` on A3.
Trace SHA256: `d6f295f873eea8585603a354e6da6f067bcc985cb0ac769ab1cffb4b48f1530f`.
An identical local archive is retained outside tracked Git content.

WP2 runtime/oracle/task-lifecycle gates are now closed. Batch2/4 engineering
qualification is running; primary quality and speedup are not yet qualified.
