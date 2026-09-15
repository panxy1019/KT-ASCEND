# QD2 defer1 capture audit incident

2026-09-15 UTC. State: BLOCKED_CAPTURE_AUDIT; not a numerical rejection.

All8192 requested positions were collected. The original unmodified final
audit failed at `run_qualification_capture.py:157` with
`UNEXPECTED_COMPLETE_FORWARD`. Client supervisor recorded ERROR_LOG, not PASS.
There are16386 complete capture files rather than the expected16384.

Read-only inspection reproduced the first failure at Pass00000.pt, PID284588,
forward sequence0, request `HEALTH_CHECK_1789464331.9784992`. This ID is outside
the8192 qualification request IDs; provenance differing keys are empty.
This health-check timestamp precedes the client's start1789464354.3829384.
Only the first offending capture was inspected by this diagnostic; it does
not yet establish that every other capture is valid or explain the second
extra file. No cross-mode numerical outputs were compared.

The exact server guardian284411 was verified from its full command and
signalled TERM. It cleaned only its own child group284416 and recorded
CANCELLED, SIGTERM/SIGKILL, end1789470849.9046605. This was deliberate cleanup
after client failure. Other workloads were not targeted.

Evidence remains under `/workspace/results/r62-qd2-defer1`,
`r62-qd2-defer1-server/tensors`, `r62-client-qd2-defer1`,
`r62-guard-qd2-defer1`, and `r62-qd2-defer1-incident-inspection` on A3.
Diagnostic source: `tools/inspect_unexpected_forward.py` (CPU-only).

Next: account for ALL captured forwards without deleting or filtering evidence,
inspect startup health-check/capture session boundaries and the frozen protocol.
Any instrumentation correction requires explicit recorded provenance and tests;
do not relabel this failed run PASS or silently reconstruct its missing final
manifest. Candidate freeze, D1 runtime, HD2 and quality remain NOT_RUN.

## Full inventory in progress and boundary correction

CPU-only supervisor `r62-qd2-defer1-full-incident-audit` now runs
`tools/audit_incident_forward_inventory.py` (deadline3600s, OMP_NUM_THREADS=1).
It retains a file SHA inventory for every capture, checks all recorded tensor
hashes/provenance/finite logits, and independently checks request target/lookahead
inputs, ownership and terminal transitions. Every non-qualification forward is
reported, not removed. Output `inventory.json` is FORENSICS ONLY and cannot
replace the original failed qualification manifest. Await its complete result.

CODE_INSPECTED: child `http_server.py:481–540` aliases `/health` and
`/health_generate`; `environ.py:468` defaults health generation to true.
An idle health request creates `HEALTH_CHECK_<timestamp>`, input `[0]`, one
generated token. Server log records GET /health at09:25:33, preceding formal
client start09:25:54. `/model_info` at line572 returns metadata without invoking
generation; future startup readiness must use ready log plus this endpoint,
not either generating health endpoint.

The frozen CONTRACT_V2_SPEC explicitly invalidates capture failures and allows
only documented infrastructure retry with unchanged inputs/rules. Accordingly
the old run remains INVALID; no deleting startup files, reconstructing a PASS
manifest, selecting prompts or numerical fitting is authorized by this audit.

Minimal local tool correction: require the capture directory to exist and contain
no completed or incomplete forwards BEFORE the first sweep request. Final
full-session audit remains strict, so later contamination still fails closed.
Tests: local63 tests,56 passed,7 torch-dependent skips; new boundary tests4/4.
No production code changed. This new tool is NOT yet deployed to A3 or used for
a fresh model run. A future infrastructure retry must freeze new tool/provenance
hashes and new output directories first, keep all8192 positions/history, and
validate smoke/boundary behavior before the full replay.

## Completed full inventory / infrastructure retry decision

A3 CPU-only inventory exited0, started1789471041.226895,
ended1789471376.3252609. All16386 captures accounted for;16384 qualification
forwards checked, failures=[], pair_failures=[], incomplete_files=[], contiguous
single-process sequence true. Two outside forwards: sequence0 input[0], position[0],
and sequence1 input[185], position[1], both request
HEALTH_CHECK_1789464331.9784992. They are the startup health generation and its
lookahead, not qualification requests. No numerical cross-mode comparison.
Exact raw inventory SHA256:
`ad713b711402d06fdd2df614afbb628a9fefa2c6c26009c8025aeb8fc0d6d195`.
Portable evidence: `evidence/qd2-capture-incident/inventory.json.gz`,
`audit-result.json`, `client-failure.json`.

Decision: INFRASTRUCTURE_ONLY_RETRY, permitted by preregistered protocol.
Original run permanently INVALID. Preserve complete original captures, baseline
history and successful defer0. Retry the ENTIRE defer1 corpus in a new process
and directory, never resume/select/merge positions. Same128 prompts ×64 positions,
same history/placement/GGUF/model/source, numerical rules and21600s deadlines.
The only code change is early rejection by the client test tool; model capture
and production sources remain unchanged. Non-generating readiness does not
change model arithmetic. `prepare_capture_retry_metadata.py` rehashes all old
artifacts and rejects changes except the explicitly updated collector test tool;
records additional helper hashes and incident provenance. Old metadata untouched.

Planned fresh smoke root `r62-boundary-smoke-defer1`; planned full retry roots
`r62-qd2-defer1-retry1`, `r62-qd2-defer1-retry1-server`, supervisors
`r62-client-qd2-defer1-retry1`, `r62-guard-qd2-defer1-retry1`.
A3 boundary unit tests4/4 passed. No fresh model inference at this decision.
First complete20-request smoke and40-forward audit using non-generating readiness;
then stop smoke server and start a new, empty full-replay capture session.

Boundary smoke completed:20 requests EXIT_ZERO, complete transaction audit40/40,
independent target/lookahead input/ownership/transition parser40/40. No extra
forward. `/model_info` readiness+empty directory PASS. Portable reports are in
`evidence/qd2-capture-incident/boundary-smoke-*.json`.
Verified guardian286289 was signalled to stop only child group286294; planned
shutdown end1789472303.391125. No model production source changes.
Retry metadata SHA `f5f64dd4f1a4d11e707b229318981b31692096d734567d04e5eade8fb0dacad3`.
Collector SHA `8f0a501a587750db692c674f0e56a96c7e6830ba70cba378078402d35390909c`.
The startup-boundary correction is smoke-verified; full QD2 integrity remains
BLOCKED until fresh8192 replay and all complete audits PASS.

Client launch typo: `r62-client-qd2-defer1-retry1` used nonexistent
`frozen-corppora/qd2.json`, failed at input-file open before output directory
creation or requests. Explicit check showed0 captures. No model run was resumed
or merged. Corrected path exists; new supervisor is
`r62-client-qd2-defer1-retry1-launch2`, with the same fresh output root and
already-ready uncontaminated server. Original launch-error evidence retained.
