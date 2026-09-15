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
