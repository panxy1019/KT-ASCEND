# Stability results and descriptive-capture harness recovery

## Stability: A3 verified, finite observation only

Two independent campaigns,6 prompts×64 tokens×3 cycles each:36 requests,
2304 generated tokens. Both `STABILITY_RUNTIME_PASS_PENDING_HBM_REVIEW`;
cross-campaign references exact, finite outputs and pending tasks drained.

| Metric | Campaign A | Campaign B |
|---|---:|---:|
| RSS start bytes | 31725993984 | 31730311168 |
| RSS peak/end bytes | 32254189568 | 32270389248 |
| Cycle-end RSS increments bytes | 0,0 | 0,8192 |
| Cycle-end HBM MB | 30896,30896,30896 | 30896,30896,30896 |
| Owned HBM samples | 13 | 13 |

Warmup growth followed by an observed plateau; no sustained cycle-end growth
or memory runaway observed. RSS sums can double-count shared pages. This is
not proof of zero leakage or asymptotic boundedness. No other NPU0 memory
process observed during these samples. Resource observation supports this
finite stability gate; final qualification remains separate.

Evidence: `evidence/r64-minimal-stability-{a,b}-v1/`,
`evidence/r64-stability-memory-analysis.json` and corresponding raw JSONL.

## First descriptive attempt: preserved environment failure

`r64-descriptive-final-v1` failed before any generation request:
`ModuleNotFoundError: audit_deferred_transitions`. Its capture index contains
zero rows; server log contains readiness GET but no generation request.
The deployed Round6.3 script assumes its original repository sibling layout,
whereas the actual dependency is in
`/workspace/results/round6_2-prequalification/tools`.

Classification: TEST_HARNESS_DEPLOYMENT_IMPORT, not model numerical failure.
Old failure, source hashes, process logs and empty index remain preserved.
The original queue remains `STOPPED_FIRST_FAILURE`; do not relabel it PASS.

Minimal tool-only fix: supply the existing helper directory in child
PYTHONPATH while retaining inherited paths; import-check before model
allocation. No package install, production change, corpus or rule change.
Independent A3 helper import passed. Local tests cover inherited environment
preservation and missing PYTHONPATH. New deployment `r64-minimal-next-v3` and
new output `r64-descriptive-final-v2`; previous output is not overwritten.

At this document's creation v2 is RUNNING, not yet integrity-qualified.
Completed quality/performance/stability are not rerun.
