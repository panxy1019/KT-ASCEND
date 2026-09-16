# Batch2/4 engineering qualification — in progress

No batch quality or speedup approval has been issued. Batch1 remains the sole
primary quality endpoint. Deferred OFF and ON use distinct same-path references.

## Attempt v1: tooling failure before requests

`r64-batch-campaign-v1` fails on its first batch2 OFF model load:
`NameError: generate_gpu_experts_masks is not defined`. No request ran.
The KTEP import guard caught an ImportError, leaving its imported symbols absent.

Root cause is the new test-only capacity adapter's `execv` argv[0]. It passed
the virtualenv executable but the bare program name `python`. CPython discovered
the system prefix, not the virtualenv containing kt_kernel. Read-only A3 A/B:

| argv[0] | sys.prefix | kt_kernel discovery |
|---|---|---|
| python | /usr/local/python3.11.15 | None |
| /workspace/results/venv-r6/bin/python | /workspace/results/venv-r6 | installed package |

The corrected adapter preserves the full interpreter path as argv[0]. It still
changes only capacity/batch CLI flags. No production model, routing, placement,
kernel, or Deferred code changed. A local behavioral exec regression prevents
reintroducing the interpreter identity defect; three adapter tests pass.

The serving guard terminated only its owned failed process group. Its readiness
helper was still waiting for a server that had exited; that exact owned helper
was then stopped. The wrapper now observes the guard while waiting and exits
immediately on early server death. Three owned-process lifecycle tests pass.

Failure evidence is retained in `evidence/r64-batch-campaign-v1` including the
resolved argv, actual metadata, traceback, and guard termination report.

## Attempt v2

New root `/workspace/results/r64-batch-campaign-v2` starts the unchanged frozen
16-request short test. Sequence: batch2 OFF/ON, batch4 OFF/ON, then ten repeats
and two independent restarts per profile, each followed by full transaction
audit. Never reuse the v1 failed result as a PASS. Actual model batch, RID/KV
association, protected-expert union, route partition, terminal drain, and
repeat schedule must all be verified before reporting engineering success.

### Short-test results (A3_VERIFIED)

| Profile | Requests | Physical forwards | Actual model batch | Deferred routes | Transaction audit |
|---|---:|---:|---:|---:|---|
| batch2 OFF | 16 | 520 | all2 | 0 | PASS |
| batch2 ON | 16 | 520 | all2 | 85 | PASS |
| batch4 OFF | 16 | 260 | all4 | 0 | PASS |
| batch4 ON | 16 | 260 | all4 | 44 | PASS |

All physical forwards reconcile to known RID histories/KV positions; route
partition, global protected union, terminal drain, tensor hashes and finiteness
pass. Batch2 ON has83 exposed forwards; batch4 ON has43. These counts include
the entire captured session and are not a per-token exposure percentage.
The batch2 OFF/ON schedule hashes agree, as do batch4 OFF/ON schedule hashes.
Cross-mode token equality is not required and is not claimed.

The ten-repeat/restart campaign is still running. Short-test PASS alone does
not promote batch2/4 to deterministic, quality-qualified, or speedup-verified.
Archived JSON evidence: `evidence/r64-batch-campaign-v2/*-pilot/`.

### Batch2 OFF repeated/restart result (A3_VERIFIED)

Ten repeats completed160 requests and5200 physical forwards. All5200 actual
batches contain2 rows; all10 normalized schedule hashes equal
`6ab8ba743160654ec4a22eea9263f496e08217211f3f4061d53df82e9a438d42`.
7130 actual CPU tasks and10400 worker completions reconcile with the capture.
Both independent restarts completed16 requests/520 forwards each and matched
the prior process's per-request trajectories and normalized execution schedule.
All three final transaction audits PASS, finite outputs, no unexplained forwards.
The corresponding five JSON evidence files per stage are archived locally.

Deferred ON and batch4 long-repeat qualification remain in progress. No
cross-profile or quality conclusion is inferred from this completed OFF gate.
