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
