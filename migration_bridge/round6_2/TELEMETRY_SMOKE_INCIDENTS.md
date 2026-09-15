# Round 6.2 telemetry smoke incidents

## 2026-09-15: outer hooks bypassed

Evidence state: A3_VERIFIED for the failed smoke; LOCAL_VERIFIED for the
subsequent adapter correction. No QD2/HD2 output was used.

The first instrumentation-ON D1 smoke request returned HTTP 200, but the
client found no committed `Pass*.pt` within its frozen 30-second capture
deadline. This is a capture failure, not a numerical conclusion.

Run: `/workspace/results/r62-smoke-defer1-on`.
Guard: `/workspace/results/r62-guard-defer1-on`.
Own process group: 264157; guard PID: 264152. SIGTERM was sent to the guard,
which terminated only its own group (SIGTERM followed by SIGKILL). Guard
result records CANCELLED, returncode -9. Other workloads were not signaled.

Static cause: this SGLang ModelRunner invokes `self.model.forward(...)`
directly. PyTorch module forward hooks attached to the outer model do not
execute on that call. The debug transaction therefore never began or
committed, although the nested model and production inference executed.

Correction: bind the debug capture around the actual instance `forward`
method, only with the Round 6.2 telemetry/reference switch enabled. Do not
alter model arithmetic, scheduler, placement or Deferred ownership. Add
tests explicitly calling `model.forward`, including the reference observer.
The local suite passes 29 tests after this correction.

Retry is a new prequalification smoke directory, `r62-smoke-defer1-on-v2`.
It is not a retry of numerical held-out data. Its result remains pending.
