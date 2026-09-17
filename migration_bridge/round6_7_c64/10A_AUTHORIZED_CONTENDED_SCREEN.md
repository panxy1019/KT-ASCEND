# User-authorized concurrent performance / quality screen

2026-09-17: user requested immediate C64 performance and CMMLU while existing
R16/R8 C32 quality jobs continue. No existing product job is stopped.

## Explicit protocol change

The isolated-host prerequisite is waived **only for exploratory timing**.
Formal isolated performance remains unqualified. The original 3 ABBA blocks,
five frozen input lengths, batch1, placements and native implementation remain
unchanged. A `formal_performance=false` field travels with timing and quality
provenance. An observed latency ratio is not proof of intrinsic C64 speedup.

Quality entry is explicitly authorized independently of demonstrated isolated
speedup. There is no change to the frozen CMMLU256 manifest or -5pp quality
criterion. Successful system execution, finite outputs, unchanged native and
wrapper hashes, memory audit and R8 compact/local smoke remain prerequisites.
Without the explicit new CLI option, default tools still require beneficial
isolated timing and reject exploratory results.

## Execution

Container `kt-r67-c64-development-v1`, physical NPU2, C64 CPU128–191 NUMA4/5.
Reference canary `r16-canary-v3/summary.json`: PASS.
Host admission at 07:30 UTC: available approximately 935 GiB, swap used zero.

Order:
1. `performance-contended-v1`: C32/C64 three-block ABBA, same NPU.
2. Memory audit (no OOM, reserve, no sustained major-fault storm).
3. `r8-canary-authorized-v1`: compact/restart/observer and extended local smoke.
4. `r16-quality-authorized-v1`: frozen CMMLU256.
5. `r8-quality-authorized-v1`: frozen CMMLU256.

Driver: `continue_authorized_screen.py`; state:
`/workspace/c64-dev/authorized-screen-progress.json`.
Any runtime/safety failure stops this new chain, not the existing products.
A quality score failure remains a per-profile result, never a threshold refit.

Deployed tool archive SHA256:
`10d6a7d0863dda593e87ddd02073658d8d7cf9bc4c807fb7426490b9cfe80024`.
Production native SHA256 is unchanged:
`a5497fa085d704d8f67484b3936eaf448080f5d445899d88e4e76bea4305f077`.

This authorization does not establish formal speedup, whole-round readiness,
or permission to terminate other tasks. Quality/performance results remain
pending until their output artifacts are complete.
