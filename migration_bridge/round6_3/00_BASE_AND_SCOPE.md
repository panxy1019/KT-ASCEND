# Round 6.3 execution baseline

Execution authorized by the maintainer on 2026-09-15 after review of
`ROUND6_3_REVIEW_PLAN.md`. The review draft is preserved unchanged.

Parent base: `5e708dc63d07afe4d79d26f0bdbb97d95ad3aaba`.
Child base: `0a667fee2afd4b2d65fa6e4bff13a714f1407f37`.
Execution branch: `codex/round6-3-decode-exposure-qualification`.
Inherited dirty `third_party/llama.cpp` is outside this work and remains untouched.

## Scope and authority

Execute reviewed A (stateful decode batch1 qualification), then gated B
(batch2/4). Investigate and minimally repair failures without requesting routine
approval. A failed frozen experiment remains failed; a repair requires a new
identified run and, where statistical rules change, independent qualification
and holdout. Do not silently relax acceptance or recycle held-out results.
Do not interpret autonomous continuation as authorization to add unrelated
models, TP2, Dynamic Placement, Graph, new quantization or unsafe transfers.

Round6.2 remains `BLOCKED_INSUFFICIENT_EXPOSURE`: 343 < 500; no numerical
candidate was derived. HD2 remains sealed. New pilot precedes new QD3/HD3 freeze.

## A3 initial observation

2026-09-15 14:32:56 UTC: SSH accessible; npu-smi reports healthy devices,
logical visible chip 0 HBM 3118/65536 MiB, AICore 0%, no reported NPU processes.
This is an observation, not a resource reservation. Recheck before serving.
Only this project's disposable container and NPU0 are in scope. No host library
or other workload changes. Deployed SHA plus overlay hashes must be recorded
separately from local commits before model runs.

## Checkpoints

1. Source/provenance and offline selector audit.
2. Default-OFF teacher-forcing adapter and transaction schema; negative tests.
3. A3 forced-history / observer parity, independent pilot, cost measurement.
4. QD3/HD3/FD3 and statistical preregistration, QD3 integrity/exposure/derivation.
5. Candidate freeze, D1 runtime, HD3, free generation, quality, ADR-A.
6. Gated batch2/4 and final evidence freeze.

No new gate is A3_VERIFIED_READY at this initial checkpoint.
