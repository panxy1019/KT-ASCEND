# Validated BF16 backing cache feasibility

User authorized acceleration after observing large Dynamic qualification costs.
Current runtime-v7 remains active for the original DEV selection. This document
does not claim a speedup or authorize mixing measurements across implementations.

Measured observer-OFF threshold512: three triggered requests take428.3s total;
logged preparation367.30032s and update33.72897s across78 layer calls. These are
host-call scopes, not independent device event timings. Repeated F32→BF16,
BF16→F32 identity comparison and finite validation occur in every stager.submit.

Candidate source commit: `3c596d289dc328222fbda73acf59501bcf8461bc`.
`verified_bf16_cache.py` is experimental and not imported by production. It
retains per-expert validated BF16 CPU tensors on the existing wrapper, across
temporary stager reconstruction. Cache keys bind loader, layer and all projection
addresses/shapes/dtypes/versions. A changed binding fails; failed validation does
not publish. External changes to mmap files cannot be detected by tensor versions:
production use additionally requires the frozen read-only artifact contract.
Pinned output slots and original framework copy/event lifecycle are unchanged.

Maximum cached payload for64×26 experts is26.8125GiB, excluding allocator
overhead. This is ordinary host backing, not a new globally pinned weight pool.
Memory admission and actual steady/peak usage must precede full-model deployment.

## Isolated experiment

Early/middle/late layers1/13/26, all56 initially CPU-owned experts per layer:
original staging, cold cache construction, three warm passes with fresh temporary
stagers, all-expert output equality, independent checkpoint/H2D controls, and
stale-binding rejection. Separate timings and cache bytes are reported. A3 model
speedup, numerical equivalence or final scheduler benefit cannot be inferred
solely from this primitive benchmark.

106 pure tests pass both locally and on A3. New-source syntax check passes.
ReleaseSHA `c3a842148e34b93babeb2d19f1458a922fc22200d4a0bfb3c1192e8efcc723af`;
completeSCP and host/container hashes verified before separate extraction to
`/workspace/c64-dev/round7/cache-tools-v1`.

Owned flock watcher `cache-feasibility-auto-v1.log` waits for the original DEV
selected JSON AND the original threshold_sweep process to exit, then runs
`benchmark_verified_bf16_cache.py` under the same C64 environment. Output:
`/workspace/c64-dev/round7/cache-feasibility-v1`. Six-hour bounded wait; it cannot
start HD, alter production runtime, or publish a candidate acknowledgment.

Next: audit isolated correctness/performance and host memory; if worthwhile,
port as a new provenance-bound runtime, preserve old DEV results, validate
new runtime identities/restart/parity and repeat affected DEV measurements before
candidate freeze. Do not overlap this benchmark with formal host-bandwidth timing.
ABBA process reuse remains a separate proposed orchestration change requiring
explicit initial-resident-state equivalence; no measurement protocol changed yet.
