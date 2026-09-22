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

## Completed A3 feasibility and original DEV screen

2026-09-18 03:21UTC. Original runtime-v7 DEV screen completed48/48. Its prescribed
selector chose1024; median E2E ratio1.003366, long-bucket ratio3.198006 versus
Static. This is historical DEV exploratory evidence, not formal benefit or
held-out qualification. No acknowledgment was issued for post-DEV consumers
while the user-authorized acceleration changes are being evaluated.

Cache feasibility PASS (torch CPU threads1, same original environment):

| Layer | Experts | Original seconds | Warm median seconds | Staging speedup |
|---|---:|---:|---:|---:|
|1|56|4.850536|0.151439|32.03x|
|13|56|4.751025|0.149763|31.72x|
|26|56|4.736833|0.149378|31.71x|

Every expert output matches original staging. Independent checkpoint/H2D controls
and stale-binding rejection pass. Each layer caches968,884,224bytes for56experts;
full-model resident memory overhead and end-to-end gain are still unverified.
Cold conversion is4.56–5.03s, so first-use latency is not eliminated.

Lossless original DEVscreen, completedDEV1024 and cacheprobe evidence:
`../raw/dev-screen-and-cache-feasibility-v1.tar.gz`, SHA
`7c556cef6d5543c2d399d021eb505b5edec7a80215ea282cd497b1e80de61749`.
SCPexit0/local+A3hash/fulltarPASS. Original256/512archives remain separate.

Candidate private runtime port commit9c8859187 adds an explicit opt-in cache
source port and helper hash binding. Default source-port behavior unchanged.
Tools-v22 archiveSHAa9034ef92809087c1c94060992567c841aa857d6d133f57b6ae50cdbe4e8f0e3;
completeSCP/hash-beforeextract,108LOCAL+A3puretestsPASS.

Only Dynamic staging is changed; native CPU arithmetic and Static execution
remain untouched. Required fresh checks: Uniform compact/restart, original
Dynamic probes256/512/1024 including observer parity and memory/lifecycle, then
same DEV selection under new runtime. Existing operator/Frequency/R64 gates
remain prior evidence; they are not copied into new runtime as newly executed.
No repeat of the entire Static Frequency campaign is planned for this scoped
cache change. A fresh selected-Static HOLDOUT comparator remains planned once.
