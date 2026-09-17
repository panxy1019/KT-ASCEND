# Sparse physical NUMA qualification checkpoint

Date: 2026-09-17 UTC. Target: 110.129.0.22 / 910B3 host.
Evidence state: TARGET_VERIFIED for the isolated C++ probe only; no model rollout.

## Negative control

Compiled the actual deployed unmodified `worker_pool.cpp` with AddressSanitizer.
Map `[2]` produces heap-buffer-overflow in `WorkerPool::init` at the physical-node
index into a vector sized by number of subpools. Full report is retained in
`evidence/worker_pool_v3/asan-old.28`; exit134. This is not a synthetic toy check.

## Candidate changes

- Use physical-node-keyed counters; keep subpool arrays indexed by slot.
- Resolve hwloc NUMA objects using OS index rather than topology logical index.
- Distributor memory binding maps slot to physical NUMA ID.
- Validate node/count/core availability before constructing worker threads.
- Fail on binding errors; do not silently run unbound workers.
- Fix optional direct-work branch's physical-node-to-slot lookup.

These are isolated candidate changes. The running R32 source and extension were
not modified. Full native extension and model regression remain outstanding.

## Target container

`kt-r67-c64-runtime`, image
`b81e664642aecd5f4cc8ec38b233aae38f65b9cf7feea40e54e13c08cd319347`.
CPU64–191, memory nodes2–5, SYS_NICE for NUMA memory policy, NPU2 visible but unused
by these probes. No host library installation, business container modification,
or shared virtualenv write. Probes execute sequentially, not all 128 cores at once.

The first default-capability disposable container denied memory-policy syscalls;
that failure is preserved, not called a pass. Only the new dedicated container
was granted SYS_NICE. R32 privileges were unchanged.

## v4 results

| Map | Workers | Repeats | Observations | Failures | Exit |
|---|---:|---:|---:|---:|---:|
| `[2]` |32|10|320|0|0|
| `[2,3]` |64|10|640|0|0|
| `[4,5]` |64|10|640|0|0|
| Invalid node999 |—|—|—|Expected rejection|3|
| 33 workers on 32-core node4 |—|—|—|Expected pre-thread rejection|3|

Each observation checks actual `sched_getcpu` NUMA, one-CPU affinity, thread name,
and freshly touched page's physical node using `get_mempolicy`. Logs identify
each worker/TID/CPU. This proves scratch-page locality, **not loaded expert-weight
page locality** (weights have not been loaded into C64).

ASAN and `_GLIBCXX_ASSERTIONS` enabled; leak detection disabled. No leak-free claim.
The earlier v3 oversized-core test aborted after thread creation; v4 prevalidation
corrects this and its v3 failure remains archived.

Candidate probe binary SHA256:
`981af847140f8f52b59bdb021813080cc310a03284d551055fcfcebcb0b3504d`.
Candidate cpp SHA256:
`a3076a95a3408b10fdbc73bb6010b3d876b3f33b9d23de6a9eeffc576d1772e1`.
Candidate header SHA256:
`fe7d2b7b901874f9811921331565a8c0e3536980c67cfc2523a3afe80817ed3c`.

## Remaining qualification

`[0]` and `[0,1]` intentionally NOT_RUN while R32 uses node0. Do not disturb it
to complete this test matrix. Dense-node regression, full-extension regression,
loaded-weight locality, 1408 TP2 operators, and model compact tests are pending.
Therefore overall C64 runtime readiness remains NOT_QUALIFIED.
