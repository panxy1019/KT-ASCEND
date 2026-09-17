# C64 runtime qualification — isolated from active R32

User-approved runbook: `/home/admin/Downloads/02_R16_R8_64CORE_DUAL_CONTAINER_RUNBOOK.md`.
Audit branch: `codex/round6-7-c64-numa`; parent `efac267a12412e8117468c95823f06da17dff622`.
No change to active R32 code, binary, environment, processes or output directory.

Execution order: sparse-NUMA negative control and fix → physical affinity and
memory locality → F32-only1408 split audit/operator tests → R16 compact and
C32/C64 short A/B → R8 smoke → measured dual-container admission → CMMLU.
B16 is deferred in this runbook. C64 is a new identity, not a silent update to C32.

Container `kt-r67-c64-dev` exposes only NPU2, CPU128–191 and memory nodes4,5.
All existing mounts are read-only. Only `/workspace/c64` is writable host evidence.
Snapshot image `b81e664642aecd5f4cc8ec38b233aae38f65b9cf7feea40e54e13c08cd319347`.
No install into shared `/workspace/results/venv-r6`, no production source edit.
Initial low-parallelism C++ probes are CPU-only; no model or weight loading.

Current finding: both WorkerPool and distributor index per-node counters using
physical IDs into vectors sized by slot count. Distributor worker memory policy
also uses slot rather than physical node. Negative control must compile the
unmodified deployed cpp, not a reconstructed toy formula.

Updated checkpoint: sparse maps [2], [2,3], [4,5] pass the isolated candidate
probe; 64/64 worker affinity and scratch-page locality verified. Dense maps
[0]/[0,1] deferred to protect R32. Loaded expert weight locality NOT_RUN.
1408_TP2_OPERATOR=NOT_RUN; current F32 path does not support qualified TP2.
LONG_C64_QUALITY=NOT_STARTED. See 01/02 for preserved evidence and capability stop.
The original C32 serial controller remains active and is not redirected here.
