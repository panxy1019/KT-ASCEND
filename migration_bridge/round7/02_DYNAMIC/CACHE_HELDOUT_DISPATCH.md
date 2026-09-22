# Runtime-v8 frozen candidate and parallel qualification dispatch

2026-09-18, after all3 system probes and all48DEV screen requests completed.

- Exact candidate commit: `0be78ade3835d7adadc2f984fc39e63dd817af12`.
- Threshold1024, unchanged frozen selector; no held-out/CMMLU fitting.
- Candidate bytes independently checked against `git show` before ACK.
- Tools-v25 SHA256 `97b21b257d9ab4bfae99225cd9c76504984ce6c1739741cd32d0dfec75fe3cf6`.
- 118 local and A3 pure tests PASS; archive transfer/hash verified.

| Worker | Task | Output |
|---|---|---|
|A/NPU4,CPU0–63|Static Uniform HOLDOUT128×16|`/workspace/parallel-worker/runtime-v8/uniform-holdout-v1`|
|B/NPU6,CPU64–127|Dynamic1024 HOLDOUT128×16|`/workspace/parallel-worker/dynamic-holdout-v1`|
|Original/NPU2,CPU128–191|Dynamic1024 CMMLU256|`/workspace/c64-dev/round7/cache-dynamic-quality-v1`|

Dispatch is not completion. No quality score or held-out PASS claimed yet.
Parallel timing is not isolated-performance evidence. FormalABBA waits for
completion/reaping and an eligible host resource window.

Quality system predecessor is completed1024 migration probe. Explicit unchanged
CPU operator evidence binding PASS on A3: source summarySHA
`e350642f9d42893929344aa218f10aff5e911581e9cc5ab4ea0764738068dc53`;
source provenanceSHA `3e720e018d0675332f6ac090d0f266fbeb4821a7ba895f2c4583b8273a153c94`;
target provenanceSHA `94dd778a6cd465f81124edd4a70244ddae351a5b88c13ca1e5e1d88507715716`.
Native461821.../GGUF/model/software identity checked, no operator rerun fabricated.
The cache/migration safety prerequisites are the newly completed system probes.
Quality cannot override any later held-out system failure.

Host reserve before dispatch985GiB, owned servers idle. Host cold swap5MiB is
not silently called zero; existing per-run admission checks require container
swap0 and no active host swapping. Business container untouched.
