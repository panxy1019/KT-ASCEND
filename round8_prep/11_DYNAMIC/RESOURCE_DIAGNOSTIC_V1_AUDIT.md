# Single-cell resource attribution audit — 2026-09-19 22:20 UTC

Owned .22 unit ended successfully at22:15:54; state is
DIAGNOSTIC_COLLECTED_PENDING_AUDIT with exactly dynamic-0-0 collected.
All owned model containers are stopped; unrelated workloads untouched.
This instrumented run is diagnostic, NEVER formal performance evidence.

## Observed result

Original protocol SHA remains
1b8ad5587cab6cccbc64618ac96616222d0b01015407d6ee27f03cccbfe70243.
Short and medium both HTTP200; E2E26.092079/471.731613seconds.
No Traceback, watchdog termination, NZ offset warning or OOM string observed.
Forty samples span1789855538.109034 through1789856064.584155;
requests span1789855562.009005 through1789856059.833400, so end boundaries
are covered. Maximum sample interval13.672662seconds; not continuous tracing.

| Metric | Observed |
|---|---:|
| Owned memory.current start/peak/end bytes |234194624512 /265018466304 /262862860288|
| Owned anonymous start/end bytes |140037251072 /239721967616|
| Owned file start/end bytes |93479706624 /22235451392|
| Owned major-fault increment |95532|
| Owned file refault increment |4459278|
| Owned memory.swap.current, all samples |0|
| Owned process VmSwap maximum, all samples |0KiB|
| Owned memory.events increment |all zero|
| Owned memory PSI some/full total delta |1871015 /1871014 microseconds|
| Host swap-in/out increment |4758 /6834 pages|

Owned anon/file bytes remain on NUMA4/5 according to memory.numa_stat.
At final sample nodes4/5 free memory is171400/683892KiB respectively.
The approximately99.685GB anonymous increase and71.244GB file-page decrease,
with reclaim/refault evidence, establish owned memory/file-working-set pressure.
They do NOT prove owned swapping: owned swap remained zero while host swap
increased. Do not identify which unrelated process swapped without evidence.
Do not equate major faults with swap; file-backed faults are distinct.
Field interpretation follows the Linux primary documentation:
https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html

## Deployment audit and next discriminating check

Actual wrapper SHA remains
77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b.
It imports cached_stager_class in the LLAMAFILE BF16 full-context path.
Actual verified_bf16_cache.py SHA:
fc67bac2d5e1f6b78a520e4fa6f0b3e711fb72970ffa9f27032d0b9aa02a689b.
Its docstring claiming not imported by production is stale for this candidate.
Cache is attached per wrapper and retains every converted expert; it has no
byte-budget eviction. For40layers x224initial CPU-owned experts x3projections
x2048x512x2bytes, first-fill payload is52.5GiB. This is a concrete retention
source, but DOES NOT explain the entire92.84GiB anonymous increment by itself.
Allocator retention/conversion temporaries or additional buffers require
attribution; no speculative production patch was applied.

Actual GGUF loader SHA:
fb65e04d751c833b43deb7b05d26e60f412da06d22d3705aa9c0711c811cc483.
get_undequanted_tensor_and_ggml_type uses mmap-backed from_numpy views, not an
obvious full anonymous copy at this boundary. Next inspect cache resident-byte
accounting/allocator behavior or evaluate a separately identified bounded-cache
diagnostic with exact staging regression. Do not alter frozen candidate in place.
No further full ABBA launched, no expanded mems/C64, no host swap/drop-cache edits.

## Evidence and status

Archive exists on persistent .22 /home/admin/kt-round8-qwen35 and local
round8_prep/raw/qwen-dynamic-resource-diagnostic-v1.tar.gz; hashes match:
1f9bb78f60e551d13838ff682798e14071840ef56dc5f4062ec5d1f36a2f5728.
Resources SHA:c1068d235c3ca8d94a93e5054f0d8d2173649c3ccce044b5cd7dea372502c305.
Raw SHA:4f591ea9857224d621207611157d4a0bf5f8045a2568df08717c25c46e2862d9.
Summary SHA:94dc50dc2c9957bf279bb98e71f32aad0cf1ef0bc4d8675b22ba7d75ee5e8304.

Resource attribution advanced, not fully closed. Existing12-cell strict-isolation
qualification remains NOT_QUALIFIED_RESOURCE_AUDIT. Static product/system/quality
evidence remains valid; defaultDynamicOFF,DeferredOFF. Round8notCOMPLETE.
Desktop free disk~12MiB: do not copy further large evidence here or delete
user-owned files. Remote evidence remains persistent; small manifests only.
