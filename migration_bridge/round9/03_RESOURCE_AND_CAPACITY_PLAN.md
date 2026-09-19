# Capacity plan — theoretical, not admission approval

From frozen config:3*2560*640=4,915,200 routed parameters/expert.

| Payload | GiB |
|---|---:|
| All48*512 routed BF16 |225|
| All routed F32 CPU backing |450|
| Q64 routed NPU BF16 |28.125|
| Q32 routed NPU BF16 |14.0625|
| PLE BF16 rough model-card estimate |about95, actual census pending|

F32+PLE alone approximately545GiB, excluding runtime, source pages, anonymous
copies, NUMA replicas and other weights. Existing Round8 384GiB container
settings are NOT an admitted Round9 budget. Do not automatically enlarge them
or launch a second full pool. Q64 payload does not prove single64GiB NPU fits.

Index checkpoint size359999963128bytes (~335.276GiB); keeping source plus
450GiB conversion needs roughly785GiB disk before temporary/old artifacts and
download cache duplication. Use streaming identity/conversion and account for
same-inode mmap page sharing separately from per-process anonymous copies.

Before large download: fresh host/cgroup MemAvailable and limits, disk and
temporary peak budget, NUMA physical cores/locality and NPU inventory. Before
runtime: complete header-derived expert/PLE/nonexpert totals, loading copies,
KV/indexer/GDN/GR and staging peaks, reserve and recovery plan. Check C64
subpool implementation does not replicate the full450GiB expert bank.

Record RSS/RssAnon/RssFile, memory.stat anon/file, host cache estimate,
major/minor faults, swap baseline plus pswpin/out deltas, processVmSwap and PSI.
Only bounded staging may be pinned. Existing unrelated jobs remain untouched.
Result stays RESOURCE_ADMISSION_PENDING until actual inventory and budget
are frozen; insufficient capacity is BLOCKED_RESOURCE_ADMISSION, not permission
to switch model/precision/TP/host or suppress PLE/QSA.
