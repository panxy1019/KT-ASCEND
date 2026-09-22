# Round9 resource proposal for 110.123.0.3

2026-09-20 03:16–03:17 UTC read-only inventory, requested by user considering
this machine for subsequent development. This supersedes .22-only planning as
a target preference, not an admission PASS or completed migration. .22 download
and paused automation were not changed.

## Observed machine

- Host a3-server-00, openEuler aarch64 kernel6.6.0-72.0.0.76.oe2403sp1.
- lscpu:640logical CPUs,320physical cores,4sockets×80cores,SMT2.
- Eight NUMA nodes,40physical cores/~250GiB RAM each; siblings are adjacent
  pairs (e.g.0–1). Do not copy .22's contiguous64-CPU cpuset or count SMT
  threads as physical workers. CPU flags include SVE/SVEBF16/BF16.
- free:about2.0TiB total,1.9TiB available.116MiB historical swap residency;
  no swap-activity delta measured, so this is not a no-swap-activity claim.
- Root48GiB free, /home962GiB free, /mnt/data1 about3.2TiB free on NVMe-backed
  XFS LVM. Use /mnt/data1 for proposed artifacts; /tmp is tmpfs.
- npu-smi26.0.rc1:8NPU devices with2chips each,16physical IDs,each65536MiB HBM.
  All16 chips report running python3.10 processes, about9GiB total HBM used
  each and AICore utilization17–71%. No exclusive NPU currently admitted.
  HealthOK does not imply available. No running task was changed.
- Selected PCI numa_node entries return-1; NPU/CPU locality remains unresolved.
  CPU/memory/I/O PSI files are absent. Container inventory requires sudo;
  no privileged inventory was performed in this read-only preliminary pass.

## Proposed resource envelope, not measured runtime peaks

| Stage | CPU | Host RAM reservation / limit proposal | NPU | Persistent disk |
|---|---|---|---|---|
| Source/build/header work |8–16physical cores|32–64GiB|none|50–100GiB plus checkpoint if copied|
| Bounded real-operator gates |C32 thenC64;2×32 forC64|64–128GiB; avoid materializing whole packed layers unnecessarily|one exclusive64GiB die when NPU comparison needed|within main workspace|
| First Q64 BF16-storage full reference |64expert workers+4–8auxiliary physical cores|reserve640GiB; provisional768GiB hard limit only after admission|one exclusive64GiB die,TP1|reserve1.0–1.2TiB total workspace|
| MXFP4 or Q8_0 product study |sameC64|provisional384GiB MXFP4 /512GiB Q8_0; conversion/loading may still need reference envelope|same die and NPU precision|same workspace,serial candidate lifecycle|
| C128 study |4×32expert workers+4–8auxiliary physical cores|start with same full-reference envelope,not4copies of bank|same die,TP1|same source/bank artifacts|

RAM rationale:BF16 routed bank225GiB + preliminaryPLE~95GiB =about320GiB
payload before nonexpert weights, source pages/loading copies, runtime, staging
and allocator. Proposed640GiB reservation gives investigation headroom; it is
not a claim of640GiB steady RSS. Census and bounded loader peaks must precede
final memory caps. Avoid pinning full weights/PLE. CPU bank partitions must not
duplicate the full pool. MXFP4/Q8_0 bank payloads are~59.77/~119.53GiB;
PLE is unchanged. Keep originalBF16 source for exact NPU promotion.

Two NUMA nodes together have only about500GiB total RAM. A640/768GiB envelope
cannot be memory-bound solely to those two nodes. Place C64 expert partitions
locally on the worker nodes where feasible; plan PLE, source pages and other
allocations on additional admitted nodes. Record actual NUMA page placement and
bandwidth contention. CPU cpuset and allowed memory nodes are separate choices.

Q64 resident routed BF16 payload28.125GiB is not total HBM. A64GiB die is a
reasonable first candidate, not proof of fit: attention/shared/GR/runtime/KV
need measurement. Long-context tiers need separate memory admission; no128k
capacity promise and no silent TP2 expansion.

Disk budget accounts for~335.3GiB checkpoint +225GiB BF16 bank +one active
low-bit candidate, build/runtime and conversion temporary space.1.2TiB is the
preferred reservation to tolerate source/temporary coexistence, not permission
to preallocate it. Do not place models on root or tmpfs.

## Migration and scheduling implications

Keep the .22 downloader working; after full identity verification, plan one
verified transfer to persistent local storage on .3, not a second Internet
download or unqualified network-backed expert bank. This proposal launches no
copy or new service. User preference for .3 does not authorize interrupting
existing NPU jobs. Select the die/core IDs after an availability window exists.
For timing, also inspect contention on the paired die/card and memory channels.

The CPU advertises SVE whereas the current reference code's BF16→F32 effective
vec-dot override applies only to aarch64 builds without __ARM_FEATURE_SVE.
Actual build flags and imported native variant must be frozen anew; hardware
flags alone do not prove which path executes. Recheck CANN/torch/torch_npu and
device mappings in an isolated environment. Preserve the F32 partial contract
and the already identified640/320/160 shape gaps. Do not inherit .22 qualification.

Recommended default reservation request:one exclusive64GiB NPU die,
72physical cores (64expert+8auxiliary),640GiB RAM,1.2TiB /mnt/data1 disk.
Request a separate136physical-core window forC128 rather than reserving128
expert cores throughout development. Current blocker for NPU work is scheduling,
not observed whole-host RAM or disk capacity.
