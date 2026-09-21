# MXFP4 NUMA admission validation

User authorized a storage-specific reduction followed by one monitored validation
on2026-09-21. Preserve NUMA2/3/6/7, exclusive physical15, C128, global704GiB
MemAvailable admission,640GiB container/no-swap limits and existing HBM limits.

Only MXFP4 uses48GiB free per selected node. BF16 and Q8_0 retain80GiB. This is a
provisional validation policy, not a new universal capacity qualification.

The original80GiB check is a startup free-memory threshold, not a measured peak.
BF16 v10 scheduler numa_maps shows roughly56.3–59.0GiB anonymous resident pages
per node at one snapshot, not continuous per-node maxima. MXFP4 expert payload
is59.75GiB total, approximately14.94GiB per pool if evenly placed. File-backed
residency and cgroup charging differ, so neither bank size nor cgroup-total/4 is
proof of per-node runtime peak. This validation collects the missing evidence.

Admission requires the hard free floor and, for MXFP4, a conservative estimate:

`max(0, MemFree + max(0, Inactive(file)-Dirty-Writeback-Shmem-Mapped) -8GiB)`

must cover`59.75GiB/4 +32GiB`. This is explicitly a heuristic, not a Linux NUMA
MemAvailable API or a reservation. Dirty, mapped and shared file pages are
excluded conservatively; ordinary kernel reclaim remains under kernel control.

The owned guard now records all selected-node meminfo fields each iteration,
plus owned cgroup memory.stat/memory.numa_stat when present. It stops only its
named container if a selected node has both less than8GiB MemFree and less than
8GiB conservative reclaim headroom, or if the owned cgroup reports an allocation
failure/high/OOM event. Existing host, device-identity, HBM and monitor-timeout
stops remain active. Sampling intervals include telemetry execution time; they
are not a guaranteed one-second deadline.

Targeted admission/guard tests:18 PASS. Cases cover current79.4GiB admitting only
MXFP4, rejecting47GiB even with abundant cache, excluding dirty cache from the
estimate, and rejecting unknown storage. Full-model trial33 is the first actual
validation, using frozen framework-numa-validation-v1 and previously installed
F4 overlay8. A new admission/source receipt is mandatory; old qualification is
not inherited for the new memory policy. Result remains PENDING until real load,
prefill migration, decode and abort/reuse evidence is collected.

## Trial33: clean file-cache growth exposed a runtime false stop

Trial33 stopped during loading at07:57:27UTC. NUMA3 MemFree fell to5.23GiB,
while its anonymous pages were4.85GiB and inactive file pages229.26GiB. Owned
cgroup file pages increased rapidly; allocation failure/high/OOM counters stayed
zero. HBM owned was24822MiB. The8GiB MemFree-only rule stopped the owned process
before a full-model smoke. This is a guard-triggered incomplete trial, not PASS.

The original runtime rule repeated the same free-versus-cache confusion as the
startup blocker. Updated runtime policy stops on low free *and* low conservative
reclaim headroom, or any owned cgroup high/OOM/allocation-failure counter. Startup
48GiB remains a hard requirement. No working-set peak is inferred from filecache.
Two added tests reproduce this clean-cache case and reject dirty/unreclaimable
shortage and actual cgroup failures. The final policy plus receipt summarizer has
23 targeted admission/guard/summary tests PASS.

After trial33 exited, only275 previously inventoried project-owned unmapped files
received DONTNEED advice. Every inode/size/mtime was checked before and after;
no file bytes were written and no global drop_caches was used. Trial34 uses a new
frozen source directory framework-numa-validation-v2 and fresh admission to
repeat the same F4 candidate with the corrected runtime guard. All prior trial33
logs remain intact.

## Trial34: monitored framework validation passed

Trial34 started from a fresh48GiB MXFP4 admission and completed model loading,
two HOLDOUT HTTP requests, four Prefill-End Dynamic transactions, active abort
and request-state reuse. The two canaries were finite; the arithmetic canary
returned391. Dynamic evidence contains four commits and four request drains.
The first three commits changed all48 layers and copied1420,785 and785 experts;
the reuse request required no further migration. First-output-token generations
were0,1,2,3 and decode generations were1,2,3,3. Every commit records preserved
live request state across the migration barrier.

The monitored receipt contains1210 samples, of which1205 include owned cgroup
NUMA statistics. Per-node peak owned anonymous memory was23.086,18.063,15.072
and14.997GiB on NUMA2/3/6/7. Minimum MemFree was6.619,15.555,208.550 and
207.287GiB. Minimum conservative available headroom was129.239,148.963,202.084
and200.732GiB. Owned cgroup peak was342.867GiB; peak owned HBM was25.816GiB.
There were no cgroup allocation failures, high events, OOMs or OOM kills.

The machine-readable result is
`evidence/physical15/20260920-full-model-v34/numa-validation-summary.json`, bound
to SHA256 of the raw monitor and driver receipts. Status is
`PASS_MONITORED_MXFP4_NUMA_VALIDATION`. This qualifies the storage-specific
48GiB startup policy for the exact MXFP4/C128/Q32/Layerwise/Prefill-End short
framework smoke. It does not prove a universal minimum, shared-workload safety,
quality, performance, or a BF16/Q8_0 threshold change. BF16 and Q8_0 remain at
80GiB. The v34 container was operator-stopped after evidence collection; Docker
exit137 reflects the controlled stop after its grace period, not an owned OOM.
