# Round7 additional workers — 2026-09-18 03:34 UTC

User authorized two additional containers to accelerate independent tests.
Evidence: live root inspection of 110.129.0.22, Docker and PCI NUMA sysfs.

| Worker | Physical NPU | CPU set | NUMA memory nodes | State |
|---|---|---|---|---|
| kt-r67-c64-development-v1 | 2 | 128–191 | 4,5 | Existing runtime-v8 validation, untouched |
| kt-r7-parallel-a-v1 | 4 | 0–63 | 0,1 | Created/running, idle |
| kt-r7-parallel-b-v1 | 6 | 64–127 | 2,3 | Created/running, idle |

New container IDs:

- A: `055ce8d5247d7934fa6d506f5ebdbcc435c3a97aceaed922912af8fb93ad8028`
- B: `ff5eac176ce15c719c2027df1fc10fc8286d7382932d9e7c476ccf8c0ad7a095`

Same image as existing worker. Each has 192 GiB memory/swap-total limit,
16 GiB shm, private network, SYS_NICE, one compute device and management
devices. All existing source/model/result mounts are read-only in new workers.
Each has independent writable `/workspace/parallel-worker`, backed by
`/home/admin/kt-r64-migration/round7-parallel/<container-name>`.
No model copies; the underlying model inode is shared. No business-container
changes. `verl` has unrestricted cpuset: these assignments isolate our workers
from each other, not an exclusive host reservation.

Admission snapshot: 906 GiB MemAvailable, zero swap used. PCI 01:00.0 (NPU4)
reports NUMA0; PCI 41:00.0 (NPU6) reports NUMA2. Both devices had no compute
processes. New containers have NOT yet passed device/runtime probes or run
model tests. Existing Uniform runtime-v8 compact request file reached 32 rows;
that count alone is not a completed compact/restart PASS.

## Dispatch update: user-authorized parallel system probes

User subsequently authorized immediate512/1024 system probes in these workers.
They first run their own compact32/restart4 gates; no inherited worker PASS.
Original256 continues. Its parent dispatcher191211 alone was SIGSTOP'ed to
prevent duplicate sequential512/1024. No signal was sent to the active probe
or model. Host coordinator joins verified results before serial DEV timing.

Initial tools-v23 attempt was rejected before model launch by retained
`owned_guard.py` (`GUARD_SCOPE`: new output directory not in old whitelist).
Both failures are preserved under each worker's `failed-guard-scope-v1`.
Tools-v24 adds a Round7 scoped guard retaining the exact generic process guard,
subreaper and owned-descendant cleanup. No production model/kernel change.
Source commit `2ce653c13`; archive SHA256
`427c8e6afca5f0c91054f6b14848388e3bd14c495a45ee7621774a6136daf57e`.
Local suite115 PASS; earlier v23 pure suite112 PASS on A3.

Two detached worker chains were restarted with v24. Host coordinator PID1725839,
log `/home/admin/kt-r64-migration/round7-parallel/coordinator-v1.log`, validates
three thresholds' artifacts, corpus/mask hashes, lifecycle completion/reaping.
Then it runs `dev-threshold-screen-v2` serially on original worker; it never
creates a candidate commit/ack or bypasses held-out entry gates.

## Subsequent dispatch prerequisites

Keep new workers idle during later DEV timing selection. Runner affinity,
NUMA and result paths now support these three explicitly whitelisted topologies.
Preserve C64 as two subpools of32, not a different worker topology. Actual device
execution still needs the new workers' completed gate evidence.

After runtime system gates and candidate commit/ack, parallelize independent
Static HOLDOUT and Dynamic HOLDOUT/quality where their predecessor gates allow.
Do not fit thresholds on held-out outputs. Deferred remains dependent on the
selected scheduler. Formal ABBA stays serial with other owned compute stopped;
parallel timings are not isolated-performance evidence. Keep original host
observer isolation policy intact.
