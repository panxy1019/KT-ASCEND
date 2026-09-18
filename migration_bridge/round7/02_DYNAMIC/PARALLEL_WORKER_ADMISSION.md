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

## Dispatch prerequisites

Keep new workers idle during current DEV timing selection. First adapt/test
runner affinity, NUMA arguments and writable result/runtime paths: existing
scripts hardcode 128–191, nodes4/5 and original output paths. Preserve C64 as
two subpools of32, not a different worker topology. Validate device visibility,
source hashes, outputs and topology before dispatch.

After runtime system gates and candidate commit/ack, parallelize independent
Static HOLDOUT and Dynamic HOLDOUT/quality where their predecessor gates allow.
Do not fit thresholds on held-out outputs. Deferred remains dependent on the
selected scheduler. Formal ABBA stays serial with other owned compute stopped;
parallel timings are not isolated-performance evidence. Keep original host
observer isolation policy intact.
