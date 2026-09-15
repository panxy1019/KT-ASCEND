# A3 resource check — 2026-09-15

Observation timestamp: `2026-09-15T05:53:08Z` (end of read-only check).
This document transcribes the observed command results, not a raw output log.

Dedicated container `kt-r6-deferred-ascend`:

- ID `b91b94ef502933c63247d03ae6150e3c72ce29061c4034d2c8b435b18b6d8a1f`
- running; CPU affinity `0-15`
- `docker top ... -eo pid,args`: only `sleep infinity`, PID 3738367

`npu-smi info -t usages -i 0`:

| Chip on card 0 | HBM capacity MiB | HBM use | NPU utilization |
| --- | ---: | ---: | ---: |
| 0 | 65536 | 14% | 54% |
| 1 | 65536 | 13% | 27% |

`npu-smi info -t proc-mem -i 0`:

| Chip | PID | Command | Process memory MiB |
| --- | ---: | --- | ---: |
| 0 | 1510639 | python3.10 | 6125 |
| 1 | 1510640 | python3.10 | 6125 |

Read-only `/proc/<PID>/cgroup` inspection associates both processes with Docker
container `0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef`,
not the dedicated project container. The observed elapsed time was ~3h43m.
No process was killed, paused or modified. No model server was launched.

## Execution restriction

`A3_EXECUTION = WAITING_FOR_NPU0_RESOURCE_WINDOW`.

Do not move to NPU1+, silently co-run with another workload, or stop that
workload. Recheck resources and environment before model execution. Local
preregistration/tooling may continue independently; neither corpus freeze nor
V2 preregistration is yet complete, so resource availability alone is not
sufficient to start qualification.
