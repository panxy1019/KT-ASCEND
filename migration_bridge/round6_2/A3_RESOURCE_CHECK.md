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

Historical restriction at the time of this check:
`A3_EXECUTION = WAITING_FOR_NPU0_RESOURCE_WINDOW`.

Do not move to NPU1+, silently co-run with another workload, or stop that
workload. Recheck resources and environment before model execution. Local
preregistration/tooling may continue independently; neither corpus freeze nor
V2 preregistration is yet complete, so resource availability alone is not
sufficient to start qualification.

## Subsequent explicit user authorization

The user subsequently authorized the original full execution plan on shared
NPU0, not limited to short smoke. Contention with the existing workload is
accepted. The prior resource-window restriction is therefore superseded:

`A3_SHARED_EXECUTION = AUTHORIZED_WITH_OWN_PROCESS_ONLY_ABORT`.

Never stop, pause, reprioritize or reconfigure the other workload. On inadequate
memory, OOM, timeout or device error, abort only this round's newly launched
tests. No migration to NPU1+ and no change to placement, contract or corpus.

A fresh read-only check found the same external PIDs using 6125 MiB each;
HBM use remained 14%/13%, with utilization 60%/55% in that sample. These are
transient observations, not a guarantee of memory available at launch.

`tools/run_owned_process.py` is an inside-container foreground-command guard.
It creates a new session, records its PID/PGID and log, and terminates only
that group on recognized error messages or a preset deadline. It retains the
leader PID through group cleanup to avoid signaling a recycled identifier.
Commands which detach workers into new sessions are unsupported and must not
be launched with this guard. The guard itself must not wrap SSH/docker exec
and assume remote process ownership follows the local process group.

Memory admission checks and request-level watchdogs remain required; log
matching is supplementary and does not cover all device failure modes.
No arbitrary total-duration value is adopted for the official matrix: the
deadline must be written into the execution manifest before that run.

Local test result: 18/18 including own-process OOM cleanup while an independent
sentinel process remains alive, timeout, normal/nonzero exit and refusal to
overwrite evidence. No A3 model test was started during this authorization
update. Shared-environment performance measurements must be labelled as such;
they are not an isolated hardware speedup claim.
