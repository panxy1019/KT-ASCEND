# Read-only host performance contention observer

Source commit: `00d321211f923865c2e61bc834ca2561fb5c54cd`.
Script SHA256:
`92ab38ceafe86ff494b0e1d3c9af246f1db69b8377f58e70704d757e3e6b3e45`.

Host-only source: `/tmp/round7-host-resource-observer-v1.py`.
Owned output mapped into the existing container:
`/workspace/c64-dev/round7/host-resource-monitor-v1/`.
It samples `/proc`, `docker top` for owned PIDs, host memory and read-only
`npu-smi info -t proc-mem` for all eight dies. It never stops a task, launches a
model or changes host libraries/containers. It excludes owned model PIDs from
outside CPU/NPU contention checks.

Criterion inherited from retained performance admission: no other model or
controller/NPU process, outside CPU total<=2 cores and per-process<=1 core,
no swap use, MemAvailable>=max(64GiB,20%RAM), at least60 seconds continuous
quiet. It records WAIT/EXCLUSIVE rather than making another task stop.

Samples bind to actual request start/end time. Missing/stale coverage or a
contended sample rejects formal eligibility. This is a **contention proxy**, not
direct DRAM-bandwidth measurement or permission to claim host bandwidth.
Existing release-v6 is unchanged: the independent host log can be checked
against its future performance windows after collection. Future tools accept
`--host-observer .../samples.jsonl` explicitly. No observer remains in the timed
model path.

The first observer launch occurred before SCP finished and exited with missing
script; its stderr is preserved at `/tmp/round7-host-resource-observer-v1.log`.
After transfer completion and source-hash equality, a second launch uses
`/tmp/round7-host-resource-observer-v1-start2.log`. This orchestration error did
not launch NPU work, create evidence samples or affect the healthy HOLDOUT run.
Verify fresh samples before using them as actual host evidence.

Local suite: 33 passing tests, including complete/contended/stale performance
window checks and parsing the target host's actual npu-smi process PID format.
This is local tooling verification; EXCLUSIVE status must come from A3 samples.

## Actual retained V2 admission alignment

V1 sampled correctly but stays WAIT because the host has262144 bytes of
historical swap allocation. Read-only inspection found about912GiB available
RAM, owned cgroup `memory.swap.current=0`, all observed owned processes
`VmSwap=0`, and repeated host counters `pswpin=0 / pswpout=1` unchanged.

The actual retained serving admission already uses
`CGROUP_ZERO_SWAP_HOST_NO_ACTIVE_SWAP_V2` in
`/workspace/c64-dev/tools-v16/migration_bridge/round6_7_c64/tools/serial_session.py`,
lines15–33, SHA256
`b1015d55845f629b816389b32bfa5588faeace76a73009d1adfdbb4c2c0902df`.
It requires owned cgroup zero swap and no active host swap I/O when unrelated
historical swap allocation exists. The new observer initially copied a stale
strict-zero host lease criterion instead of this actual qualification policy.
That mismatch is repaired before any Round7 performance measurement, not fitted
to a throughput/quality result. No swapoff/sysctl or other workload mutation.

V2 preserves the memory reserve and CPU/NPU quiet requirements, requires zero
owned cgroup swap, and continuously rejects any host swap counter change from
its startup snapshot (stronger temporal coverage than the admission two-second
check). MemAvailable, historical host swap bytes, actual current counters,
cgroup swap bytes and source hash are included in every observation. VmSwap
does not cover shmem swap; the cgroup/global checks are therefore retained,
rather than relying on process VmSwap alone. Primary field definitions:
[Linux kernel proc documentation](https://docs.kernel.org/filesystems/proc.html).

V2 source commit `65061e6646c5266fa2bd199fe2b05e7feee2f094`, SHA256
`86066a50bc0bc02dba7eb0fa2bf8b0687fc62091333ac00c11ef954a99e815c1`;
host source `/tmp/round7-host-resource-observer-v2.py`, actual samples under
`/workspace/c64-dev/round7/host-resource-monitor-v2/`, stderr
`/tmp/round7-host-resource-observer-v2.log`. Source hash matched after SCP
completion before launch. The obsolete owned V1 observer PID283141 was
command-identity checked and SIGTERM stopped; its samples/logs remain preserved.
No model/test process or unrelated task was stopped.

Local suite34 tests pass, including acceptance of inactive unrelated historical
swap and rejection of owned swap or new host swap I/O. A3 EXCLUSIVE status still
requires fresh V2 samples and60-second quiet interval; do not fabricate it.
