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
