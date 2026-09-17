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
## Unchanged release-v6 timing: conservative post-audit adapter

Future `tools/audit_performance_windows.py` audits all36 completed ABBA cells
without changing or restarting healthy v6. Run it on A3 against original files
before archiving: v6 retained input wallclock timestamps but not result end
wallclocks. The original measured output JSON file mtime conservatively bounds
completion, including serialization after timing; it is not a latency denominator.
Do not use copied/extracted file mtimes as original measurements.

For each cell, the adapter verifies request/response IDs, raw summary equality,
closed/reaped lifecycle, timestamp containment and wallclock/monotonic duration
consistency. It rejects overlapping/reordered actual intervals and incomplete
or reordered36-cell campaigns. It snapshots the existing observer samples for
the bounded campaign, requires their frozen source SHA and independently audits
every measured interval. Missing/stale/contended windows remain pending, never
manufactured as formal isolated performance. No future sample or single latest
EXCLUSIVE value substitutes for the actual measurement windows.

Only a complete campaign may be analyzed. Suggested later A3 invocation through
the separately deployed future tools release, not active v6:

```
audit_performance_windows.py \
  --root /workspace/c64-dev/round7/static-phase-v1/uniform-vs-frequency-performance \
  --host-samples /workspace/c64-dev/round7/host-resource-monitor-v2/samples.jsonl \
  --observer-source-sha256 86066a50bc0bc02dba7eb0fa2bf8b0687fc62091333ac00c11ef954a99e815c1 \
  --output /workspace/c64-dev/round7/static-performance-window-audit-v1
```

This tool is LOCAL_VERIFIED only. Original A3 first-cell timestamps were inspected:
input1789654548.3274822, original output mtime1789654678.578983,
monotonic request129.96785391692538s; serialization/clock slack~0.283647s.
That timestamp audit is not a completed campaign or performance conclusion.
54 local tests pass, including full synthetic36-cell independent-window audit,
timestamp/identity/lifecycle/reordering failures. Actual isolated eligibility
must await complete raw timing and sample audit.

Separately staged release-v9 source commit
`bb3c59eced61670fd23dcbd6a6e722030d548bdb`, archive SHA
`724534ee6705dd7b271bf15a61e45896f34402cb31a818657d6992b60a214e9e`.
Host/container archive hashes verified after SCP completion. A3 read-only
partial smoke used original first4/36 closed cells: each interval EXCLUSIVE
with15 independent samples, serialization slack0.2748..0.2837s. Observer source
identity matched frozen V2. This validates adapter feasibility only; the complete
campaign, its nonoverlapping36-cell audit and Static selection are still pending.
# Future V3 ownership attribution — prepared, not active

2026-09-17. Current active observer remains V2; current model driver/tools-v16
are unchanged. V3 sourceSHA
`9e92b9cd8981a104277a851d7e45883ec5a1239935d7d0ad8a1498b70cd4c063`.
The former two-step `docker top` then `/proc` snapshot can disagree when owned
children appear/disappear between observations. That is a CODE_INSPECTED
attribution risk, not proven attribution of the old six blocked Static windows:
V2 did not retain outside per-process records sufficient to settle their cause.
All old whole-campaign formal claims remain PENDING_HOST_ISOLATION.

For future observations only, classify each process by its actual unified
cgroup membership in the snapshot, scoped to the exact owned container cgroup
or descendants. CPU deltas also bind `/proc/stat` process start identity to avoid
PID-reuse misattribution. Retain outside process executable/PID/cgroup/CPU deltas
without serializing complete possibly-sensitive command lines. No process is
migrated, stopped or resource-tuned by the observer. Scope/CPU limits/60S quiet/
memory reserve/zero-container-swap/no-active-host-swap rules remain unchanged.
The [kernel cgroup-v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)
defines unified membership and fork inheritance; the [proc documentation](https://docs.kernel.org/filesystems/proc.html)
defines CPU/lifetime fields. Root cgroup is accepted for outside host processes,
but forbidden as the owned container scope. Unreadable ownership is not silently
promoted to exclusive. This remains a read-only contention proxy, not exhaustive
accounting of short-lived tasks or direct DRAM bandwidth measurement.

Three targeted ownership regressions pass; full suite97/97 LOCAL_VERIFIED. V3
still needs independent deployment/hash verification/A3 smoke and fresh complete
performance windows before use. Never replace a healthy live model release or
reclassify old V2 samples using V3. Launch it only as an independently versioned
host observer; stop only the identity-verified obsolete owned observer, not tests
or business workloads. Current active samples remain host-resource-monitor-v2.

## V3 independent A3 deployment — future windows only

2026-09-17 after static campaign/model children reaped. Tools-v17 A3 pure suite
99/99PASS includes the V3 snapshot-cgroup/start-ID regressions. Host source
`/tmp/round7-host-resource-observer-v3.py` hash matches the frozen9e92b9... above.
Actual new samples at `host-resource-monitor-v3` carry that hash,
`V3_ACTUAL_SNAPSHOT_CGROUP_AND_PROCESS_START_ID`, the exact owned docker cgroup
and outside per-process CPU deltas. The observer reachedEXCLUSIVE, blockers[].
Only obsolete ownedV2 PID287781 was SIGTERM stopped after exact cmdline/output/
sourcehash verification and freshV3 samples. Original V2 logs/samples remain.
No model driver or business task was stopped. Use V3 for future formal windows;
one exclusive sample is not a complete performance qualification and no old
Static contaminated window is reclassified.
