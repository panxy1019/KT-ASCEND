# Resource attribution follow-up — 2026-09-19 21:40 UTC

Static audit of deployed orchestration confirms the gap: admit_q64_host.py
checks >=30seconds pre-launch host paging, owned swap and topology; the frozen
performance loop checks other Docker CPU/NPU use but only records hostvmstat,
without enforcing paging during requests. It also lacks a final post-client
snapshot and per-cgroup/NUMA time series. Thus prelaunch PASS cannot imply
full timed-window resource PASS. All v3results remain resource-limited.

Existing frozen topology is64physicalworkers,CPU128–191,NUMA4/5; each selected
node has132114580KiB (~126GiB) total memory. The384GiB cgroup limit is larger
than the ~252GiB combined physical capacity of those two nodes. This is a
plausible local-reclaim risk despite host-wide free memory, NOT proof of the
observed swap cause. Do not expand cpuset/mems or raise memory budgets silently.

Next bounded diagnostic must preserve model/mask/C64 and use one cold Dynamic
medium replay, not another twelve-cell performance campaign. It must collect:

- Before/after plus interval snapshots bracketing the entire request.
- Owned cgroup memory.current/stat/events/swap.current/pressure/numa_stat.
- NUMA4/5meminfo and vmstat; processRssAnon/RssFile/VmSwap for owned scheduler.
- Hostpswpin/out and pressure deltas; intervalCPU counters rather than ps lifetime%.
- Freeze input256/output64 and admission/source hashes, preserve failed runs.

Classify separately: owned swap/lifecycle defect; NUMA-local reclaim; unrelated
host paging; insufficient attribution. Global activity cannot be relabeled as
owned-model failure, nor dismissed from strict performance. Diagnostics are
not timed qualification. No swapoff/drop_caches/other workload changes.

Round8 product system,quality and boundedStaticstability results remain intact.
DefaultStatic and DynamicOFF remain conservative; formalperformancegate and
finalRound8closure are still open. No Round9fullconversion/NPUbringup yet.
