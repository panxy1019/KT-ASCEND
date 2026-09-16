# Capacity/performance Pareto — NOT_COMPLETE

The observer-OFF capacity queue exited on static-predecessor failure before
launching any of its four timed profiles. No Pareto curve or heavy-profile
throughput optimum can truthfully be published.

Available **diagnostic-only** four-layerR32 window:72samples,no sampling errors.

| Sampled metric | Observed maximum |
|---|---:|
|Container cgroup memory.current|90,998,427,648bytes(~84.75GiB)|
|Sum PSS of owned processes|32,433,692,672bytes(~30.21GiB)|
|Sum RSS of owned processes|36,562,042,880bytes(~34.05GiB)|
|Owned NPU process accounting|28,948MiB|
|Owned process swap|0|

Container limit192GiB. Minimum sampled hostMemAvailable1,028,878,299,136bytes.
RSS sums may double-count shared pages;host cache is not entirely this task;
sampled peaks are lower bounds. These observations do not prove zero leak or
DRAM bandwidth saturation. Loader peaks and steady decode are not interchangeable.

32workers permitted healthy deterministic single/four-layer execution. Current
blocker is numerical,not observed OOM or worker starvation. Planned16/32timing
and final1000forward/stability were not run after the hard failure. Do not claim
32workers optimal,or extrapolate a64worker speedup. NUMA nodes have32physical
cores each;48/64would require a separately controlled cross-NUMA experiment.

CPU wait/host bandwidth Pareto:NOT_MEASURED. Formal TTFT/decode comparison:
NOT_RUN. Incidental serving log throughput is not a benchmark.
