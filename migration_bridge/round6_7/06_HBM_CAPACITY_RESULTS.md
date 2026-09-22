# Capacity telemetry — results pending

All profiles use one existing container, NPU0, fixed maximum4096KVtokens,
same model/non-routed weights and32CPUworkers. Resident routed BF16 payload
theory is26.8125/13.40625/6.703125/3.3515625GiB forR64/R32/R16/R8.
Actual device usage includes runtime allocations, KV and allocator reservations.

Host read-only sampling every10seconds records NPU0 process memory, device
HBM usage, utilization, per-process RSS/anonymous/file pages/faults, host cache,
MemAvailable/swap and container cgroup accounting. Device usage includes idle
device baseline; report before-model level and loaded/steady levels explicitly.
Sampled peak is not asserted to be an exact transient peak.

The optional idle SGLang/server_info RPC timed out in startup v3; it is labeled
UNAVAILABLE. Resolved CLI config is validated separately and is never presented
as a measured memory counter. Framework allocated/reserved/peak counters are
recorded by the observer; capacity under normal execution comes from host
sampling. Host bandwidth and CPU-wait counters remain null when unmeasured.

Host page cache and summed file-backed RSS can double-count shared pages.
Do not sum these as private working set or infer bandwidth saturation from low
NPUutilization alone. Review cgroup anon/file and faults alongside RSS.
