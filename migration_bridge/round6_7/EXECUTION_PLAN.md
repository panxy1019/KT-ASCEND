# Round6.7 serial execution and pre-registered decisions

1. Freeze source/asset provenance, exact CMMLU bytes, compact inputs, profiles,
   runner hashes, performance workload and acceptance policy before outputs.
2. R64 then R32/R16/R8, one model process at a time in the current container.
3. Each profile: 32 observer-OFF compact requests, four clean-restart requests,
   four observer-ON requests with exact parity and physical-layout/route audit.
4. Hybrid local operator sampling: E1 and E63 in each of 26 layers, each on
   three actual decode operands at chronological calls 1/4/7 of the first
   diagnostic prompt (early/middle/late). Production CPU vs the retained Ascend
   grouped-MoE reference, relL2 <= .01, finite, three repeats byte-identical.
   Real-input controlled NPU-owned routes must produce exact-zero CPU partial.
   R64 has no CPU experts: CPU sampling/control is N/A, not a fabricated PASS.
5. Only SYSTEM PASS profiles run the same frozen 5-shot CMMLU256 letter-
   likelihood evaluation. No retries with changed items/masks. Request timeout
   3600s is a resource guard, not a scoring change. Failed requests fail the
   system screen, not an incorrect answer. Fresh R64 is mandatory.
6. Product PASS: `100*(correct_R-correct_R64)/256 >= -5`, invalid=0.
   Therefore at most 12 fewer correct. Correctness-pair diagnostics and paired
   bootstrap CI are reported; this fixed small sample does not establish a
   population-level non-inferiority claim.
7. Accepted profiles only: observer-OFF performance, same container/device/
   CPU set, serial sessions. Two short input shapes, 2 warmups + 3 timed
   32-token greedy requests each. Client streaming TTFT/decode timing, identical
   KV capacity, host/device telemetry. No throughput hard floor is invented.
8. Select a capacity/throughput Pareto candidate; do not automatically equate
   lowest ratio with best usable product. Static Frequency is a subsequent
   independent study; Round7 migration does not start in this experiment.

## Compact protocol

Four frozen S6 families: English, Chinese, math, JSON. Per family: one 16-token
anchor, five 8-token repeats, one 4-token and one 2-token prefix = 8 requests;
total32. Prefixes must equal anchor exactly. Restart repeats four 16-token
anchors. Instrumented restart runs each prompt for8 tokens and must match the
unobserved anchor prefix. Temperature0, ignore_eos=true for compact only.
Quality uses the previously frozen base-model CMMLU evaluator, ignore_eos=false.

## Capacity and safety

No admission to a session when swap used >0 or MemAvailable below
max(64GiB,20%hostRAM). Existing container192GiB limit unchanged. All owned
model descendants are reaped between sessions; no stop-all, broad kill or
business-container actions. A system failure stops that profile, not others.
R64/reference or source/asset provenance failure stops the entire study.
A quality failure excludes only that profile from performance/selection.

HBM: host read-only sampler at10s, before load/after load/warmup/steady/peak;
framework allocated/reserved/peak snapshots in diagnostic audit. The deployed
idle server_info RPC timed out before requests in v3 and is explicitly
UNAVAILABLE in v4; no configured number is called a measured counter.
Resolved launch argv proves TP1/batch1/Graph OFF/fixed KV configuration.
Host telemetry: MemAvailable/cache/swap, cgroup
anon/file/events, process RssAnon/RssFile/PSS/faults/NUMA. Shared cache is not
called process-private RAM. Host bandwidth/CPU wait unavailable counters are
null; do not claim measured saturation from CPU utilization alone.

No R48/R4/R0, new kernel, additional cores or Deferred experiments are started
automatically. If all tested Hybrid profiles fail product/system gates, close
with a specific blocker and no Dynamic baseline.
