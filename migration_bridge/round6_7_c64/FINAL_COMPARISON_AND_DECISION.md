# C64 final product-screen comparison and decision

Date: 2026-09-17. Scope: DeepSeek-V2-Lite, all26 MoE layers, batch1,
NPU TP1, LLAMAFILE F32, CPU64 workers across NUMA4/5. Deferred/Dynamic/Graph OFF.

## Outcome

**Recommend R8-C64 as the next Static Frequency development baseline under the
user-approved abbreviated product screen. Keep R16-C64 as fallback.**
Both pass system canary and CMMLU256/-5pp. R8 quality-request latency remains
practical for this evaluation workload. No need to start R32/R48 fallback tests.
This is not full qualification of the original C64 runbook: full isolated3-block
performance was explicitly cancelled, and general decode/stability is unproven.

## Quality

| Profile | Correct/256 | Accuracy | Delta vs fresh R64 | Invalid |
|---|---:|---:|---:|---:|
| Fresh R64 |174|67.96875%|0pp|0|
| R16-C32 |173|67.578125%|-0.390625pp|0|
| R16-C64 |174|67.96875%|0pp|0|
| R8-C64 |174|67.96875%|0pp|0|

| Candidate vs reference | Regressed | Improved | Delta |
|---|---:|---:|---:|
| R16-C64 vs R16-C32 |2|3|+0.390625pp|
| R8-C64 vs fresh R64 |2|2|0pp|
| R8-C64 vs R16-C64 |1|1|0pp|

Equal totals do not mean identical answers. Each C64 profile versus R64 has
paired-bootstrap95% CI [-1.5625,+1.5625]pp; no population noninferiority claim.
The quality dataset/evaluator are the frozen256-item screen, not broad model
certification. Original low-offload numerical failures are not retroactively PASS.

## System/operator evidence

- Operator matrix52/52 PASS;156 real decode and64 prefill-shape cases per TP
  coverage,10 repeats, maximum relL2 .0017353741484431432 < .01.
- Original C32 parity156/156 exact;26-layer native weight reconstruction PASS.
- R16 and R8 canary32 compact requests, clean restart, observer parity PASS.
- R8 extra local runs8/8 PASS,24 C64 real cases, maximum relL2 .0017083658550991103.
- R8 short smoke16/16 completes; both quality service lifecycles COMPLETE.
- Partial R8-C32 remains STOPPED_BY_RESOURCE_PRIORITY, not quality FAIL.

## Performance interpretation

Retain only the first complete ABBA: two observations/mode/shape,20 measurements.
Median C32/C64 latency by input256,384,512,640,768:

| Input tokens | C32 seconds | C64 seconds |
|---:|---:|---:|
|256|100.891|2.211|
|384|153.241|2.497|
|512|201.877|2.906|
|640|253.719|3.322|
|768|303.137|3.724|

Median of paired shape latency ratios .01439275. This reflects grouped-prefill
path, F32 tail-safe704+704 partition, NUMA layout and worker changes together,
NOT pure core-count scaling. Concurrent workloads and later stop events are
recorded; no isolated benchmark claim or decode tok/s inferred.

Actual CMMLU request totals: R16-C64 607.524s, R8-C64 644.328s (about6.1% more).
Median per-item request:2.308s versus2.426s. Service lifecycle including load and
cleanup:757.974s versus794.538s. These are descriptive sequential workload data,
not randomized replicated R16/R8 performance comparisons.

R8 selection prioritizes capacity with acceptable observed evaluation latency;
there is no invented absolute throughput gate. Routed BF16 payload theoretical:
R16=6.703125GiB, R8=3.3515625GiB. These are NOT measured total HBM; attention,
shared experts, KV/cache/workspace remain. Full HBM Pareto remains incomplete.

## Memory and provenance caveats

First-ABBA memory audit PASS, no sustained major-fault storm. Median summed
process RSS C32 ~60.52GiB, C64 ~102.49GiB; C64 anonymous ~60.33GiB. RSS sums are
not PSS and cannot prove physical footprint or absence of long-term leaks.
Later host had256KiB swap residency while this container had0 and no OOM.
Admission V2 explicitly requires container zero swap, sufficient reserve, and
no observed host swap activity over2s when global swap residency exists. Original
SWAP_USED failure retained; no host swapoff or sysctl changes.

Model revision604d5664dddd88a0433dbae533b7fe9472482de0.
Actual source parent e3084d309d6f39b8a9f7185d36ac27e4b77526d7,
child eaa58ccea4ee372b8b207dd16f398eafe3da37ff plus recorded overlays.
Native a5497fa085d704d8f67484b3936eaf448080f5d445899d88e4e76bea4305f077.
CMMLU manifest bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04.
R16 profile27433a83062c6eda0fe9b957c772ff96c01e5069fabe970ffa892d595efe3e0c.
R8 profile f2ae533c52318cb5015ba6396c2329865abdb157c2e774061eb01557c4d40c0d.

Machine evidence: evidence/final-screen/FINAL_QUALITY_COMPARISON.json,
per-profile quality.json, QUALITY_FREEZE.json, summaries, operator audit,
canaries and memory audit. Remote full raw logs remain under /workspace/c64-dev.

## Final states

ABBREVIATED_C64_PRODUCT_SCREEN = COMPLETE
R16_C64_SYSTEM_AND_CMMLU_SCREEN = PASS
R8_C64_SYSTEM_AND_CMMLU_SCREEN = PASS
SELECTED_DEVELOPMENT_BASELINE = R8_C64
FALLBACK = R16_C64
FORMAL_ISOLATED_PERFORMANCE = NOT_COMPLETED_BY_USER_SCOPE_CHANGE
FULL_ORIGINAL_C64_RUNBOOK_QUALIFICATION = NOT_CLAIMED
ROUND7_DYNAMIC = NOT_STARTED

Next: freeze this R8 uniform mask as comparator, then separate Static Frequency
SCHED_DEV/HOLDOUT study. Do not choose experts using these CMMLU outcomes.
