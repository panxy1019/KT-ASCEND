# Round6.5 — Final blocked qualification summary

Date2026-09-16. **The successful full round was not achieved.** Execution ended
at the preregistered hard numerical gate after bounded reproducibility and
first-failure investigation. No approval request is pending and no later-stage
experiment is running on an unqualified profile.

## Frozen scope

- Branch:`codex/round6-5-official-style-offload-alignment`.
- Local base:`c74931235c60d2e899e3acd9fc9277b06962920a`.
- Actual deployed parent:`e3084d309d6f39b8a9f7185d36ac27e4b77526d7`.
- Actual deployed child:`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`.
- Runtime is that lineage plus recorded overlays,not a clean localHEAD deployment.
- DeepSeek-V2-Lite:`604d5664dddd88a0433dbae533b7fe9472482de0`.
- New fullGGUF57,579,411,648bytes;
  SHA`20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2`.
- Target110.129.0.22,910B3NPU0,container`kt-r65-static-offload`,32workers,
  CPU0–31,192GiBcontainer limit,TP1,batch1,Graph/Dynamic/DeferredOFF.
- Frozen numerical gate:B_pair2.1875,K32,expert relative-L2≤0.01.
- No production source change,upstream merge,quantization change or expert reselection.

## Gate matrix

| Gate | Result | Evidence |
|---|---|---|
|Capacity preflight|PASS for planned capacity estimate|01_CAPACITY_PREFLIGHT.md|
|Complete source/GGUF identity|4992/4992PASS|evidence/r65-assets-v1/identity.json|
|Frozen uniform masks|12profiles,unit-tested|02_STATIC_MASK_CONTRACT.md|
|R64 oracle|28DET/restart requests;512numeric positions captured|evidence/r65-r64-det-v3; r65-numeric-r64-v1|
|SingleR32|compact PASS,maxpairwise0.625|03_SINGLE_LAYER_RATIO_PROBE.md|
|SingleR16|compact PASS,maxpairwise0.6875|03_SINGLE_LAYER_RATIO_PROBE.md|
|SingleR8|compact PASS,maxpairwise0.6875;physical hooks verified|03_SINGLE_LAYER_RATIO_PROBE.md|
|FourR32same-path/prefix/restart|28/28PASS|evidence/r65-scale-matrix-v2/four-R32/det/report.json|
|FourR32numerical|FAIL:one position,27pair overflows,max9.59375|evidence/r65-scale-matrix-v2/four-R32/pairwise.json|
|Independent logits repeat|512/512vectors byte-exact|evidence/r65-four-R32-localization-v1/full-repeat-comparison.json|
|Failure transfer/merge audit|260/260exact;physical ownership PASS|evidence/r65-four-R32-localization-v1/transfer-audit.json|
|Implicated expert safety|5/5PASS;10repeats each|evidence/r65-failure-experts-v1/report.json|
|FourR16/R8 and all-MoE profiles|NOT_RUN_PREDECESSOR_FAILURE|05_ALL_LAYER_RATIO_SWEEP.md|
|Capacity/throughput Pareto|NOT_COMPLETE|06_HBM_HOST_PERFORMANCE_PARETO.md|
|Static Frequency|inputs/tools frozen,experiment NOT_RUN|07_STATIC_FREQUENCY_PLACEMENT.md|
|Round7baseline|NOT_FROZEN|08_ROUND7_BASELINE_SELECTION.md|
|Bounded Deferred probe/quality|NOT_RUN_PREDECESSOR_FAILURE|09_DEFERRED_HEAVY_PROFILE_PROBE.md|

The physical-hook evidence was added during singleR8;earlier singleR32/R16
captures do not retroactively gain that instrumentation. FourR32localization
validates actual32resident tensors at1,9,17,26,but does not qualify its failed
numerical gate.

## Meaning of the failure

`h2_en_01`,zero-basedtoken54:baseline andHybrid both choose82. All259stable
positions exact and253ambiguous positions inside the allowed set. Nevertheless
the numerical envelope fails,and the protocol explicitly requires zero overflow.

The same failure reproduces across processes. Trace observer parity is exact.
First router-set change atLayer17 swaps62for29;later layer activations amplify.
Individual implicated experts remain within their same-input safety envelope.
See04A_FIRST_FAILURE_INVESTIGATION.md and10_FINAL_ADR.md for evidence boundaries.

This is not demonstrated nondeterminism,OOM,transfer corruption or downstream
quality collapse. It also cannot be dismissed as PASS simply because top1matches.

## CPU sizing

32workers ran the bounded single/four-layer workloads deterministically. The
observed failure is numerical,not evidence that32cores are insufficient.
Each NUMA node has32physical cores;48/64would change locality as well as count.
Strict memory binding was not established(set_mempolicy EPERM),and NPU0 is
PCI-local toNUMA6 while the frozen affinity isNUMA0. No tuning claim is made.
Planned16/32controlled timing was not completed before the hard stop;optimality
and full-model bandwidth limits remain unknown.

## Final

```
ROUND6_5_HEAVY_OFFLOAD_ALIGNMENT = BLOCKED_HEAVY_OFFLOAD_RUNTIME
BLOCKER = STATIC_NUMERICAL_SCALE_BLOCKER_AT_FOUR_R32
MINIMUM_ALL_MOE_R32_SUCCESS = NOT_ACHIEVED
ROUND7 = NOT_STARTED
DEFAULT_DEFERRED = OFF
```

Historical Round5/6.4results are unchanged. No A3-verified success tag is created
for this blocked round. New commits contain tooling,documents and evidence only;
the pre-existing llama.cpp modification is preserved and excluded.

## Resource release and evidence preservation

All owned model processes reaped;11final lifecycle guards audited PASS.
After confirming the container had only its idle PID1,the dedicated container
was stopped and the owned resource sampler terminated. NPU0 then reported no
process. No container/volume/model was deleted and no business task stopped.
825metadata/first-failure artifacts archived locally with matchingSHA256;
large full raw matrices and GGUF remain in the preserved server artifact volume.
