# Round6.6 V1 summary — blocked quality screen

Target:110.129.0.22,Ascend910B3,NPU0,CPU0–31,32workers,one pool.
Actual production parent:e3084d309d6f39b8a9f7185d36ac27e4b77526d7.
Actual child:eaa58ccea4ee372b8b207dd16f398eafe3da37ff,with frozen overlays.
Model:DeepSeek-V2-Lite604d5664dddd88a0433dbae533b7fe9472482de0.
Profile:four-R32,odd CPU IDs in layers1/9/17/26;Deferred/Dynamic/Graph OFF.

| Gate | Result |
|---|---|
|External reference-lane source research|Complete,primary sources with limitations|
|Three-way expert decomposition|608cases PASS,128expert instances|
|Production/test-stage parity|608/608 exact|
|Expanded real prefill|11/16/64/128 rows covered|
|Same-path/prefix and clean restart|52+4 requests PASS|
|QD6|768positions audited,system controls PASS|
|Candidate before HD6|Frozen43a0bd582a4b126feb7ed851b85e1e94c1de98131700547eead4615d3e21967a|
|HD6|768positions,system controls PASS,old-bound3overflows retained|
|Post-HD tail operators|52/52 PASS,original0.01|
|Free generation|12/12 observer parity;10exact cross-path trajectories,2ambiguous first differences|
|Quality|GSM8K3→2/32;CMMLU145→142/256;FAIL frozen zero-observed-loss screen|
|Raw scoring and source audit|288/288 pairs rechecked,no runtime/asset/tool drift|
|Production arithmetic change|NO|
|Final|BLOCKED_QUALITY_SCREEN|

No claim of population noninferiority,all-MoE readiness,Dynamic readiness or
Deferred speedup. Old Round6.5 remains blocked. See12_QUALITY_ADR.md for the
failure interpretation and separately identified autonomous followup.
