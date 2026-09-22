# ADR — Round6.5 blocked at static numerical scale

Date:2026-09-16. Decision:**STOP_AT_FROZEN_NUMERICAL_GATE**.
This closes the bounded investigation,not the successful heavy-offload roadmap.

## Evidence and decision

Full26MoE-layer F32GGUF exists and4992projections exactly preserve sourceBF16
values. SingleLayer17R32/R16/R8 compact determinism,matched-history numerical
and controlled expert tests pass. Four-layerR32 determinism/restart passes,
but its512position sweep has one position with pairwise9.59375>2.1875.
27candidate pairs overflow. Stable259/259 and ambiguity253/253still pass.

Independent fresh-process full replay reproduces all512logit vectors exactly.
Debug observerON/OFF logits are exact for the failing prompt in both paths.
260CPU-layer forward transfers/merges are exact;physical resident/mapping
checks pass. Five same-operand CPU-vs-NPU expert probes pass0.01:

| Layer/expert | relative-L2 |
|---|---:|
|1/E25|0.0039275593|
|9/E41|0.0037111187|
|17/E41|0.0036959657|
|26/E63|0.0009067905|
|17/E29|0.0036334734|

Each has10identical CPU repeats and finite outputs. These probes test selected
implicated experts,not every expert/shape. No universal absence-of-bugs claim.

At the failure,accumulated hidden differences precede the first router-set
change atLayer17(E62→E29). Layer17MLP relative-L2 grows from input0.04145to
output0.21778;later layers amplify further. This is direct evidence of
reproducible layerwise amplification with router-set changes. It is **not**
proof of a specific CPU reduction tree as the unique cause,and is not a
measured downstream quality regression.

No identified source bug supports a surgical production fix. Changing router
selection to imitate this one baseline,using NPU for CPU-owned experts,
removing the prompt,reducing CPU offload or raisingB_pair would violate the
approved experiment. Those are not repairs authorized by autonomous execution.

## Official comparison

The [fixed official Ascend guide](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md)
supports resident/DRAM tiering and separately reports accuracy validation for
DSV4W8A8/MXFP4. It does not claim that arbitrary DSV2BF16/F32 static masks obey
our inherited2.1875pairwise bound. No directly applicable fix was established
by that reference. Official performance cannot be transplanted to this profile.

## Rejected shortcuts

- Do not mark PASS merely because the failing position's token82still matches.
- Do not use Deferred's approved2ppapproximation screen to override static gates.
- Do not relabel missing Pareto/frequency/quality evidence as completed.
- Do not infer CPU bandwidth saturation from low NPU utilization.
- Do not add cores or change NUMA placement as a purported numerical fix.

## Final states

```
ROUND6_5_HEAVY_OFFLOAD_ALIGNMENT = BLOCKED_HEAVY_OFFLOAD_RUNTIME
BLOCKER_SUBCLASS = STATIC_NUMERICAL_SCALE_BLOCKER
FIRST_BLOCKED_PROFILE = four-R32
RESIDENT_RATIO_PARETO = NOT_COMPLETE
ROUND7_BASELINE = NOT_FROZEN
ROUND7 = NOT_STARTED
DEFERRED_HEAVY_PROFILE_PROBE = NOT_RUN_PREDECESSOR_FAILURE
ROUND6_4_CLASSIFICATION = IMPLEMENTED_NO_SPEEDUP
DEFAULT_DEFERRED = OFF
```

## Next engineering boundary

Retain this exact first-failure fixture and investigate heavy-static numerical
scaling independently before Dynamic. A useful next bounded study separates
same-input arithmetic error,KV-history propagation and router-topK sensitivity;
counterfactual route experiments must remain diagnostics,never production
token-specific fixes. If a real implementation defect is found,repair it and
rerun the original frozen gate. If no defect is found,a genuinely new numerical
acceptance study would require its own preregistered protocol and independent
holdout;this ADR does not approve or implement a relaxed contract.

32physical cores remain a viable execution starting point,not a proven optimal
heavy-offload allocation. No hardware purchase or cross-NUMA expansion is
justified by the current blocker. The16/32scaling question remains unmeasured.
