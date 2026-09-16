# Round6.6A — quality-failure diagnosis result

State: `DIAGNOSIS_COMPLETE_NO_NEW_IMPLEMENTATION_DEFECT_FOUND`.
This is not a quality PASS and does not waive Round6.6 V1's failed screen.

All8 correctness-discordant cases were frozen,including2 gains and6 losses.
Fresh All-NPU/Hybrid matched-history captures reproduce the original observer-OFF
quality choices through first divergence. Transfer/merge/ownership audits pass.

| Case | Outcome | NPU pair margin | Hybrid pair margin |
|---|---|---:|---:|
|GSM8K01036|loss|0|−0.125|
|marketing00126|loss|0.1875|0|
|genetics00101|loss|0.0625|0|
|virology00140|loss|0.0625|0|
|public_relations00126|gain|0.0625|0|
|high_school_chemistry00108|gain|0.0625|−0.125|
|education00086|loss|0.25|−0.125|
|clinical_knowledge00060|loss|0.0625|0|

These are margins between the original two chosen tokens,not a newly fitted
acceptance bound. Small margins do not excuse incorrect task answers.

## Complete mixed-partial replay

For each case's first-divergence forward and all4 Hybrid layers,fresh installed
LLAMAFILE execution and32-resident Ascend GMM execution received the exact captured
hidden rows,global routing IDs and weights. The complete prefill tensor was used,
not only its final row. CPU buffer capacity2048 matches serving.

-32/32 CPU partials: exact versus production.
-32/32 NPU partials: exact versus production.
-32/32 FP32 sums plus visible final casts: exact versus production.
-10 repeated outputs per backend/case: exact,finite.
-Observed row lengths:1,67,75,86,94,110,113,134.
-Owned replay and capture guards:EXIT_ZERO,all owned descendants reaped.

This substantially strengthens the existing arithmetic-sensitivity explanation.
It does not prove absence of defects for every possible input or justify a
speculative production arithmetic rewrite. No production fix is introduced.

## Next bounded investigation

Round6.6B separately preregisters a base-model few-shot/letter-likelihood protocol
on independently selected evaluation IDs. Its criterion remains zero observed
net loss per endpoint. It cannot retroactively repair V1 or the old B_pair failure.
No repeated reseeding after an unfavorable result is permitted.

Target execution was on110.129.0.22 Ascend910B3,not a claim of new execution on the
historical A3 hardware. Deferred/Dynamic remainOFF;all-MoE remains unqualified.
