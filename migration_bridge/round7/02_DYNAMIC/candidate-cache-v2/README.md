# Runtime-v8 DEV threshold freeze

2026-09-18. New cache candidate; previous runtime-v7 DEV results remain historical.
All three runtime-v8 system probes PASS before selection. Full DEV screen48/48,
12 fixedDEV prompts per configuration,64 generated tokens each, four separate
processes COMPLETE/reaped. No parallel model inference during DEV selection.

| Threshold | Median matched E2E ratio vs Static | Short | Medium | Long |
|---|---:|---:|---:|---:|
|256|1.1151016024|1.0236575026|1.8243872940|1.1045468140|
|512|1.0415611707|1.0015614214|2.1020842874|1.0874033219|
|1024|1.0086745580|1.0107725588|0.9902471368|1.1872696227|

Frozen selector: minimum median matched-case E2E ratio; ties prefer larger
threshold. Selected1024. This is NOT a speedup claim, not formal ABBA and not
held-out qualification. Cold first-trigger costs remain represented in DEV.

Independent local audit reconstructed every ratio from raw response timestamps,
checked exact request IDs/counts,64-token outputs, lifecycle/reaping, protocol
and candidate-results SHA, and independently reproduced selected1024. Quality
and held-out outputs were not used. No sample/threshold-rule change.

Raw archive `../../raw/round7-cache-dev-screen-v2.tar.gz` (repository round7/raw):
SHA256 `e98674efbd666c9d2fc9345739ecaeec35798775a068c520f7d4d69b06badf27`.
Selected JSON SHA256:
`bca98065f7e852961d5d4c3273861812865bb9597c6cf18162fe13d203b2e98a`.
The exact frozen Uniform profile is included; code/backend/weights stay frozen.

Next: commit these exact bytes, verify git-object identity, create bound ACK,
then parallel independent Static HOLDOUT, Dynamic HOLDOUT and Dynamic quality.
Quality uses completed Dynamic1024 system gate and explicitly reused unchanged
CPU-operator evidence; it does not assert HOLDOUT has already passed. Failed
held-out system gates still reject Dynamic regardless of a quality score.
