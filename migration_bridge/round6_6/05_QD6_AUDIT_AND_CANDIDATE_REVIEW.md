# QD6 audit and conditional candidate review

Target-executed; no HD6/quality outputs used. Frozen independent QD6:12prompts×64target positions, plus12lookahead forwards per path. All-NPU and four-R32 each completed780 audited forwards. Actual source/extension91files and repository states remain byte-identical to B0.

## Exact observed system checks

All3120 CPU-layer forwards passed hidden/IDs/weights D2H, CPU output H2D, Deferred-OFF, finite and FP32 merge checks. Physical ownership/layout check passed26/26MoE layers. All128 CPU expert instances were naturally exercised. CPU route events:12680; exact no-hit rows:55. These are observed checks plus retained source/weight/scaling/shared-ownership proofs, not an assertion that different hidden operands must select identical router IDs.

## Distribution

| Metric |p50|p95|p99|max|
|---|---:|---:|---:|---:|
|Full-logit max absolute|0.375|0.7703125|1.145625|2.387939453125|
|Full-logit mean absolute|0.0546315|0.1348013|0.2003200|0.6045697|
|Full-logit relativeL2|0.0122452|0.0293601|0.0431439|0.1321624|
|Candidate-union pairwise distortion|0.125|0.375|0.625|1.3125|

Top1 agreement761/768. Seven differing positions are historical ambiguous/member positions; no historical stable flip or overflow occurs in this new QD. Do not use this fact to retroactively promote the original failing workload. Do not turn1.3125 or0.1321624 into a new threshold.

544/768 positions have at least one cross-path router-set change across26layers. Per-layer changes range from0 atL1 to101 atL20. The count is not the number of failed model outputs; routing discontinuities can be locally amplified or later suppressed. The largest full-logit relativeL2 is atqd6-09/token48, with identical top1 and only0.25 pairwise distortion. Broad full-vector error and candidate ordering disturbance are distinct measurements; neither proves downstream quality.

## Review decision

Accept this QD interpretation for independent held-out validation **conditional on** completing fresh same-process ten-repeat/prefix and clean-restart gates. B0 and system audit show no new implementation anomaly. Preserve model-level distributions as diagnostics under the already frozen V1 contract; do not fit a replacement global maximum.

`evidence/qd6_candidate_review_v1.json` binds exact QD report hashes. It is not a final approval. The candidate-freeze tool refuses to run before fresh exact controls complete, after any HD6 output directory exists, or if actual runtime/frozen inputs drift. Quality remains NOT_RUN and cannot be inferred from these metrics.
