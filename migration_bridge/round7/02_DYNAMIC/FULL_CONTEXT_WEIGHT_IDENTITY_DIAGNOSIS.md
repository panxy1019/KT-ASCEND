# BF16 full-context weight-source identity diagnosis

2026-09-17. Status: investigation, not Dynamic qualification.

## Preserved actual failure

`backing-requalification-v5`/runtime-v6/tools-v17 closedoperators100/100,
Uniform32+restart4, Frequency32+restart4 and freshdirectR64174/256 invalid0.
The threshold256 triggered fixture entered full-context preparation and completed
one auditedLayer1 update (generation1, selected2/6/17/24/32/33/51/61). Layer2
then failed `FULL_CONTEXT_CHECKPOINT_EXPERT_IDENTITY`. Only1 audit record exists;
this is a partial update, not a successful request or coherent completed model
migration gate. Original guard reaping is complete; container has onlysleep.
No512/1024, thresholdfit, HD or productquality runs followed.

Lossless original archive `raw/dynamic-full-context-weight-identity-failure-v1.tar.gz`,
SHA256 `528cc3636b7fce044cb109794e19314707fee1f35611aab83fdb981fb77ac29d`.
Contains originaldriver/probe/process/generation records, freshR64 fullraw,
Frequencycompact/restart and actual runtime source/provenance. CompleteSCPexit0,
matching A3/local hash and full tar validationexit0.

## Independent resident-layout negative control

A reduced probe called the actual runtime-v6 `_prepare_weight_bf16` with raw NPU
full-context buffers and inference-transposed resident NPU matrices. Original
Phase1 directcopy raises aclnnInplaceCopy161002: source[4,12] cannot broadcast to
destination[12,4]. This independently exposes an unsupported inference→raw
layout boundary, but DOES NOT yet explain the actual Layer2 identity failure:
the model reached publication rather than that shape exception. Do not conflate
the two reproductions or prematurely claim the reduced negative is the root
cause of Layer2. It performed no modelrequests or business task mutation.

## Next minimal diagnostic replay — no production change

The untimed default-off Dynamic auditor now records actual resident/source and
prepared-context boundaries for fixedlogical controls0/1/8/56/63, including
shapes and exact checkpoint comparisons. Failed projections retain observed and
expected tensors. Its existing post-update checkpoint assertion remains hard,
now retaining first failinglayer/logical/name/slot, wrapperlayer and finite/error
metrics before rejecting. No weight/math/mask/threshold change.

99 existing localtests PASS; this is not certification of the new A3 diagnostic.
Deploy separatelyversioned tools-v18. Keep runtime-v6 and its closed operator/
Static/reference provenance unchanged. Replay the SAME threshold256 DEVfixture
in a new diagnostic output, without rereading HD or fitting thresholds. This
is root-cause instrumentation, not promotion or a substitute for OFFparity.

Interpret the snapshots before making a production repair: wrong initial
resident weights, source logical/layer identity, raw/inference copy, context
reuse, CPUstaging and postprocess boundaries are distinct hypotheses. If a real
semantic defect is localized, minimallyrepair, regress it and rerun the original
trigger gates. Weight-source identity cannot be relaxed by product quality.
