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

## Actual boundary replay, 23:27 UTC

tools-v18 against unchanged runtime-v6 reproduced the same Layer2 hard failure.
For both Layer1 and Layer2, initial resident E0/E8/E56 weights were checkpoint
exact in inference orientation (`w13` 2048×2816; `w2` 1408×2048).
Immediately after BF16 prepare, those controls were NOT checkpoint exact in raw
orientation (2816×2048; 2048×1408). CPU-staged E1/E63 were exact in both layers.
Layer2's reported E0 w13 maxabs was 0.5719451904296875. This directly isolates
the bad boundary to resident-to-raw full-context preparation rather than source
weight identity, wrapper layer identity, or all CPU staging. Original failed
request and partial generation are not qualified. Owned guard reaped it.

The candidate repair converts the entire NPU resident tensor to base ND format
before slicing experts, explicitly reverses the inference transpose when target
raw dimensions require it, and rejects unknown shapes/dtypes/non-base targets.
It is limited to BF16 prepare Phase1 on NPU; CUDA, CPU staging, arithmetic and
update selection are unchanged. A3 repair validation is pending, not PASS.

Primary-source cross-check (2026-09-17): installed torch_npu revision
`912882db09313e70f5312212669cd71e02e5c5d8` routes NPU-to-NPU copy to
aclnnInplaceCopy without the later internal-format guard. Current master explicitly
adds an A2/A3 internal-format fallback because that copy can corrupt storage.
This supports testing ND normalization; it does not by itself prove that backend
defect caused this model failure or authorize changing the installed wheel.

- [Installed copy implementation](https://github.com/Ascend/pytorch/blob/912882db09313e70f5312212669cd71e02e5c5d8/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp)
- [Current official implementation, comparison only](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp)
