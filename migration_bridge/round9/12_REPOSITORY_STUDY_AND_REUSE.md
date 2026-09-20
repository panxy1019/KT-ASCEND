# Round9 repository study and reusable development context

Date: 2026-09-20. Requested by the user. Evidence level: CODE_INSPECTED.
This records a focused architecture/code/documentation review, not a claim that
every repository file was read or that any new runtime gate passed.

## Verified repository map

| Role | Repository / immutable revision | Interpretation |
|---|---|---|
| User project, published main | https://github.com/panxy1019/KT-ASCEND at5feb57f91aa9c786d2bc0002585a8958a456eb0f | Round5 published baseline; main alone is not the latest development evidence. |
| User published Round8 | codex/round8-qwen35-prep at9c1f1dae92303c1a13078319ab2aa2b5026fdb9e | Contains Round2–6 histories,182 Round7 files and Round8 final artifacts. |
| Official KT | https://github.com/kvcache-ai/ktransformers atf7607c0f66c643220ff43b822b24483c98c41392 | Official source and reuse reference; no automatic replacement of locally qualified behavior. |
| Official KT SGLang gitlink |541ddc37cbc92c60dc748db5ff1a2aad0b069a80|Verified from the official commit tree in this session; closes the earlier PENDING gitlink item. |
| Official KT llama.cpp gitlink |a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f|Actual GGML/block/kernel source pin to audit for format work. |

Remote refs were obtained with git ls-remote. Official commit/tree was fetched
to an isolated bare audit repository at /tmp/kt-round9-official-audit.git, without
changing the Round9 checkout or any submodule. Ten selected official files were
read through immutable raw URLs and verified against their Git blob identities;
SHA256 and local comparisons are in evidence/repository-study-20260920.json.

The published Round8 tree contains Round7 source/evidence and
round8_prep/10_HANDOFF/ROUND7_FINAL_BASE.md, which points to local final commit
30de6a234284bfa25d196037ed692d9584855ddf. The exact path
migration_bridge/round7/05_FINAL/ROUND7_FINAL_REPORT.md is absent from that
published Round8 tree. Its full final report was read in the separate kt-round7
worktree during this review. Do not equate branch upload with every later file in
every local worktree being present; this is an inventory distinction, not a
request or authorization to publish anything.

## Architecture retained for Round9

The serving/model layer remains SGLang. Its KTEP wrapper owns CPU/NPU logical
expert partitioning, placement and merge. KTMoEWrapper selects the CPU backend;
BaseMoEWrapper owns host/device staging and submit/drain; LLAMAFILE implements
expert arithmetic; worker pools own CPU/NUMA execution; the Ascend adapter and
callback worker implement stream callback/report integration.

Keep QSA/GDN/GR/PLE mathematical semantics in the model layer, outside generic
Ascend callback code. PLE gets its own bounded, exact host lookup/transfer
lifecycle. Do not create another independent Hybrid coordinator.

Relevant official/local comparisons:

| File | Comparison | Reuse action |
|---|---|---|
| kt-kernel/python/experts.py | Identical to official snapshot | Reuse backend factory, but distinguish method names from storage formats. |
| cpu_backend/ascend_callback_worker.cpp | Identical | Reuse report/subscription infrastructure; revalidate lifecycle for new workload. |
| cpu_backend/vendors/ascend_npu.h | Identical | Reuse adapter; borrowed framework stream remains framework-owned. |
| python/experts_base.py | Local changes include standalone CPU path, dtype-aware staging and diagnostic/deferred history | Preserve required contracts; do not blindly replace with official file or enable diagnostic hooks for timing. |
| operators/llamafile/moe.hpp | Local mapping, F32 partition/partial and capture changes | Audit exact shape/dtype paths before adding BF16 or low-bit. |
| cpu_backend/worker_pool.cpp | Local physical NUMA mapping/binding checks and other changes | Preserve physical-ID semantics and error checks; C128 requires its own topology proof. |

## Lessons grounded in previous rounds

- Round4A.5 isolated raw ACL pointer transfers as the trigger for same-path
  nondeterminism; framework Tensor.copy_ restored determinism under the tested
  controls. Never reintroduce raw pointer shortcuts for expert or PLE copies.
- Round5 used UPSTREAM_FIRST plus minimal missing semantics. Its retirement
  matrix explicitly rejects a second coordinator and preserves logical/physical
  IDs, route/shared exact-once behavior, FP32 merge and lifecycle regressions.
- Round6.5 heavy-offload numerical failures remain historical failures. Later
  product policy did not retroactively make them PASS.
- C64 improvements combined grouped-prefill, tail-safe partition, NUMA layout
  and worker changes. The historical speed difference is not pure core scaling
  and cannot predict C128 benefit.
- Round7 selected Uniform R8-C64, Dynamic/Deferred OFF after reduced-scope
  closure. Frequency hit improvements were not enough to establish net benefit.
- Round8 selected Qwen3.5 Q32-C64 Static, Dynamic/Deferred OFF. Its successful
  weight/restart/quality/triggered migration evidence is reusable methodology,
  not Qwen3.8 qualification. Performance remains cancelled/unqualified.
- Round8's transposed-NZ copy repair normalizes the full source to ND before
  logical-ID selection and copying into the existing destination. Same dtype
  and shape alone do not prove logical identity; budget temporary tensors too.

Primary local reading: round5/ROUND5_FINAL_SUMMARY.md and
19_PATCH_RETIREMENT_MATRIX.md; round4a5/05_RAW_ACL_TRANSFER_ROOT_CAUSE.md;
round6_7_c64/FINAL_COMPARISON_AND_DECISION.md,05_F32_TP_PARTITION.md and
07_FP32_PARTIAL_MERGE.md; round8_prep/FINAL/ROUND8_FINAL_REPORT.md and
11_DYNAMIC/NZ_TRANSFER_FIX.md. All round paths are under migration_bridge
unless explicitly prefixed round8_prep.

## Concrete Round9 precision/topology findings

1. Official experts.py maps method BF16/MXFP4 to NativeMoEWrapper, whereas
   LLAMAFILE maps to LlamafileMoEWrapper. Therefore BF16 storage through
   LLAMAFILE is not selected merely by setting --kt-method BF16. Likewise,
   MXFP4 GGUF through LLAMAFILE differs from the native MXFP4 backend.
2. Official Native-Precision-Tutorial is x86 AVX512/AMX-oriented. Its BF16/FP8
   support is not evidence of Kunpeng execution. The official Ascend DSV4
   tutorial does provide a Kunpeng/910B3 LLAMAFILE/MXFP4 precedent, but uses
   DSV4 weights and NPU W8A8 and even a separately named SGLang deployment
   branch. Its launch flags, clamp behavior and conversions are not Qwen defaults.
3. In the current local operators/moe-tp.hpp:83–104, the32-aligned multi-pool
   path is gated on all three weight types being F32. With BF16/low-bit and
   intermediate640, the alternate256-alignment path rejects tp_count>1.
   Thus current source has a concrete constructor-level gap for proposed
   BF16 C64/C128, beyond an untested performance expectation.
4. Local llamafile/moe.hpp:846–850 rejects non-F32 forward_many when its
   intermediate dimension is not256-aligned. Shapes640/320/160 require a
   deliberate implementation/probe;32-weight quantization blocks do not solve
   the grouped kernel's tile contract.
5. fp32_tp_partial_ currently activates only for multi-pool all-F32 weights.
   When bf16_numerical_compat is enabled, other formats still take the individual
   down-partial BF16 rounding branch. The BF16-storage/F32-compute reference must
   explicitly preserve the agreed partial reduction contract, not just rename
   the weight dtype.

These are source-level applicability findings, not a new device failure or an
authorization to skip real operand tests. Do not remove guards before verifying
load partition identity, decode/grouped arithmetic, same-quantized-weight
reference and merge behavior. Keep the latest precision amendment: bounded F32
reference → BF16-storage reference candidate → MXFP4/Q8_0 → same-format C128.

## Next use and boundaries

Source acquisition is now sufficient to stop guessing the official SGLang pin.
The separate upstream Qwen4 semantic snapshot4ccff141 remains a candidate for
selective ports; the official kvcache pin's native Qwen4 registration gap remains.
Checkpoint identity/census, actual imported Python/native hashes and independent
resource admission are still required before model execution.

This research changed documentation only. It did not update upstream in the
development tree, install packages, touch remote workers, resume automation,
restart Round8 performance, push, merge main, or create tags.
