# Round15 provider map

Status: `P0_OFFLINE_PROVIDER_AUDIT_COMPLETE_DEVICE_VALIDATION_PENDING`

This map binds the Round14a promoted runtime to the mature Ascend operator
families inspected for Round15.  A symbol being present is only an API
capability result.  Numerical and performance qualification remain separate.

## Frozen runtime identity

| Item | Frozen value |
|---|---|
| Parent compute snapshot | `artifacts/round14a/a-gdn-model-snapshots/v295` |
| Qwen model source SHA-256 | `1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545` |
| QSA primitive source SHA-256 | `0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35` |
| GDN primitive source SHA-256 | `65d990b19b6cfb35ea8f86e94985120a4a37c8397f234173365ec8655f25ebd1` |
| KT image | `kt-round64-migration:20260916`, image ID `sha256:601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a` |
| KT software | torch 2.9.0+cpu, torch-npu 2.9.0.post2, CANN 9.0.0 |
| Donor image | `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`, image ID `sha256:7e3ab7e56a54259d3ecec91073346f9ca8b2a4c1f58e15826e3ed6ff67036f00` |
| Donor software | torch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1.0, vLLM-Ascend commit `550ab66e7feb89f6e2ba49394f6ffb153189ac2b` |

The donor source is Apache-2.0.  Round15 does not upgrade the KT image or copy
the donor framework integration wholesale.

## A1 — QSA Lightning Indexer

**Local call point.** `Qwen4ReferenceIndexer.forward()` projects and normalizes
index queries and keys, updates the compression ring, then calls
`round9.qsa.select_tokens()`.  The current selector materializes score tensors
and uses eager `topk`; Round14a measured QSA indexer as a material Decode cost.

**Reusable implementation.** The donor adapter calls
`torch.ops.npu.npu_lightning_indexer` and expands selected compressed groups
back to logical token IDs.  The exact donor files and hashes are:

- `vllm_ascend/models/qwen4_exp/lightning_indexer.py`: `58f4eb39...97942`
- `vllm_ascend/models/qwen4_exp/qsa.py`: `fbd45f3a...c2949`

**Local capability.** torch-npu 2.9 registers
`npu_lightning_indexer(query, key, weights, ..., layout_query, layout_key,
sparse_count, sparse_mode, ...) -> (indices, values)`.  BF16, TND query and
PA_BSND key are documented in the installed package.  The donor's E3 index
expansion is a separate Triton-Ascend kernel and is not assumed available.

**Semantic contract.** Preserve one signed ranking score per compressed group,
per-query causal visibility, compression ratio 4, Top-K budget, stable request
ownership, logical output token IDs, and existing ordinary KV/index identity.
The reference remains the frozen eager selector and its FP32 oracle.  Candidate
output may reorder only equal-score ties if the final attention result remains
within the pre-registered module tolerance.

**Layout gap.** The frozen KT pool uses page size 4 for ordinary KV and indexes
compressed keys by `physical_token_slot // 4`.  The donor adapter requires a
`[pages, 192, 1, 128]` PA cache.  Those layouts are not interchangeable.  A
candidate must use a bounded, request-owned contiguous index cache or prove an
exact page table; it may not reinterpret the existing tensor with guessed
strides.  Unsupported shapes must select the eager implementation before any
state update.

**BF16/TP1 status.** `PENDING_COMPONENT`.  The operator exists in the frozen
stack and the model geometry is compatible (BF16, index head width 128,
compression ratio 4, TP1).  PA layout construction, short-history behavior,
causal expansion and numerical equivalence still require a device component
probe.

**Fallback.** Frozen eager `select_tokens()` plus promoted Vectorized Prefill /
Direct BF16 Row Decode attention.

## A2 — GDN native RMSNorm plus local sigmoid gate

**Local call point.** The promoted Round14a GDN uses the already qualified NPU
convolution/recurrent core, then calls `sigmoid_gated_rms_norm(core, z,
norm.weight)` and the existing output projection.

**Reusable implementation.** torch-npu 2.9 registers
`npu_rms_norm(Tensor self, Tensor gamma, float epsilon) -> (Tensor, Tensor)`.
The candidate replaces only RMS normalization with that mature BF16 operator,
then applies the frozen local `sigmoid(z)` multiplication.  It does not use the
donor swish parent or change recurrent state.

**Semantic contract.** `normalize(core) * norm.weight * sigmoid(z)`, FP32
recurrent state, existing convolution state, identical norm epsilon, identical
BF16 output and existing output projection.  The original FP64 GDN oracle and
Round14a component reference stay authoritative.

**BF16/TP1 status.** `PENDING_COMPONENT`.  API and shapes are compatible.  The
candidate is expected to affect only the post-core part and must improve the
complete parent call, including projections and state handling, by 15% in both
component blocks to advance.

**Fallback.** Promoted Round14a implementation using the native core and local
eager sigmoid-gated RMSNorm.

## A3 — resident BF16 MoE grouped execution

**Local call point.** `UnquantizedFusedMoEMethod.forward_npu()` already calls
`npu_moe_init_routing_v2`, `npu_grouped_matmul`, `npu_swiglu`, a second
`npu_grouped_matmul`, and `npu_moe_finalize_routing`.  KTEP masks CPU-owned
routes with in-range zero-weight placeholders and merges the FP32 CPU
contribution afterward.

**Reusable implementation.** This is already the mature Ascend grouped-MoE
provider present in the frozen SGLang stack.  The new vLLM-Ascend source does
not provide a distinct BF16/TP1 algorithm that can replace it without also
changing routing, quantization, placement or framework ownership.

**Semantic contract.** Uniform Q32 resident subset, compact physical expert
mapping, zero contribution for CPU-owned routes, BF16/NZ29 weights, router
weights, empty resident/CPU branches, and FP32 CPU/NPU merge.

**BF16/TP1 status.** `NO_NEW_A_CANDIDATE_ALREADY_CURRENT_PROVIDER`.  It remains
a graph-capture target for B, where eager and graph modes keep the same
operator identity.  It is not counted as an A speedup candidate.

**Fallback.** The same eager grouped-MoE path.

## A selection

Round15 will component-screen A1 and A2.  A3 is closed as already integrated.
At most one component-qualified implementation may consume model smoke and
request ABBA budget.  A Decode request gate is frozen if A1 advances; a
Prefill request gate is frozen if A2 advances only when the fresh P0 profile
shows Prefill dominance.  No result-dependent gate switching is allowed.

## B provider boundary

B targets only the existing A3 resident-NPU `gpu_method.apply()` region for
Decode batch 1.  CPUInfer submission/wait, route production, Python scheduler,
PLE host work, cancel decisions, CPU/NPU merge and subsequent layers remain
outside.  This keeps the operator identity constant and gives the graph a
fixed tensor-only surface: hidden state, physical top-k IDs, top-k weights and
static per-layer BF16/NZ29 weights.
