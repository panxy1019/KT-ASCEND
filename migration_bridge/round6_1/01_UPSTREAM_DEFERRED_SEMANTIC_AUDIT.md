# Round 6.1 upstream Deferred semantic audit

Evidence state: `CODE_INSPECTED`.

This audit compares the rejected Round 6 lineage with the frozen upstream
KTransformers snapshot `0dce4c9b097710951d1b3e232f50b973ff0a5505` and its SGLang gitlink
`3424f35d0e60f03b9e75d9b90941f8f4cb944524`. It does not change production
code and does not reinterpret the Round 6 held-out failure.

## Dataflow and arithmetic semantics

The current Deferred implementation is not a same-layer delayed merge. For a
CPU-enabled layer `L`, `select_deferred_experts()` partitions routed experts
into immediate and deferred ID tensors. The immediate task writes the buffer
slot selected by `L % 2`; the deferred task writes the opposite slot and marks
layer `L` as pending. At layer `L+1`, the CPU kernel receives `incremental=True`
and reads the existing opposite-slot output before adding the current layer's
CPU result.

```text
Layer L CPU routes
  immediate subset -> output_cpu[L % 2] -> merged with Layer L NPU partial
  deferred subset  -> output_cpu[(L + 1) % 2]
                                         |
                                         v
Layer L+1 immediate CPU result -- incremental add --> Layer L+1 CPU partial
                                                        |
                                                        v
                                             Layer L+1 NPU/CPU merge
```

Therefore a deferred expert contribution selected by the router at layer `L`
is omitted from layer `L`'s routed MoE output and is injected into the routed
MoE output of `L+1`. Between those two boundaries, the model executes the
remaining layer-`L` residual/normalization path and the layer-`L+1` attention
path without that contribution. This is an execution-graph approximation, not
merely a different addition order. WP2-WP4 must still prove that this mechanism
explains the observed `hd_en_02:43` residual before classification is frozen.

## Frozen upstream implementation

| Semantic point | Frozen upstream source | Observed behavior |
| --- | --- | --- |
| Route partition | `kt-kernel/python/experts_base.py:492-520`, `select_deferred_experts` | Protects `protected_k` highest-score routes per row, then protects those expert IDs globally across the batch; all remaining IDs are deferred. |
| Buffer topology | `kt-kernel/python/experts_base.py:209-286`, `KExpertsCPUBuffer` | Two pinned CPU input/ID/weight/output slots and two device-output slots. Slots are selected by layer-index parity. |
| Immediate submission | `kt-kernel/python/experts_base.py:730-800`, `submit_forward` | On NPU, waits for framework D2H visibility and submits the immediate CPU task to `CPUInfer`; `incremental` is read from `layer_idx - 1`. |
| Deferred submission | `kt-kernel/python/experts_base.py:801-817`, `submit_forward` | Writes deferred work to `next_slot` and marks the producer layer pending. |
| Incremental C++ merge | `kt-kernel/operators/llamafile/moe.hpp:954-999`, `TP_MOE<LLAMA_MOE_TP>::merge_results` | When incremental, converts the existing output buffer to FP32 and adds it to the current layer's local CPU output before final output conversion. |
| CPU/NPU merge | upstream SGLang `kt_ep_wrapper.py:5870-5890`, `KTEPWrapperMethod.apply` | Synchronizes CPU output and adds it to the current layer NPU expert output. It has no separate same-layer deferred term. |
| Terminal handling | upstream SGLang `kt_ep_wrapper.py:4711-4729`, `create_kt_config_from_server_args` | Frozen upstream passes the requested defer budget to every wrapped layer; it does not guard a non-model terminal CPU block. |

## Local Round 6 deltas relevant to the investigation

| Local behavior | Location | Classification for Round 6.1 |
| --- | --- | --- |
| Disable defer at every contiguous CPU-block terminal | child `eaa58ccea4`, `resolve_layer_max_deferred()` | `LOCAL_REQUIRED_SYSTEM_FIX`; without an immediate `L+1` KT consumer the pending contribution has no consumer. This is independent of numerical acceptance. |
| Clear the consumed previous-layer pending flag | `kt-kernel/python/experts_base.py:603-606`, called after consumer submission | `LOCAL_REQUIRED_LIFECYCLE_FIX`; prevents a consumed producer from remaining logically pending across later requests. Must be tested by WP4 controls. |
| Suppress deferred no-op work when no CPU-owned route is present | `kt-kernel/python/experts_base.py:880-885` | `LOCAL_REQUIRED_CONTROL_FIX`; avoids falsely marking the following layer incremental. |
| LLAMAFILE CPU output buffer is FP32 | `kt-kernel/python/utils/llamafile.py:160`; buffer allocation and merge honor `output_dtype` | `LOCAL_NUMERICAL_SEMANTIC`; retained from Round 4A FP32 partial accumulation. Upstream raw uses hidden/BF16 output type. |
| BF16-visible gate/up/mul/down/route boundaries | `kt-kernel/operators/llamafile/moe.hpp` | `LOCAL_NUMERICAL_SEMANTIC`; retained from Round 4A.1. Not a Deferred-specific change. |
| Opt-in Deferred metadata and tensor tracing | `kt-kernel/python/experts_base.py`; child debug hooks | `TEST_ONLY_DIAGNOSTIC`; disabled by default. Used only after WP2 reproduction. |

## Upstream race fix review

Upstream commit `e7d1c1de09b375178eaf94db8749e36f8de3da2c` is titled
`fix(llamafile): resolve deferred experts data race and update README (#1646)`.
Its code change initializes each pinned `bsz_tensor_cpu` once and removes
per-forward `bsz_slot_tensor.fill_(batch_size)` and
`deferred_experts_ids_cpu[current_slot].fill_(-1)`. This removes host writes
that could race with consumers of the same pinned buffers. The fix is present
in the frozen upstream snapshot and in the Round 6 lineage.

That upstream fix does **not** change where a deferred layer-`L` expert output
is merged. It therefore cannot, by itself, establish whether the remaining
deterministic logit difference is a bug or the intended approximation.

## Numerical boundaries

The Round 6 lineage differs materially from raw upstream:

1. CPU expert gate, up, fused activation/multiply, down, and route weight have
   explicit BF16-visible rounding boundaries.
2. LLAMAFILE returns an FP32 routed CPU contribution.
3. SGLang adds CPU and NPU routed partials in FP32, then casts once to the model
   hidden dtype.

These verified Round 4A semantics are held fixed for both `D1_NODEFER` and
`D1`. Consequently, the controlled comparison changes Deferred scheduling,
not the already-qualified CPU/NPU numerical boundary contract.

## Audit conclusions and open questions

The source establishes that Deferred changes the layer at which a selected
expert contribution becomes visible. It also identifies three system-level
guards in the local lineage (terminal block, consumed-pending clearing, and
no-op suppression) that must remain exact.

Source inspection alone does not prove the `hd_en_02:43` root cause. The next
gates are:

1. reproduce the exact historical logits and `+0.75` pairwise distortion;
2. capture layer 7 and layer 8 stagewise tensors under identical history;
3. replay the deferred expert partial at its original layer boundary and at
   the implemented next-layer boundary;
4. use runtime/lifetime controls to exclude stale-buffer or ordering defects.

No threshold, corpus, placement, expert set, or acceptance rule was changed by
this audit.
