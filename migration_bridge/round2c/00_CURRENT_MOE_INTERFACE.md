# Round 2C Current MoE Interface Audit

## Entry points

| Component | Current source | Contract used by Round 2C |
|---|---|---|
| `KTMoEWrapper` | `kt-kernel/python/experts.py:72` and factory `_create_inference_wrapper`:316 | Selects `LlamafileMoEWrapper` for `method="LLAMAFILE"` at lines 343-400. |
| `BaseMoEWrapper` | `kt-kernel/python/experts_base.py:265` | Owns pinned transfer buffers and the callback-based `submit_forward`/`sync_forward` path. |
| `LlamafileMoEWrapper` | `kt-kernel/python/utils/llamafile.py:21` | Loads GGUF routed-expert weights and provides synchronous CPU execution. |
| `KExpertsCPUBuffer` | `kt-kernel/python/experts_base.py:86` | Allocates double-buffered pinned BF16 input/output, int64 IDs, float32 weights, and device output. |
| `HybridMoECoordinator` | `kt-kernel/python/hybrid_moe.py:211` | Partitions ownership, invokes the existing CPU path and a synthetic NPU provider, then merges routed contributions. |

## Router and tensor contract

- Router IDs remain **global logical expert IDs**. They are rank-2 `[tokens, top_k]` tensors with `int32` or `int64` accepted at the placement boundary; LLAMAFILE receives contiguous CPU `int64` (`utils/llamafile.py:270-289`).
- Router weights have the same shape as IDs, are floating point at validation, and enter LLAMAFILE as contiguous CPU `float32` (`utils/llamafile.py:273-290`).
- CPU hidden input/output is BF16. The direct CPU path flattens leading token dimensions, returns the original shape, and rejects a non-BF16 input (`utils/llamafile.py:255-295`).
- The asynchronous path uses pinned host BF16 input/output, pinned int64 IDs, pinned float32 weights, and an output tensor on the input device (`experts_base.py:99-153`).
- Hybrid output is on the NPU, BF16, and has the same shape as the input. `HybridMoEResult` additionally exposes CPU and NPU routed contributions (`hybrid_moe.py:202-209`).

## Placement and mapping semantics

- The legacy variable name `gpu_experts_mask` remains for compatibility. On Ascend, `True` means **NPU-owned** and therefore skipped by the CPU backend; `False` means CPU-owned. Python stores a pinned CPU bool mask (`experts_base.py:320-334`). C++ skips negative, out-of-range, or mask-true IDs (`operators/common.hpp:230-258`).
- `physical_to_logical_map[p] = l`: physical weight slot `p` contains global logical expert `l`. Python validates an exact permutation (`utils/llamafile.py:171-201`). C++ loops over physical slots and copies each weight into its logical slot (`operators/llamafile/moe.hpp:194-258`).
- CPU router weights are applied in LLAMAFILE, not in the coordinator: the decode/small path applies one matching weight at `moe.hpp:445-455`; the grouped path applies `weights[i*k+j]` once at `moe.hpp:717-735`.
- NPU weights are applied exactly once in `TorchNPUExpertProvider.forward` (`hybrid_moe.py:162-199`).

## Callback path

`BaseMoEWrapper.submit_forward` (`experts_base.py:415-493`) queues NPU-to-pinned-host copies, then a CPUInfer host callback. `sync_forward` (`experts_base.py:495-521`) queues the CPUInfer synchronization callback followed by pinned-host-to-NPU output copy. Round 2C does not put device work or device synchronization inside a callback.
