# KTEP CUDA Dependency Matrix

The audit was performed against SGLang base `0f36b26d`, not against a historical
copy. Device-neutral names such as `gpu_layer` are retained where they are data
model terminology; they do not imply a CUDA runtime call.

| Dependency class | Examples | Round 3 treatment |
|---|---|---|
| Hybrid hot path | current stream, auxiliary stream, event, stream context, native handle | Ported through the active accelerator device module and Round 2B native stream helper |
| Import/runtime probe | `torch.cuda.get_device_capability`, router CUDA JIT assumptions | Guarded or replaced so Ascend import does not initialize CUDA |
| BF16 MVP load | ordinary tensor creation/copy and physical expert slicing | Uses active device; verified with 63 physical NPU experts at the hybrid layer |
| Advanced weight transport | `cudaHostRegister`, CUDA copy/post streams in FP8/MXFP paths | Explicitly excluded from the BF16 MVP; not mechanically rewritten |
| Debug/profiling | optional CUDA synchronizations in timing helpers | Not used by the verified configuration |
| Naming only | `gpu_experts_mask`, `gpu_method`, `gpu_layer` | Kept for compatibility; semantics are accelerator ownership |

## Patch results

- CUDA capability probing is no longer unconditional during Ascend import.
- Router linear setup avoids CUDA JIT on NPU.
- KTEP hot-path stream/event creation uses the selected accelerator backend.
- Dual-stream ordering is preserved on Ascend.
- All-accelerator layers skip the KT wrapper.
- CPU mapping is normalized to contiguous `int32` at the LLAMAFILE boundary.
- Ascend grouped MoE derives expert count from the physical weight dimension.
- CPU-owned `-1` routes are zero-weighted and sanitized before Ascend routing.

No whole-file `torch.cuda -> torch_npu` replacement was performed.

