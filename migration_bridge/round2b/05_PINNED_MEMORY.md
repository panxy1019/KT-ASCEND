# Pinned Host Memory

## Allocator decision

The installed PyTorch/torch_npu combination supports `torch.empty(..., pin_memory=True)` for CPU tensors and reports `is_pinned() == True`. Therefore the runtime uses PyTorch's pinned allocator; an `aclrtMallocHost` fallback is **BYPASSED**, not required.

`require_pinned_host_tensor` enforces all transfer-buffer preconditions at the Python boundary:

- CPU device;
- pinned allocation;
- contiguous storage.

Pageable CPU memory and NPU tensors are rejected with explicit `ValueError`s. Python tensors remain the owners; raw `void *` ownership is not distributed across Python code.

| Buffer role | dtype | owner | lifetime |
|---|---|---|---|
| hidden input staging | BF16 | Python tensor | through D2H, CPUInfer and sync callback |
| expert IDs staging | int64 | Python tensor | through CPUInfer task completion |
| routing weights staging | float32 | Python tensor | through CPUInfer task completion |
| output staging | BF16 | Python tensor | through sync callback and H2D completion |

Evidence: `pytorch-pinned-audit.log`, `ascend-transfers.log`.
