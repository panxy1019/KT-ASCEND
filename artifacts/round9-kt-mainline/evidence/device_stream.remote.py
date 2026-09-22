"""Public framework stream handles used by CPUInfer callbacks."""

from __future__ import annotations

import torch


def get_current_device_stream_handle(device_type: str) -> int:
    """Return the borrowed native handle for the current CUDA or NPU stream.

    CPUInfer only observes this handle while submitting/synchronizing the
    current operation; ownership remains with the framework.
    """

    normalized = device_type.lower()
    if normalized == "npu":
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError(
                "torch.npu is unavailable; import a compatible torch_npu package first"
            )
        stream = npu.current_stream()
        attribute = "npu_stream"
    elif normalized == "cuda":
        stream = torch.cuda.current_stream()
        attribute = "cuda_stream"
    else:
        raise ValueError(f"unsupported device stream type: {device_type!r}")

    handle = getattr(stream, attribute, None)
    if not isinstance(handle, int) or handle == 0:
        raise RuntimeError(
            f"current {normalized} stream has no non-zero public {attribute} handle"
        )
    return handle


def require_pinned_host_tensor(
    tensor: torch.Tensor, name: str = "host_buffer"
) -> torch.Tensor:
    """Validate the host-side contract for an asynchronous device transfer."""

    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {tensor.device}")
    if not tensor.is_pinned():
        raise ValueError(
            f"{name} must use pinned host memory for asynchronous device transfer"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor
