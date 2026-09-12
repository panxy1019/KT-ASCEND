"""Round 5 Ascend framework transfer contract.

These tests intentionally exercise the public torch/torch_npu and kt-kernel
interfaces used by the aligned CPU-expert path.  They do not depend on the
Round 2B-only ``CPUInferTestTask`` pybind fixture or its historical method
aliases.
"""

from __future__ import annotations

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import (
    get_current_device_stream_handle,
    kt_kernel_ext,
    require_pinned_host_tensor,
)


def _initialize_npu() -> None:
    torch.npu.set_device(0)
    torch.empty(1, device="npu")


def test_callback_worker_subscribes_active_framework_stream() -> None:
    _initialize_npu()
    handle = get_current_device_stream_handle("npu")

    assert handle != 0
    try:
        kt_kernel_ext.init_ascend_callback_worker()
        kt_kernel_ext.subscribe_ascend_stream(handle)
        assert kt_kernel_ext.is_ascend_callback_worker_running()
    finally:
        kt_kernel_ext.shutdown_ascend_callback_worker()
    assert not kt_kernel_ext.is_ascend_callback_worker_running()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32, torch.int64))
def test_framework_d2h_h2d_roundtrip_is_exact(dtype: torch.dtype) -> None:
    _initialize_npu()
    if dtype == torch.bfloat16:
        expected = torch.arange(4096, dtype=torch.float32).to(dtype)
    elif dtype == torch.float32:
        expected = torch.linspace(-1.0, 1.0, 4096, dtype=dtype)
    else:
        expected = torch.arange(4096, dtype=dtype).remainder(64)

    device = expected.to("npu")
    host = require_pinned_host_tensor(
        torch.empty_like(expected, pin_memory=True), "round5_d2h_host"
    )
    returned = torch.empty_like(device)

    for _ in range(1000):
        host.copy_(device, non_blocking=True)
        torch.npu.current_stream().synchronize()
        returned.copy_(host, non_blocking=True)

    torch.npu.current_stream().synchronize()
    assert torch.equal(host, expected)
    assert torch.equal(returned.cpu(), expected)


def test_pageable_host_tensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="pinned"):
        require_pinned_host_tensor(torch.empty(16), "pageable")
