from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("torch_npu")

from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    layer_needs_kt_wrapper,
)
from sglang.srt.layers.quantization.unquant import _sanitize_ascend_moe_routes


def test_kt_wrapper_is_skipped_for_all_accelerator_layers():
    assert not layer_needs_kt_wrapper(torch.ones(64, dtype=torch.bool))
    one_cpu_expert = torch.ones(64, dtype=torch.bool)
    one_cpu_expert[6] = False
    assert layer_needs_kt_wrapper(one_cpu_expert)


def test_llamafile_expert_mapping_is_int32_at_load_boundary(monkeypatch):
    loaded_mappings = []

    class _Wrapper:
        def load_weights(self, mapping):
            loaded_mappings.append(mapping)

    class _GpuMethod:
        def process_weights_after_loading(self, _layer):
            pass

    metadata = SimpleNamespace(
        physical_to_logical_map_cpu=torch.arange(8, dtype=torch.int64).view(2, 4)
    )
    monkeypatch.setattr(
        "sglang.srt.eplb.expert_location_dispatch.get_global_expert_location_metadata",
        lambda: metadata,
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(synchronize=lambda: None),
    )

    method = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    method.gpu_method = _GpuMethod()
    method.tp_rank = 0
    method.wrapper = _Wrapper()
    method.kt_config = SimpleNamespace(layer_idx=1)
    method.kt_expert_lora_enabled = False
    layer = SimpleNamespace(parameters=lambda: iter([torch.empty(0)]), num_experts=4)

    method.process_weights_after_loading(layer)

    assert loaded_mappings[0].device.type == "cpu"
    assert loaded_mappings[0].dtype == torch.int32
    assert loaded_mappings[0].is_contiguous()
    assert loaded_mappings[0].tolist() == [4, 5, 6, 7]


def test_ascend_routes_disable_cpu_owned_experts():
    route_ids = torch.tensor([[4, -1], [-1, 2]], dtype=torch.int64, device="npu")
    route_weights = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32, device="npu"
    )

    sanitized_ids, sanitized_weights = _sanitize_ascend_moe_routes(
        route_ids, route_weights
    )

    assert sanitized_ids.dtype == torch.int32
    assert sanitized_ids.cpu().tolist() == [[4, 0], [0, 2]]
    torch.testing.assert_close(
        sanitized_weights.cpu(),
        torch.tensor([[0.25, 0.0], [0.0, 0.4]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
