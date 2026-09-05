from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import KTMoEWrapper, get_current_device_stream_handle, kt_kernel_ext, require_pinned_host_tensor
from kt_kernel.utils.llamafile import LlamafileMoEWrapper


FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_moe" / "create_tiny_moe_gguf_fixture.py"
SPEC = importlib.util.spec_from_file_location("tiny_moe_ascend_pipeline", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiny_fixture
SPEC.loader.exec_module(tiny_fixture)


def _rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _wrapper(path: Path):
    LlamafileMoEWrapper._gguf_loader_instance = None
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=tiny_fixture.DEFAULT_HIDDEN_SIZE,
        moe_intermediate_size=256,
        gpu_experts_mask=None,
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=str(path),
        chunked_prefill_size=64,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    wrapper.load_weights()
    return wrapper


def _reference(hidden_states, expert_ids, routing_weights, weights):
    rows = []
    for token_index, bf16_input in enumerate(hidden_states):
        result = torch.zeros_like(bf16_input.float())
        for route_index in range(expert_ids.shape[1]):
            expert = int(expert_ids[token_index, route_index])
            gate = weights["gate"][expert] @ bf16_input.float()
            up = weights["up"][expert] @ bf16_input.float()
            expert_output = weights["down"][expert] @ (torch.nn.functional.silu(gate) * up)
            result += float(routing_weights[token_index, route_index]) * expert_output
        rows.append(result.to(torch.bfloat16))
    return torch.stack(rows)


def _buffers():
    torch.manual_seed(tiny_fixture.SEED + 200)
    hidden_cpu = torch.randn(1, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    ids_cpu = torch.tensor([[1, 3]], dtype=torch.int64)
    weights_cpu = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    return {
        "input_npu": hidden_cpu.to("npu"),
        "ids_npu": ids_cpu.to("npu"),
        "weights_npu": weights_cpu.to("npu"),
        "input_host": require_pinned_host_tensor(torch.empty_like(hidden_cpu, pin_memory=True), "input_host"),
        "ids_host": require_pinned_host_tensor(torch.empty_like(ids_cpu, pin_memory=True), "ids_host"),
        "weights_host": require_pinned_host_tensor(torch.empty_like(weights_cpu, pin_memory=True), "weights_host"),
        # LLAMAFILE exposes a FP32 routed contribution.  This test must not
        # provide BF16 storage to the CPU task, or the task writes twice the
        # allocated byte count before the final model boundary cast.
        "output_host": require_pinned_host_tensor(
            torch.empty(hidden_cpu.shape, dtype=torch.float32, pin_memory=True),
            "output_host",
        ),
        "output_npu": torch.empty(hidden_cpu.shape, dtype=torch.float32, device="npu"),
        "qlen": torch.tensor([1], dtype=torch.int32),
    }


def _enqueue_cycle(wrapper, buffers, marker=None):
    buffers["input_host"].copy_(buffers["input_npu"], non_blocking=True)
    buffers["ids_host"].copy_(buffers["ids_npu"], non_blocking=True)
    buffers["weights_host"].copy_(buffers["weights_npu"], non_blocking=True)
    handle = get_current_device_stream_handle("npu")
    task = wrapper.moe.forward_task(
        buffers["qlen"].data_ptr(),
        2,
        buffers["ids_host"].data_ptr(),
        buffers["weights_host"].data_ptr(),
        buffers["input_host"].data_ptr(),
        buffers["output_host"].data_ptr(),
    )
    wrapper.cpu_infer.submit_with_device_stream(handle, task)
    independent_npu = buffers["input_npu"].float().square().sum()
    wrapper.cpu_infer.sync_with_device_stream(handle)
    buffers["output_npu"].copy_(buffers["output_host"], non_blocking=True)
    verified_npu = buffers["output_npu"].float() + independent_npu * 0.0
    if marker is not None:
        wrapper.cpu_infer.submit_with_device_stream(handle, marker.task(0))
    return verified_npu


def _assert_numerical(actual, expected):
    difference = actual.float() - expected.float()
    reference_norm = float(torch.linalg.vector_norm(expected.float()))
    metrics = {
        "max_abs_error": float(difference.abs().max()),
        "mean_abs_error": float(difference.abs().mean()),
        "relative_l2_error": float(torch.linalg.vector_norm(difference)) / reference_norm if reference_norm else 0.0,
    }
    print(json.dumps(metrics, sort_keys=True))
    assert metrics["max_abs_error"] <= 1e-3
    assert metrics["mean_abs_error"] <= 1e-4
    assert metrics["relative_l2_error"] <= 1e-2


def test_d2h_cpuinfer_moe_h2d_pipeline(tmp_path):
    fixture, _ = tiny_fixture.create_fixture(tmp_path / "pipeline.gguf", num_experts=4)
    wrapper = _wrapper(fixture)
    buffers = _buffers()
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)[0]
    verified_npu = _enqueue_cycle(wrapper, buffers)
    torch.npu.current_stream().synchronize()
    expected = _reference(buffers["input_host"], buffers["ids_host"], buffers["weights_host"], weights)
    _assert_numerical(verified_npu.cpu().to(torch.bfloat16), expected)


def test_runtime_pipeline_1000_cycles_and_rss(tmp_path):
    fixture, _ = tiny_fixture.create_fixture(tmp_path / "pipeline-1000.gguf", num_experts=4)
    wrapper = _wrapper(fixture)
    buffers = _buffers()
    expected = None
    for _ in range(10):
        expected = _enqueue_cycle(wrapper, buffers)
    torch.npu.current_stream().synchronize()
    wrapper.cpu_infer.sync()
    rss_before = _rss_bytes()

    markers = []
    for _ in range(1000):
        marker = kt_kernel_ext.testing.CPUInferTestTask()
        markers.append(marker)
        expected = _enqueue_cycle(wrapper, buffers, marker)
    torch.npu.current_stream().synchronize()
    wrapper.cpu_infer.sync()
    rss_after = _rss_bytes()

    assert expected is not None
    assert sum(marker.completions for marker in markers) == 1000
    assert torch.isfinite(expected).all()
    assert rss_after - rss_before < 16 * 1024 * 1024
    print(json.dumps({"cycles": 1000, "rss_before": rss_before, "rss_after": rss_after, "rss_delta": rss_after - rss_before}))
