# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch

import kt_kernel
from kt_kernel import KTMoEWrapper

FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_moe" / "create_tiny_moe_gguf_fixture.py"
SPEC = importlib.util.spec_from_file_location("tiny_moe_lifecycle_fixture", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiny_fixture
SPEC.loader.exec_module(tiny_fixture)


pytestmark = pytest.mark.skipif(
    kt_kernel.__cpu_variant__ != "arm",
    reason="LLAMAFILE lifecycle coverage requires the A3 ARM extension",
)


def _rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _wrapper(
    path: Path,
    *,
    layer_idx: int = 0,
    num_experts: int = 4,
    intermediate_size: int = 256,
    threadpool_count: int = 1,
    numa_nodes=None,
):
    if numa_nodes is None:
        numa_nodes = list(range(threadpool_count))
    return KTMoEWrapper(
        layer_idx=layer_idx,
        num_experts=num_experts,
        num_experts_per_tok=2,
        hidden_size=tiny_fixture.DEFAULT_HIDDEN_SIZE,
        moe_intermediate_size=intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=4,
        threadpool_count=threadpool_count,
        weight_path=str(path),
        chunked_prefill_size=64,
        method="LLAMAFILE",
        numa_nodes=numa_nodes,
    )


def _inputs(tokens: int = 1):
    torch.manual_seed(tiny_fixture.SEED + tokens)
    hidden_states = torch.randn(tokens, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    expert_ids = torch.tensor([[1, 3]], dtype=torch.int64).repeat(tokens, 1)
    routing_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32).repeat(tokens, 1)
    return hidden_states, expert_ids, routing_weights


def _reference(hidden_states, expert_ids, routing_weights, weights):
    result_rows = []
    for token_idx, x_bf16 in enumerate(hidden_states):
        x = x_bf16.float()
        result = torch.zeros_like(x)
        for route_idx in range(expert_ids.shape[1]):
            expert = int(expert_ids[token_idx, route_idx])
            gate = weights["gate"][expert] @ x
            up = weights["up"][expert] @ x
            result += float(routing_weights[token_idx, route_idx]) * (
                weights["down"][expert] @ (torch.nn.functional.silu(gate) * up)
            )
        result_rows.append(result.to(torch.bfloat16))
    return torch.stack(result_rows)


def test_weight_lifetime_and_1000_repeated_forwards(tmp_path):
    fixture_path, _ = tiny_fixture.create_fixture(tmp_path / "lifetime.gguf", num_experts=4)
    wrapper = _wrapper(fixture_path)
    wrapper.load_weights()
    assert wrapper.weights_to_keep is None
    gc.collect()

    hidden_states, expert_ids, routing_weights = _inputs()
    for _ in range(20):
        wrapper.forward(hidden_states, expert_ids, routing_weights)
    rss_before = _rss_bytes()
    rss_samples = []
    first_output = None
    for iteration in range(1000):
        output = wrapper.forward(hidden_states, expert_ids, routing_weights)
        if first_output is None:
            first_output = output.clone()
        else:
            assert torch.equal(output, first_output)
        if iteration % 100 == 99:
            rss_samples.append(_rss_bytes())
    rss_after = _rss_bytes()

    assert torch.isfinite(first_output).all()
    assert rss_after - rss_before <= 16 * 1024 * 1024
    print(
        json.dumps(
            {
                "iterations": 1000,
                "rss_before": rss_before,
                "rss_after": rss_after,
                "rss_delta": rss_after - rss_before,
                "rss_samples": rss_samples,
            },
            sort_keys=True,
        )
    )


def test_create_load_forward_destroy_and_loader_path_isolation(tmp_path):
    four_path, _ = tiny_fixture.create_fixture(tmp_path / "four.gguf", num_experts=4)
    eight_path, _ = tiny_fixture.create_fixture(tmp_path / "eight.gguf", num_experts=8)
    hidden_states, expert_ids, routing_weights = _inputs()
    observed_loader_paths = set()

    for iteration in range(20):
        num_experts, path = (4, four_path) if iteration % 2 == 0 else (8, eight_path)
        wrapper = _wrapper(path, num_experts=num_experts)
        observed_loader_paths.add(wrapper._gguf_loader_path)
        gate_shape = wrapper.gguf_loader.tensor_info["blk.0.ffn_gate_exps.weight"]["shape"]
        assert gate_shape[0] == num_experts
        wrapper.load_weights()
        output = wrapper.forward(hidden_states, expert_ids, routing_weights)
        assert torch.isfinite(output).all()
        del output, wrapper
        gc.collect()

    assert observed_loader_paths == {os.path.realpath(four_path), os.path.realpath(eight_path)}


def test_two_layers_use_distinct_gguf_keys(tmp_path):
    fixture_path, _ = tiny_fixture.create_fixture(tmp_path / "two-layers.gguf", num_experts=4, num_layers=2)
    all_weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)
    hidden_states, expert_ids, routing_weights = _inputs(tokens=2)

    outputs = []
    for layer_idx in (0, 1):
        wrapper = _wrapper(fixture_path, layer_idx=layer_idx)
        wrapper.load_weights()
        actual = wrapper.forward(hidden_states, expert_ids, routing_weights)
        expected = _reference(hidden_states, expert_ids, routing_weights, all_weights[layer_idx])
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected.float(), rtol=1e-2, atol=1e-3)
        outputs.append(actual)

    assert not torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    ("threadpool_count", "numa_nodes", "intermediate_size"),
    [(1, [0], 256), (2, [0, 1], 512)],
)
def test_threadpool_and_numa_mapping_are_observable(tmp_path, threadpool_count, numa_nodes, intermediate_size):
    fixture_path, _ = tiny_fixture.create_fixture(
        tmp_path / f"tp-{threadpool_count}.gguf",
        num_experts=4,
        intermediate_size=intermediate_size,
    )
    weights = tiny_fixture.generate_weights(
        num_layers=2,
        num_experts=4,
        intermediate_size=intermediate_size,
    )
    wrapper = _wrapper(
        fixture_path,
        intermediate_size=intermediate_size,
        threadpool_count=threadpool_count,
        numa_nodes=numa_nodes,
    )
    diagnostics = wrapper.cpu_runtime_diagnostics()
    assert diagnostics["subpool_count"] == threadpool_count
    assert diagnostics["subpool_numa_map"] == numa_nodes
    assert sum(diagnostics["subpool_thread_count"]) == 4
    assert diagnostics["process_cpu_affinity"]
    assert set(numa_nodes).issubset(diagnostics["available_numa_nodes"])

    wrapper.load_weights()
    hidden_states, expert_ids, routing_weights = _inputs(tokens=2)
    actual = wrapper.forward(hidden_states, expert_ids, routing_weights)
    expected = _reference(hidden_states, expert_ids, routing_weights, weights[0])
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), rtol=1e-2, atol=1e-3)
    print(json.dumps(diagnostics, sort_keys=True))
