# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

import kt_kernel
from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper

FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_moe" / "create_tiny_moe_gguf_fixture.py"
SPEC = importlib.util.spec_from_file_location("tiny_moe_gguf_fixture", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiny_fixture
SPEC.loader.exec_module(tiny_fixture)


pytestmark = pytest.mark.skipif(
    kt_kernel.__cpu_variant__ != "arm",
    reason="LLAMAFILE GGUF runtime coverage requires the A3 ARM extension",
)


def test_fixture_is_byte_reproducible(tmp_path):
    _first_path, first_manifest = tiny_fixture.create_fixture(tmp_path / "first.gguf", num_experts=4)
    _second_path, second_manifest = tiny_fixture.create_fixture(tmp_path / "second.gguf", num_experts=4)

    assert first_manifest == second_manifest
    assert first_manifest["seed"] == 20260827
    assert first_manifest["quant_type"] == "F32"


def _reference(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    rows = []
    for token_idx in range(hidden_states.shape[0]):
        x = hidden_states[token_idx].float()
        result = torch.zeros_like(x)
        for route_idx in range(expert_ids.shape[1]):
            expert = int(expert_ids[token_idx, route_idx])
            gate = layer_weights["gate"][expert] @ x
            up = layer_weights["up"][expert] @ x
            expert_output = layer_weights["down"][expert] @ (torch.nn.functional.silu(gate) * up)
            result += float(routing_weights[token_idx, route_idx]) * expert_output
        rows.append(result.to(torch.bfloat16))
    return torch.stack(rows)


def test_gguf_loader_to_llamafile_wrapper_e2e(tmp_path):
    torch.set_num_threads(1)
    fixture_path, _manifest = tiny_fixture.create_fixture(tmp_path / "tiny-moe.gguf", num_experts=4)
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)

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
        weight_path=str(fixture_path),
        chunked_prefill_size=64,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    wrapper.load_weights()

    torch.manual_seed(tiny_fixture.SEED + 1)
    hidden_states = torch.randn(1, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    expert_ids = torch.tensor([[1, 3]], dtype=torch.int64)
    routing_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    actual = wrapper.forward(hidden_states, expert_ids, routing_weights, cuda_stream=None)
    expected = _reference(hidden_states, expert_ids, routing_weights, weights[0])
    difference = actual.float() - expected.float()
    reference_norm = float(torch.linalg.vector_norm(expected.float()))
    max_abs = float(difference.abs().max())
    mean_abs = float(difference.abs().mean())
    relative_l2 = float(torch.linalg.vector_norm(difference)) / reference_norm if reference_norm else 0.0
    print(
        {
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "relative_l2_error": relative_l2,
        }
    )

    assert actual.shape == hidden_states.shape
    # The CPU expert wrapper returns an FP32 routed partial.  Hybrid merges
    # CPU and NPU partials in FP32 and performs the single BF16 cast at the
    # model-visible boundary (Round 4A.1 Fix B).
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert max_abs <= 1e-3
    assert mean_abs <= 1e-4
    assert relative_l2 <= 1e-2
