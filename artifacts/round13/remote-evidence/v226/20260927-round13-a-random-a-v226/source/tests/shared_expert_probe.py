"""Original-weight shared-expert and final-hyperconnection component probe.

The real f02 Qwen2MoeSparseMoeBlock shared branch is checked against literal
CPU FP64 equations.  Its routed branch is replaced only at the block merge
boundary by an identifiable tensor, so that the test can prove one addition
with no extra routed scaling.  That fixture is not KTEP or routed-expert
evidence; ``ktep_bank_probe.py`` provides the separate real-KTEP evidence.

The final Qwen4 hyperconnection mixer is loaded from the original checkpoint
and checked against an independent CPU FP64 implementation.  Synthetic hidden
states are used throughout; this does not qualify a decoder or full model.
"""

import argparse
import ctypes
import faulthandler
import hashlib
import json
import math
import pathlib
import sys
from types import MethodType
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors import safe_open


faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--identity-receipt", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

base = pathlib.Path(__file__).parents[1]
sys.path[:0] = [str(base), str(base / "tools")]
from probe_identity import verify_probe_identity  # noqa: E402


revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
identity = verify_probe_identity(
    model=args.model,
    receipt=args.identity_receipt,
    expected_model=pathlib.Path("/models") / revision,
    expected_receipt_sha256=(
        "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
    ),
    revision=revision,
    manifest_sha256=(
        "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
    ),
)
weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
    "weight_map"
]

import torch_npu  # noqa: E402


torch.set_num_threads(2)
torch.npu.set_device(0)
torch.npu.matmul.allow_hf32 = False
torch.npu.conv.allow_hf32 = False
acl = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so")
physical = ctypes.c_int32(-1)
acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
if acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0:
    raise RuntimeError("failed to resolve physical NPU identity")
if physical.value != 15 or torch.npu.device_count() != 1:
    raise RuntimeError("probe is not isolated to physical NPU 15")
if list(pathlib.Path("/dev").glob("davinci[0-9]*")) != [
    pathlib.Path("/dev/davinci15")
]:
    raise RuntimeError("unexpected physical NPU exposure")

from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.distributed import (  # noqa: E402
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention  # noqa: E402
from sglang.srt.layers.moe import get_moe_a2a_backend  # noqa: E402
from sglang.srt.layers.moe.utils import initialize_moe_config  # noqa: E402
from sglang.srt.layers.round9.hyperconnection import (  # noqa: E402
    GatedResidual,
    HyperConnectionConfig,
)
import sglang.srt.models.qwen2_moe as qwen2_moe  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)


server_args = ServerArgs(
    model_path=str(args.model),
    device="npu",
    tp_size=1,
    enable_multimodal=False,
    dtype="bfloat16",
    disable_cuda_graph=True,
    disable_radix_cache=True,
    max_running_requests=1,
    page_size=4,
    chunked_prefill_size=32,
    attention_backend="ascend",
)
set_global_server_args_for_scheduler(server_args)
model_config = ModelConfig(
    str(args.model),
    trust_remote_code=False,
    enable_multimodal=False,
    dtype="bfloat16",
)
init_distributed_environment(
    world_size=1,
    rank=0,
    local_rank=0,
    distributed_init_method="tcp://127.0.0.1:29875",
    backend="gloo",
    timeout=60,
)
initialize_model_parallel(backend="gloo")
initialize_dp_attention(server_args, model_config)
initialize_moe_config(server_args)
config = model_config.hf_config.text_config
if (
    config.hidden_size != 2560
    or config.num_experts != 512
    or config.num_experts_per_tok != 10
    or config.shared_expert_intermediate_size != 640
    or config.moe_intermediate_size != 640
    or config.hidden_act != "silu"
    or config.hc_count != 4
    or config.hc_lowrank != 320
    or config.rms_norm_eps != 1e-6
    or config.norm_topk_prob is not True
):
    raise RuntimeError("frozen Qwen4 shared-expert/hyperconnection config mismatch")
if hasattr(config, "routed_scaling_factor"):
    raise RuntimeError("unexpected routed_scaling_factor changes the f02 merge contract")
if get_moe_a2a_backend().is_deepep():
    raise RuntimeError("bounded TP1 probe requires the ordinary non-DeepEP forward path")


def load_original(key):
    if key not in weight_map:
        raise KeyError(f"missing original checkpoint tensor: {key}")
    with safe_open(str(args.model / weight_map[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key).clone()


def relative_l2(actual, expected):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-30)
    )


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_parameter(parameter, source, shard_id=None):
    """Load only from a private CPU clone, through the real f02 loader."""
    staged = source.detach().cpu().clone()
    loader = getattr(parameter, "weight_loader", None)
    if loader is None:
        if shard_id is not None or tuple(parameter.shape) != tuple(staged.shape):
            raise ValueError("plain parameter shape/shard mismatch")
        parameter.data.copy_(staged.to(device=parameter.device, dtype=parameter.dtype))
    elif shard_id is None:
        loader(parameter, staged)
    else:
        loader(parameter, staged, shard_id)


class UnusedRoutedExperts(torch.nn.Module):
    """Constructor fixture; routed execution is replaced at the merge boundary."""

    def __init__(self, **kwargs):
        super().__init__()
        self.num_local_experts = kwargs["num_experts"]

    def forward(self, *unused_args, **unused_kwargs):
        raise AssertionError("routed-expert fixture must not execute")


old_default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
try:
    with patch.object(qwen2_moe, "get_moe_impl_class", return_value=UnusedRoutedExperts):
        with torch.device("npu:0"):
            moe = qwen2_moe.Qwen2MoeSparseMoeBlock(
                layer_id=0,
                config=config,
                quant_config=None,
                alt_stream=None,
                prefix="model.layers.0.mlp",
            )
    hc_config = HyperConnectionConfig(
        hc_count=config.hc_count,
        hidden_size=config.hidden_size,
        params_dtype=torch.bfloat16,
        hc_lowrank=config.hc_lowrank,
        rms_norm_eps=config.rms_norm_eps,
        hc_per_branch_norm=True,
    )
    with torch.device("npu:0"):
        mixer = GatedResidual(hc_config, use_combine=False)
finally:
    torch.set_default_dtype(old_default_dtype)


moe_prefix = "model.language_model.layers.0.mlp."
cpu_weights = {
    "router": load_original(moe_prefix + "gate.weight"),
    "shared_gate_proj": load_original(
        moe_prefix + "shared_expert.gate_proj.weight"
    ),
    "shared_up_proj": load_original(moe_prefix + "shared_expert.up_proj.weight"),
    "shared_down_proj": load_original(
        moe_prefix + "shared_expert.down_proj.weight"
    ),
    "shared_gate": load_original(moe_prefix + "shared_expert_gate.weight"),
}
with torch.no_grad():
    load_parameter(moe.gate.weight, cpu_weights["router"])
    load_parameter(
        moe.shared_expert.gate_up_proj.weight,
        cpu_weights["shared_gate_proj"],
        shard_id=0,
    )
    load_parameter(
        moe.shared_expert.gate_up_proj.weight,
        cpu_weights["shared_up_proj"],
        shard_id=1,
    )
    load_parameter(
        moe.shared_expert.down_proj.weight, cpu_weights["shared_down_proj"]
    )
    load_parameter(moe.shared_expert_gate.weight, cpu_weights["shared_gate"])

mixer_prefix = "model.language_model.hyper_connection_mixer."
mixer_weights = {
    name: load_original(mixer_prefix + name + ".weight")
    for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
}
with torch.no_grad():
    for name, source in mixer_weights.items():
        load_parameter(getattr(mixer, name).weight, source)

if "model.language_model.norm.weight" in weight_map:
    raise RuntimeError("checkpoint unexpectedly contains a post-mixer final norm")


def shared_cpu_fp64(hidden):
    x = hidden.detach().cpu().double()
    gate = F.linear(x, cpu_weights["shared_gate_proj"].double())
    up = F.linear(x, cpu_weights["shared_up_proj"].double())
    core = F.silu(gate) * up
    shared = F.linear(core, cpu_weights["shared_down_proj"].double())
    scalar = torch.sigmoid(F.linear(x, cpu_weights["shared_gate"].double()))
    return shared * scalar


def router_cpu_fp64(hidden):
    return F.linear(
        hidden.detach().cpu().double(), cpu_weights["router"].double()
    )


def mixer_cpu_fp64(hyper_input):
    x = hyper_input.detach().cpu().double()
    grouped = x.reshape(x.shape[0], config.hc_count, config.hidden_size)
    grouped = grouped * torch.rsqrt(
        grouped.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
    )
    normed = grouped.reshape_as(x) * (1.0 + mixer_weights["hc_norm"].double())
    low = F.silu(
        F.linear(normed, mixer_weights["input_mix_weight_down"].double())
        / config.hc_count
    )
    gates = torch.sigmoid(
        F.linear(low, mixer_weights["input_mix_weight_up"].double())
    )
    mixed = (gates * normed).reshape(
        x.shape[0], config.hc_count, config.hidden_size
    ).mean(dim=1)
    return mixed, normed


report = {
    "status": "RUNNING",
    "scope": "REAL_SHARED_EXPERT_AND_FINAL_MIXER_COMPONENTS_WITH_ROUTED_FIXTURE",
    "not_qualified": [
        "routed_expert_computation",
        "TopK_selection",
        "KTEP",
        "complete_MoE_output",
        "decoder_layer_output",
        "logits",
        "full_model",
    ],
    "source_identity": identity,
    "physical_device": 15,
    "runtime_shared_class": "sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock",
    "routed_fixture": (
        "identifiable tensor at _forward_router_experts boundary; not KTEP evidence"
    ),
    "separate_ktep_evidence": "migration_bridge/round9/tests/ktep_bank_probe.py",
    "routed_scaling_contract": (
        "f02 ordinary forward adds routed and gated-shared outputs once; actual "
        "config/source defines no routed_scaling_factor"
    ),
    "tokens": [1, 7, 17],
    "repeats": 3,
    "relative_l2_gate": 0.01,
    "weights": {
        "shared_and_router_tensors": len(cpu_weights),
        "final_mixer_tensors": len(mixer_weights),
        "all_sources_cpu_cloned_before_load": True,
    },
    "rows": [],
}

try:
    with torch.no_grad():
        for tokens in report["tokens"]:
            generator = torch.Generator(device="cpu").manual_seed(19600 + tokens)
            hidden = (torch.randn(tokens, config.hidden_size, generator=generator) * 0.2).to(
                torch.bfloat16
            )
            device_hidden = hidden.to("npu:0")
            want_shared = shared_cpu_fp64(hidden)
            want_router = router_cpu_fp64(hidden)
            shared_hashes = []
            boundary_hashes = []
            shared_errors = []
            router_errors = []
            boundary_calls = []
            for repeat in range(report["repeats"]):
                actual_shared = moe._forward_shared_experts(device_hidden)
                actual_router, _ = moe.gate(device_hidden)
                shared_error = relative_l2(actual_shared, want_shared)
                router_error = relative_l2(actual_router, want_router)
                if not math.isfinite(shared_error) or shared_error > 0.01:
                    raise AssertionError(("shared CPU FP64 mismatch", tokens, shared_error))
                if not math.isfinite(router_error) or router_error > 0.01:
                    raise AssertionError(("router CPU FP64 mismatch", tokens, router_error))

                marker_cpu = (
                    torch.arange(tokens * config.hidden_size, dtype=torch.int64)
                    .remainder(29)
                    .sub(14)
                    .reshape(tokens, config.hidden_size)
                    .to(torch.bfloat16)
                    / 64
                )
                marker = marker_cpu.to("npu:0")
                calls = {"count": 0}

                def routed_fixture(self, routed_hidden):
                    # Qwen2MoeSparseMoeBlock.forward creates a flattened view;
                    # require that it still aliases the exact input storage.
                    if (
                        routed_hidden.data_ptr() != device_hidden.data_ptr()
                        or routed_hidden.shape != device_hidden.shape
                    ):
                        raise AssertionError("block boundary changed routed input identity")
                    calls["count"] += 1
                    return marker.clone()

                moe._forward_router_experts = MethodType(routed_fixture, moe)
                boundary = moe(
                    device_hidden,
                    forward_batch=None,
                    use_reduce_scatter=False,
                )
                expected_boundary = marker + actual_shared
                if calls["count"] != 1 or not torch.equal(boundary, expected_boundary):
                    raise AssertionError("MoE boundary did not add routed/shared exactly once")
                if torch.equal(boundary, marker + actual_shared + marker):
                    raise AssertionError("routed fixture was applied twice")
                if torch.equal(boundary, marker + actual_shared + actual_shared):
                    raise AssertionError("shared branch was applied twice")
                shared_hashes.append(digest(actual_shared))
                boundary_hashes.append(digest(boundary))
                shared_errors.append(shared_error)
                router_errors.append(router_error)
                boundary_calls.append(calls["count"])

            if len(set(shared_hashes)) != 1 or len(set(boundary_hashes)) != 1:
                raise AssertionError("shared expert or merge boundary is nondeterministic")

            hyper = (
                torch.randn(
                    tokens,
                    config.hc_count * config.hidden_size,
                    generator=generator,
                )
                * 0.2
            ).to(torch.bfloat16)
            device_hyper = hyper.to("npu:0")
            want_mixed, want_normed = mixer_cpu_fp64(hyper)
            mixer_hashes = []
            mixer_errors = []
            norm_errors = []
            for repeat in range(report["repeats"]):
                actual_mixed, residual = mixer.mix(device_hyper)
                mixer_error = relative_l2(actual_mixed, want_mixed)
                norm_error = relative_l2(residual[1], want_normed)
                if not torch.equal(residual[0], device_hyper):
                    raise AssertionError("final mixer did not preserve original residual")
                if not math.isfinite(mixer_error) or mixer_error > 0.01:
                    raise AssertionError(("final mixer CPU FP64 mismatch", tokens, mixer_error))
                if not math.isfinite(norm_error) or norm_error > 0.01:
                    raise AssertionError(("final mixer norm CPU FP64 mismatch", tokens, norm_error))
                if actual_mixed.shape != (tokens, config.hidden_size):
                    raise AssertionError("final mixer output shape mismatch")
                if actual_mixed.dtype != torch.bfloat16 or not torch.isfinite(actual_mixed).all():
                    raise AssertionError("final mixer output dtype/finite gate failed")
                mixer_hashes.append(digest(actual_mixed))
                mixer_errors.append(mixer_error)
                norm_errors.append(norm_error)
            if len(set(mixer_hashes)) != 1:
                raise AssertionError("final mixer is nondeterministic")

            row = {
                "tokens": tokens,
                "shared_relative_l2": max(shared_errors),
                "router_logits_relative_l2": max(router_errors),
                "merge_boundary_calls": boundary_calls,
                "shared_hash": shared_hashes[0],
                "boundary_hash": boundary_hashes[0],
                "final_mixer_relative_l2": max(mixer_errors),
                "final_mixer_internal_norm_relative_l2": max(norm_errors),
                "final_mixer_hash": mixer_hashes[0],
            }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            del device_hidden, device_hyper, actual_shared, actual_router, boundary
            del actual_mixed, residual, marker, expected_boundary

    torch.npu.synchronize()
    report.update(
        status="PASS_SHARED_EXPERT_AND_FINAL_MIXER_COMPONENTS",
        rows_count=len(report["rows"]),
        final_post_mix_norm="ABSENT_IN_CHECKPOINT_AND_IN_CANONICAL_MIX_OUTPUT",
        peak_npu_allocated=torch.npu.max_memory_allocated(),
    )
except BaseException as exc:
    report.update(status="FAIL", error=repr(exc))
    raise
finally:
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.npu.empty_cache()

print(json.dumps({"status": report["status"], "rows": len(report["rows"])}), flush=True)
