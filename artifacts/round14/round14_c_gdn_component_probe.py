#!/usr/bin/env python3
"""Real-weight Round14 GDN eager/backend correctness and parent timing probe."""

from __future__ import annotations

import argparse
import copy
import ctypes
import faulthandler
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F
from safetensors import safe_open


faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--identity-receipt", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument(
    "--version",
    type=int,
    choices=(276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287),
    required=True,
)
parser.add_argument("--block", choices=("A", "B"), required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
if (args.version, args.block) not in (
    (276, "A"), (277, "B"), (278, "A"), (279, "B"),
    (280, "A"), (281, "B"), (282, "A"), (283, "B"),
    (284, "A"), (285, "B"), (286, "A"), (287, "B")
):
    raise ValueError("Round14 GDN component version/block mismatch")

root = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(root), str(root / "source/tools")]
from probe_identity import verify_probe_identity  # noqa: E402


REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
identity = verify_probe_identity(
    model=args.model,
    receipt=args.identity_receipt,
    expected_model=pathlib.Path("/models") / REVISION,
    expected_receipt_sha256=(
        "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
    ),
    revision=REVISION,
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
if physical.value != 12 or torch.npu.device_count() != 1:
    raise RuntimeError("Round14 GDN component is not isolated to physical12")
if list(pathlib.Path("/dev").glob("davinci[0-9]*")) != [
    pathlib.Path("/dev/davinci12")
]:
    raise RuntimeError("unexpected physical NPU exposure")

from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.distributed import (  # noqa: E402
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend  # noqa: E402
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # noqa: E402
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.utils import initialize_linear_attn_config  # noqa: E402
from sglang.srt.layers.dp_attention import initialize_dp_attention  # noqa: E402
from sglang.srt.layers.moe.utils import initialize_moe_config  # noqa: E402
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # noqa: E402
from sglang.srt.model_loader.weight_utils import default_weight_loader  # noqa: E402
from sglang.srt.models.qwen4_exp import (  # noqa: E402
    Qwen4ReferenceGatedDeltaNet,
    _owned_cpu_staging,
)
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402


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
    linear_attn_backend="triton",
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
    distributed_init_method=f"tcp://127.0.0.1:{29900 + args.version}",
    backend="gloo",
    timeout=60,
)
initialize_model_parallel(backend="gloo")
initialize_dp_attention(server_args, model_config)
initialize_moe_config(server_args)
initialize_linear_attn_config(server_args)
config = model_config.hf_config.text_config
if (
    config.hidden_size,
    config.linear_num_key_heads,
    config.linear_num_value_heads,
    config.linear_key_head_dim,
    config.linear_value_head_dim,
    config.linear_conv_kernel_dim,
) != (2560, 16, 48, 128, 128, 4):
    raise RuntimeError("frozen Qwen3.8 GDN configuration changed")


SUFFIXES = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
    "out_proj.weight",
)


def checkpoint_key(layer, suffix):
    return f"model.language_model.layers.{layer}.linear_attn.{suffix}"


def original(layer, suffix):
    key = checkpoint_key(layer, suffix)
    if key not in weight_map:
        raise KeyError(key)
    with safe_open(str(args.model / weight_map[key]), framework="pt", device="cpu") as f:
        return _owned_cpu_staging(f.get_tensor(key))


def relative_l2(actual, expected):
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    )


def tensor_hash(value):
    cpu = value.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def cpu_fp64_canonical(hidden, weights):
    """Independent literal GDN equations; never calls the candidate or eager helper."""
    tokens = hidden.shape[0]
    x = hidden.double()
    qkv = F.linear(x, weights["in_proj_qkv.weight"].double())
    z = F.linear(x, weights["in_proj_z.weight"].double()).reshape(tokens, 48, 128)
    a = F.linear(x, weights["in_proj_a.weight"].double())
    b = F.linear(x, weights["in_proj_b.weight"].double())
    conv_weight = weights["conv1d.weight"].double().reshape(-1, 4)
    conv_state = torch.zeros(qkv.shape[-1], 3, dtype=torch.float64)
    mixed = []
    for token in range(tokens):
        window = torch.cat((conv_state, qkv[token, :, None]), dim=-1)
        value = (window * conv_weight).sum(-1)
        mixed.append(value * value.sigmoid())
        conv_state = window[:, 1:].clone()
    q, k, v = torch.split(torch.stack(mixed), (2048, 2048, 6144), dim=-1)
    q = q.reshape(tokens, 16, 128)
    k = k.reshape(tokens, 16, 128)
    v = v.reshape(tokens, 48, 128)
    recurrent = torch.zeros(48, 128, 128, dtype=torch.float64)
    outputs = []
    for token in range(tokens):
        query = q[token] * torch.rsqrt(q[token].square().sum(-1, keepdim=True) + config.rms_norm_eps)
        key = k[token] * torch.rsqrt(k[token].square().sum(-1, keepdim=True) + config.rms_norm_eps)
        query = query.repeat_interleave(3, dim=0)
        key = key.repeat_interleave(3, dim=0)
        decay = (-weights["A_log"].double().exp() * F.softplus(
            a[token] + weights["dt_bias"].double()
        )).exp()
        recurrent = recurrent * decay[:, None, None]
        error = v[token] - torch.einsum("hk,hkv->hv", key, recurrent)
        recurrent = recurrent + torch.einsum(
            "hk,hv->hkv", key, b[token].sigmoid()[:, None] * error
        )
        outputs.append(torch.einsum("hk,hkv->hv", query, recurrent) / math.sqrt(128))
    core = torch.stack(outputs)
    normalized = core * torch.rsqrt(
        core.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    gated = normalized * weights["norm.weight"].double() * z.sigmoid()
    output = F.linear(gated.flatten(1), weights["out_proj.weight"].double())
    return output, conv_state, recurrent


class Request:
    req_pool_idx = None
    mamba_pool_idx = None
    mamba_ping_pong_track_buffer = None
    is_chunked = 0
    kv_committed_len = 0

    def is_dllm(self):
        return False


def build_layer(layer_id):
    source_cache = config.mamba2_cache_params
    bounded_cache = NS(
        shape=copy.copy(source_cache.shape),
        dtype=copy.copy(source_cache.dtype),
        layers=[layer_id],
    )
    request_pool = HybridReqToTokenPool(
        size=1,
        mamba_size=1,
        mamba_spec_state_size=1,
        max_context_len=256,
        device="npu:0",
        enable_memory_saver=False,
        cache_params=bounded_cache,
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=None,
        short_conv_layer_ids=[1],
        short_conv_state_shape=tuple(config.short_conv_state_shape),
        ngram_context_len=config.ngram_context_len,
        eos_token_id=config.eos_token_id,
    )
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("npu:0"):
            module = Qwen4ReferenceGatedDeltaNet(
                config,
                layer_id,
                quant_config=None,
                prefix=f"model.layers.{layer_id}.linear_attn",
            )
    finally:
        torch.set_default_dtype(old_dtype)
    params = dict(module.named_parameters())
    cpu_weights = {}
    with torch.no_grad():
        for suffix in SUFFIXES:
            source = original(layer_id, suffix)
            cpu_weights[suffix] = source.clone()
            parameter = params[suffix]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, source)
            del source
    if module.A_log.dtype != torch.float32:
        raise RuntimeError("candidate GDN A_log did not remain FP32")
    linear_backend = GDNAttnBackend(
        NS(device=torch.device("npu:0"), req_to_token_pool=request_pool, token_to_kv_pool=None)
    )

    class NoFullAttentionBackend:
        def init_forward_metadata(self, forward_batch):
            del forward_batch

    # Production Qwen forwards through HybridLinearAttnBackend.  The component
    # has no full-attention layer, but retains that exact dispatcher contract.
    backend = HybridLinearAttnBackend(
        NoFullAttentionBackend(), linear_backend, full_attn_layers=[]
    )
    return module, request_pool, backend, cpu_weights


def start_request(pool):
    req = Request()
    slot = pool.alloc([req])[0]
    if slot != 0 or int(req.mamba_pool_idx) != 1:
        raise RuntimeError("unexpected bounded GDN pool slots")
    state = pool.mamba2_layer_cache(active_layer)
    idx = req.mamba_pool_idx.long().reshape(1)
    if torch.count_nonzero(state.conv[0][idx]).item() or torch.count_nonzero(state.temporal[idx]).item():
        raise RuntimeError("reused GDN request slot retained state")
    return req


def finish_request(pool, req):
    pool.free_mamba_cache(req)
    pool.free(req)


def make_batch(pool, backend, req, start, end):
    width = end - start
    mode = ForwardMode.EXTEND if width > 1 or start == 0 else ForwardMode.DECODE
    value = ForwardBatch(
        forward_mode=mode,
        batch_size=1,
        input_ids=torch.zeros(width, dtype=torch.long, device="npu:0"),
        req_pool_indices=torch.tensor([req.req_pool_idx], device="npu:0"),
        seq_lens=torch.tensor([end], dtype=torch.int32, device="npu:0"),
        seq_lens_sum=end,
        out_cache_loc=torch.arange(start + 4, end + 4, dtype=torch.int32, device="npu:0"),
    )
    value.req_to_token_pool = pool
    value.token_to_kv_pool = None
    value.spec_info = None
    value.spec_algorithm = SpeculativeAlgorithm.NONE
    value.extend_start_loc = torch.tensor([0], dtype=torch.int32, device="npu:0")
    value.extend_seq_lens = torch.tensor([width], dtype=torch.int32, device="npu:0")
    value.extend_seq_lens_cpu = torch.tensor([width], dtype=torch.int32)
    value.extend_prefix_lens = torch.tensor([start], dtype=torch.int32, device="npu:0")
    value.mamba_track_mask = None
    value.mamba_track_indices = None
    value.mamba_track_seqlens = None
    value.attn_backend = backend
    backend.init_forward_metadata(value)
    return value


def run_path(module, pool, backend, hidden, parts, candidate, measure_device=False):
    req = start_request(pool)
    outputs = []
    host_started = time.perf_counter_ns()
    event_start = torch.npu.Event(enable_timing=True) if measure_device else None
    event_end = torch.npu.Event(enable_timing=True) if measure_device else None
    try:
        module.round14_gdn_backend_enabled = bool(candidate)
        offset = 0
        if event_start is not None:
            event_start.record()
        for width in parts:
            batch = make_batch(pool, backend, req, offset, offset + width)
            outputs.append(module(hidden[offset : offset + width], batch))
            offset += width
        if event_end is not None:
            event_end.record()
        torch.npu.synchronize()
        host_ms = (time.perf_counter_ns() - host_started) / 1e6
        if offset != hidden.shape[0]:
            raise RuntimeError("GDN component path does not cover hidden fixture")
        cache = pool.mamba2_layer_cache(active_layer)
        idx = req.mamba_pool_idx.long()
        state = (
            cache.conv[0][idx].detach().cpu().clone(),
            cache.temporal[idx].detach().cpu().clone(),
        )
        output = torch.cat(outputs).detach().cpu()
        device_ms = (
            float(event_start.elapsed_time(event_end))
            if event_start is not None
            else None
        )
        return output, state, device_ms, host_ms
    finally:
        finish_request(pool, req)


report = {
    "schema": 1,
    "status": "RUNNING",
    "version": args.version,
    "block": args.block,
    "scope": "REAL_WEIGHT_GDN_FULL_PARENT_COMPONENT",
    "physical_device": 12,
    "source_identity": identity,
    "layers": [0, 24, 46],
    "relative_l2_gate": 0.03,
    "timing_gate": "BOTH_BLOCKS_CANDIDATE_OVER_BASELINE_MEDIAN_LE_0.85",
    "rows": [],
    "timing": [],
    "allocator": {},
}

active_layer = -1
try:
    torch.manual_seed(142760)
    for active_layer in report["layers"]:
        module, pool, backend, cpu_weights = build_layer(active_layer)
        hidden = torch.randn(17, config.hidden_size, dtype=torch.bfloat16, device="npu:0")
        paths = {"whole": [17], "chunks_3_4_10": [3, 4, 10], "decode_after_3": [3] + [1] * 14}
        # Compile every exact path before freezing the allocator baseline.  The
        # candidate backend lazily materializes kernels/metadata on first use;
        # this is persistent setup, not a per-request leak.
        for parts in paths.values():
            run_path(module, pool, backend, hidden, parts, False)
            run_path(module, pool, backend, hidden, parts, True)
        if active_layer == 0:
            run_path(module, pool, backend, hidden[:3], [3], False)
            run_path(module, pool, backend, hidden[:3], [3], True)
        torch.npu.synchronize()
        torch.npu.empty_cache()
        persistent = int(torch.npu.memory_allocated())
        report["allocator"][str(active_layer)] = {
            "persistent_after_exact_path_warmup": persistent
        }
        for name, parts in paths.items():
            baseline, baseline_state, _, _ = run_path(module, pool, backend, hidden, parts, False)
            candidate, candidate_state, _, _ = run_path(module, pool, backend, hidden, parts, True)
            row = {
                "layer": active_layer,
                "path": name,
                "output_relative_l2": relative_l2(candidate, baseline),
                "conv_state_relative_l2": relative_l2(candidate_state[0], baseline_state[0]),
                "recurrent_state_relative_l2": relative_l2(candidate_state[1], baseline_state[1]),
                "baseline_hash": tensor_hash(baseline),
                "candidate_hash": tensor_hash(candidate),
            }
            if not all(math.isfinite(row[key]) and row[key] <= report["relative_l2_gate"] for key in (
                "output_relative_l2", "conv_state_relative_l2", "recurrent_state_relative_l2"
            )):
                raise AssertionError(row)
            report["rows"].append(row)
        if active_layer == 0:
            oracle_hidden = hidden[:3].cpu()
            oracle_output, oracle_conv, oracle_recurrent = cpu_fp64_canonical(oracle_hidden, cpu_weights)
            for candidate in (False, True):
                output, state, _, _ = run_path(module, pool, backend, hidden[:3], [3], candidate)
                row = {
                    "layer": active_layer,
                    "path": "independent_fp64_3",
                    "candidate": candidate,
                    "output_relative_l2": relative_l2(output, oracle_output),
                    "conv_state_relative_l2": relative_l2(state[0], oracle_conv),
                    "recurrent_state_relative_l2": relative_l2(state[1], oracle_recurrent),
                }
                if not all(math.isfinite(row[key]) and row[key] <= report["relative_l2_gate"] for key in (
                    "output_relative_l2", "conv_state_relative_l2", "recurrent_state_relative_l2"
                )):
                    raise AssertionError(row)
                report["rows"].append(row)
        del cpu_weights
        torch.npu.empty_cache()
        final = int(torch.npu.memory_allocated())
        if final != persistent:
            raise RuntimeError(f"layer {active_layer} allocator baseline changed: {persistent} -> {final}")
        report["allocator"][str(active_layer)]["persistent_after_replay"] = final
        del module, pool, backend, hidden
        torch.npu.empty_cache()

    active_layer = 0
    module, pool, backend, _ = build_layer(active_layer)
    timing_hidden = torch.randn(128, config.hidden_size, dtype=torch.bfloat16, device="npu:0")
    # Compile/warm both paths outside the measured set.
    run_path(module, pool, backend, timing_hidden, [128], False)
    run_path(module, pool, backend, timing_hidden, [128], True)
    sequence = [False, True] if args.block == "A" else [True, False]
    for candidate in sequence:
        for repeat in range(7):
            output, state, device_ms, host_ms = run_path(
                module, pool, backend, timing_hidden, [128], candidate,
                measure_device=True,
            )
            report["timing"].append({
                "implementation": "NPU_BACKEND" if candidate else "EAGER_REFERENCE",
                "repeat": repeat,
                "device_ms": device_ms,
                "full_parent_host_ms": host_ms,
                "output_hash": tensor_hash(output),
                "state_hashes": [tensor_hash(item) for item in state],
            })
    medians = {}
    for implementation in ("EAGER_REFERENCE", "NPU_BACKEND"):
        selected = [row for row in report["timing"] if row["implementation"] == implementation]
        medians[implementation] = {
            "device_ms": statistics.median(row["device_ms"] for row in selected),
            "full_parent_host_ms": statistics.median(row["full_parent_host_ms"] for row in selected),
        }
        if len({row["output_hash"] for row in selected}) != 1 or len({tuple(row["state_hashes"]) for row in selected}) != 1:
            raise RuntimeError(f"{implementation} timing fixture is not repeat-exact")
    report["medians"] = medians
    report["ratios"] = {
        key: medians["NPU_BACKEND"][key] / medians["EAGER_REFERENCE"][key]
        for key in ("device_ms", "full_parent_host_ms")
    }
    report["component_gate_pass"] = report["ratios"]["full_parent_host_ms"] <= 0.85
    report.update(status="PASS_ROUND14_C_GDN_COMPONENT_PROCESS", rows_count=len(report["rows"]))
except BaseException as error:
    report.update(status="FAIL_ROUND14_C_GDN_COMPONENT_PROCESS", error=repr(error))
    raise
finally:
    report["peak_npu_allocated"] = int(torch.npu.max_memory_allocated())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.npu.empty_cache()

print(report["status"], report.get("ratios"), flush=True)
