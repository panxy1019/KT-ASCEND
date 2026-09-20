"""Real f02 pool/module gate for three bounded Qwen4 reference adapters.

Uses original layer-0 GDN, layer-1 PLE, and layer-3 QSA-indexer weights with
synthetic activations and one request.  It compares one 17-token extend,
aligned/unaligned chunks [3, 4, 10], and token decode through the actual f02
modules and Round9 pools.  This is not a decoder-layer or full-model output
qualification.
"""

import argparse
import copy
import ctypes
import faulthandler
import hashlib
import json
import math
import pathlib
import sys
from types import SimpleNamespace as NS

import torch
from safetensors import safe_open


faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--identity-receipt", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--full-attention", action="store_true")
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
from sglang.srt.layers.moe.utils import initialize_moe_config  # noqa: E402
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.round9_qsa_pool import Round9QSAKVPool  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.qwen4_exp import (  # noqa: E402
    CheckpointPLETable,
    Qwen4ReferenceGatedDeltaNet,
    Qwen4ReferenceIndexer,
    Qwen4ReferencePLE,
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
    distributed_init_method="tcp://127.0.0.1:29874",
    backend="gloo",
    timeout=60,
)
initialize_model_parallel(backend="gloo")
initialize_dp_attention(server_args, model_config)
initialize_moe_config(server_args)
config = model_config.hf_config.text_config
if (
    config.num_hidden_layers != 48
    or config.short_conv_layer_ids != [1]
    or tuple(config.short_conv_state_shape) != (10240, 9)
    or config.ngram_context_len != 2
    or list(config.full_attention_layer_ids) != list(range(3, 48, 4))
):
    raise RuntimeError("frozen Qwen4 adapter configuration mismatch")


def load_original(key):
    if key not in weight_map:
        raise KeyError(f"missing original checkpoint tensor: {key}")
    with safe_open(str(args.model / weight_map[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key).clone()


def load_module(module, checkpoint_prefix):
    """Load through each real f02 parameter loader from private CPU clones."""
    cpu_clones = {}
    for name, parameter in module.named_parameters():
        key = checkpoint_prefix + name
        original = load_original(key)
        target_dtype = torch.float32 if name == "A_log" else parameter.dtype
        staged = original.to(dtype=target_dtype).clone()
        loader = getattr(parameter, "weight_loader", None)
        if loader is None:
            if tuple(parameter.shape) != tuple(staged.shape):
                raise ValueError(f"shape mismatch for {key}")
            parameter.data.copy_(staged.to(parameter.device))
        else:
            loader(parameter, staged)
        cpu_clones[name] = staged
    return cpu_clones


def load_ple_buffers(module, checkpoint_prefix):
    result = {}
    for name in (
        "layer_multipliers",
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
    ):
        value = load_original(checkpoint_prefix + "ple_embedding." + name)
        destination = getattr(module, name)
        destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        result[name] = value.clone()
    return result


source_cache = config.mamba2_cache_params
bounded_cache = NS(
    shape=copy.copy(source_cache.shape),
    dtype=copy.copy(source_cache.dtype),
    layers=[0],
)
request_pool = HybridReqToTokenPool(
    size=1,
    mamba_size=1,
    mamba_spec_state_size=1,
    max_context_len=64,
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
qsa_pool = Round9QSAKVPool(
    size=64,
    dtype=torch.bfloat16,
    page_size=4,
    head_num=config.num_key_value_heads,
    head_dim=config.head_dim,
    full_attention_layer_ids=[3],
    enable_kvcache_transpose=False,
    device="npu:0",
    mamba_pool=request_pool.mamba_pool,
    request_capacity=1,
    index_head_dim=config.indexer_head_dim,
    compress_ratio=config.indexer_compress_ratio,
    enable_memory_saver=False,
    use_mla=False,
)


class Request:
    req_pool_idx = None
    mamba_pool_idx = None
    mamba_ping_pong_track_buffer = None
    is_chunked = 0
    kv_committed_len = 0

    def is_dllm(self):
        return False


def start_request():
    req = Request()
    req_slot = request_pool.alloc([req])[0]
    if req_slot != 0 or int(req.mamba_pool_idx) != 1:
        raise RuntimeError("bounded request did not receive the expected pool slots")
    mamba = req.mamba_pool_idx.long().reshape(1)
    layer_state = request_pool.mamba2_layer_cache(0)
    if (
        torch.count_nonzero(layer_state.conv[0][mamba]).item()
        or torch.count_nonzero(layer_state.temporal[mamba]).item()
        or torch.count_nonzero(request_pool.short_conv_layer_cache(1)[mamba]).item()
        or not torch.all(request_pool.get_ngram_context(mamba) == config.eos_token_id).item()
    ):
        raise RuntimeError("reused request slot retained stale GDN/PLE state")
    # Adapter fixture only: a scheduler would normally populate this mapping.
    request_pool.req_to_token[req_slot, :17] = torch.arange(
        4, 21, dtype=torch.int32, device="npu:0"
    )
    return req


def finish_request(req):
    request_pool.free_mamba_cache(req)
    request_pool.free(req)


def batch(req, token_ids, positions, mode):
    positions = positions.to(device="npu:0", dtype=torch.long)
    token_ids = token_ids.to(device="npu:0", dtype=torch.long)
    value = ForwardBatch(
        forward_mode=mode,
        batch_size=1,
        input_ids=token_ids,
        req_pool_indices=torch.tensor([req.req_pool_idx], device="npu:0"),
        seq_lens=torch.tensor([int(positions[-1]) + 1], device="npu:0"),
        seq_lens_sum=int(positions[-1]) + 1,
        out_cache_loc=positions.to(torch.int32) + 4,
    )
    value.req_to_token_pool = request_pool
    value.token_to_kv_pool = qsa_pool
    value.spec_info = None
    value.spec_algorithm = SpeculativeAlgorithm.NONE
    return value


PATHS = {
    "whole": [17],
    "chunks_3_4_10": [3, 4, 10],
    "token_decode": [1] * 17,
}


def ranges(parts):
    start = 0
    for width in parts:
        yield start, start + width
        start += width
    if start != 17:
        raise AssertionError("adapter path does not cover all 17 tokens")


def relative_l2(actual, expected):
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    )


def tensor_hash(value):
    cpu = value.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_state(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("state tuple width mismatch")
    errors = []
    for left, right in zip(actual, expected):
        if left.dtype in (torch.int32, torch.int64):
            if not torch.equal(left, right):
                raise AssertionError("integer adapter state mismatch")
            errors.append(0.0)
        else:
            errors.append(relative_l2(left, right))
    return max(errors, default=0.0)


def qualify(name, runner):
    rows = []
    baseline_output = None
    baseline_state = None
    hashes = {path: [] for path in PATHS}
    for path, parts in PATHS.items():
        for repeat in range(3):
            output, state = runner(parts)
            if not torch.isfinite(output).all().item():
                raise AssertionError(f"{name} produced non-finite output")
            hashes[path].append(tensor_hash(output))
            if baseline_output is None:
                baseline_output, baseline_state = output, state
            output_error = relative_l2(output, baseline_output)
            state_error = compare_state(state, baseline_state)
            row = {
                "adapter": name,
                "path": path,
                "repeat": repeat,
                "output_relative_l2": output_error,
                "state_relative_l2": state_error,
            }
            rows.append(row)
            if not math.isfinite(max(output_error, state_error)) or max(
                output_error, state_error
            ) > 0.01:
                raise AssertionError(row)
    for path, values in hashes.items():
        if len(set(values)) != 1:
            raise AssertionError(f"{name} {path} is not deterministic across reuse")
    return rows, hashes


torch.set_default_dtype(torch.bfloat16)
with torch.device("npu:0"):
    gdn = Qwen4ReferenceGatedDeltaNet(
        config=config, layer_id=0, quant_config=None, alt_stream=None, prefix="linear_attn"
    )
    ple = Qwen4ReferencePLE(config, 1)
    indexer = Qwen4ReferenceIndexer(config, 3, None, "indexer")
gdn_cpu = load_module(gdn, "model.language_model.layers.0.linear_attn.")
ple_cpu = load_module(ple, "model.language_model.layers.1.ple.")
ple_hash_cpu = load_ple_buffers(ple, "model.language_model.layers.1.ple.")
qsa_cpu = load_module(indexer, "model.language_model.layers.3.self_attn.indexer.")
if gdn.A_log.dtype != torch.float32 or gdn_cpu["A_log"].dtype != torch.float32:
    raise RuntimeError("GDN A_log did not remain FP32")

torch.manual_seed(20931)
gdn_hidden = torch.randn(17, config.hidden_size, dtype=torch.bfloat16, device="npu:0")
ple_hidden = torch.randn(
    17, config.hc_count * config.hidden_size, dtype=torch.bfloat16, device="npu:0"
)
qsa_hidden = torch.randn(17, config.hidden_size, dtype=torch.bfloat16, device="npu:0")
token_ids = torch.tensor(
    [1, 42, 248043, 248044, 7, 7, 0, 248319, 13, 5, 6, 7, 8, 9, 10, 11, 12],
    dtype=torch.long,
)


def run_gdn(parts):
    req = start_request()
    outputs = []
    try:
        for start, end in ranges(parts):
            mode = ForwardMode.EXTEND if end - start > 1 or start == 0 else ForwardMode.DECODE
            fb = batch(req, token_ids[start:end], torch.arange(start, end), mode)
            outputs.append(gdn(gdn_hidden[start:end], fb))
        slot = req.mamba_pool_idx.long()
        state = request_pool.mamba2_layer_cache(0)
        final = (state.conv[0][slot].detach().cpu().clone(),
                 state.temporal[slot].detach().cpu().clone())
        return torch.cat(outputs).detach().cpu(), final
    finally:
        finish_request(req)


def run_ple(parts):
    req = start_request()
    outputs = []
    try:
        for start, end in ranges(parts):
            mode = ForwardMode.EXTEND if end - start > 1 or start == 0 else ForwardMode.DECODE
            ids = token_ids[start:end]
            fb = batch(req, ids, torch.arange(start, end), mode)
            slot, embeddings, next_context = ple.prepare(ids.to("npu:0"), fb)
            outputs.append(ple.run(ple_hidden[start:end], embeddings, slot, fb))
            request_pool.set_ngram_context(slot, next_context)
        slot = req.mamba_pool_idx.long()
        final = (
            request_pool.short_conv_layer_cache(1)[slot].detach().cpu().clone(),
            request_pool.get_ngram_context(slot.reshape(1)).detach().cpu().clone(),
        )
        return torch.cat(outputs).detach().cpu(), final
    finally:
        finish_request(req)


def run_qsa(parts):
    req = start_request()
    queries = []
    selections = []
    try:
        for start, end in ranges(parts):
            mode = ForwardMode.EXTEND if end - start > 1 or start == 0 else ForwardMode.DECODE
            positions = torch.arange(start, end)
            fb = batch(req, token_ids[start:end], positions, mode)
            query, selected = indexer(qsa_hidden[start:end], positions.to("npu:0"), fb)
            queries.append(query)
            selections.append(selected)
        physical_groups = torch.tensor([1, 2, 3, 4], device="npu:0")
        state = (
            qsa_pool.get_qsa_key_state_buffer(3)[:4].detach().cpu().clone(),
            qsa_pool.get_qsa_rope_position_buffer(
                torch.arange(4, device="npu:0")
            ).detach().cpu().clone(),
            qsa_pool.get_qsa_compressed_k_buffer(3)
            .index_select(0, physical_groups)
            .detach().cpu().clone(),
            torch.cat(selections).detach().cpu().clone(),
        )
        return torch.cat(queries).detach().cpu(), state
    finally:
        finish_request(req)


if args.full_attention:
    # Only the unexecuted MoE constructor is replaced; self_attention below is
    # the real model method with real projections, RoPE and paged KV buffers.
    from unittest.mock import patch
    from sglang.srt.models import qwen3_5
    from sglang.srt.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer
    with torch.device("npu:0"), patch.object(
        qwen3_5, "Qwen2MoeSparseMoeBlock", side_effect=lambda **kw: torch.nn.Identity()
    ):
        attention = Qwen4ExpAttentionDecoderLayer(config, 3)
    attention.indexer = indexer
    attention_weights = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        key = name + ".weight"
        original = load_original("model.language_model.layers.3.self_attn." + key)
        attention_weights[key] = original
        if name in ("q_proj", "k_proj", "v_proj"):
            parameter = attention.qkv_proj.weight
            parameter.weight_loader(parameter, original, name[0])
        else:
            parameter = getattr(attention, name).weight
            getattr(parameter, "weight_loader", lambda p, v: p.data.copy_(v.to(p.device)))(
                parameter, original
            )
    # Reuse the independently expressed float64 oracle, not production math.
    import ast
    oracle_source = (pathlib.Path(__file__).with_name("qsa_device_probe.py")).read_text()
    oracle_node = next(n for n in ast.parse(oracle_source).body
                       if isinstance(n, ast.FunctionDef) and n.name == "canonical")
    oracle_namespace = {"torch": torch}
    exec(compile(ast.Module(body=[oracle_node], type_ignores=[]),
                 "qsa_device_probe.py:canonical", "exec"), oracle_namespace)
    attention_expected = oracle_namespace["canonical"](qsa_hidden.cpu(), attention_weights)
    attention_oracle_errors = []

    def run_full_attention(parts):
        req = start_request()
        outputs = []
        try:
            for start, end in ranges(parts):
                mode = ForwardMode.EXTEND if end-start > 1 or start == 0 else ForwardMode.DECODE
                positions = torch.arange(start, end, device="npu:0")
                fb = batch(req, token_ids[start:end], positions, mode)
                outputs.append(attention.self_attention(positions, qsa_hidden[start:end], fb))
            result = torch.cat(outputs).detach().cpu()
            error = relative_l2(result, attention_expected)
            if not math.isfinite(error) or error > .03:
                raise AssertionError(f"full QSA output versus FP64 oracle: {error}")
            attention_oracle_errors.append(error)
            return result, ()
        finally:
            finish_request(req)


report = {
    "status": "RUNNING",
    "scope": "THREE_REAL_SGLANG_ADAPTERS_ORIGINAL_WEIGHTS_SYNTHETIC_INPUTS_ONLY",
    "not_qualified": [
        "decoder_layer_output",
        "MoE",
        "logits",
        "full_model",
        "scheduler_allocation",
        "radix",
        "speculation_or_MTP",
        "graph",
        "disaggregation",
    ],
    "source_identity": identity,
    "physical_device": 15,
    "tokens": 17,
    "paths": PATHS,
    "repeats": 3,
    "relative_l2_gate": 0.01,
    "weights": {
        "gdn_parameter_tensors": len(gdn_cpu),
        "ple_parameter_tensors": len(ple_cpu),
        "ple_hash_buffers": len(ple_hash_cpu),
        "qsa_parameter_tensors": len(qsa_cpu),
        "a_log_dtype": str(gdn.A_log.dtype),
    },
    "rows": [],
    "repeat_hashes": {},
}
table_reader = CheckpointPLETable(args.model)
try:
    ple.bind_original_table(table_reader)
    with torch.no_grad():
        runners = [("gdn_layer0", run_gdn), ("ple_layer1", run_ple), ("qsa_indexer_layer3", run_qsa)]
        if args.full_attention:
            runners.append(("qsa_attention_layer3", run_full_attention))
        for name, runner in runners:
            rows, hashes = qualify(name, runner)
            report["rows"].extend(rows)
            report["repeat_hashes"][name] = hashes
    torch.npu.synchronize()
    report.update(
        status="PASS_FOUR_ADAPTERS_ONLY" if args.full_attention else "PASS_THREE_ADAPTERS_ONLY",
        full_attention_oracle_errors=attention_oracle_errors if args.full_attention else [],
        rows_count=len(report["rows"]),
        peak_npu_allocated=torch.npu.max_memory_allocated(),
    )
except BaseException as exc:
    report.update(status="FAIL", error=repr(exc))
    raise
finally:
    table_reader.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
print(json.dumps({"status": report["status"], "rows": len(report["rows"])}), flush=True)
