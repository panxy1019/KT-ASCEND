"""Real bounded Qwen4 decoder-layer probe with original weights and KTEP.

Layer 0 remains the default GDN gate. Layers 3/23/47 exercise the actual QSA
attention and page pool; layer 24 exercises a later GDN state slot. Every path
includes both gated residual adapters, router/TopK, KTEP CPU/NPU experts, and
the shared expert. This is not scheduler or full-model qualification.
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
from unittest.mock import patch

import torch
from safetensors import safe_open


faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--identity-receipt", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--layer", type=int, choices=(0, 3, 23, 24, 47), default=0)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
layer_id = args.layer
is_attention = layer_id in (3, 23, 47)

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

bank_root = pathlib.Path("/round9/banks/all48-v1")
catalog_path = bank_root / "catalog.json"
catalog_bytes = catalog_path.read_bytes()
catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
if catalog_sha256 != "35e83581957a34a8074caeb9bb40bffcf1cae6b2a03eab30f220fb4ccd01c479":
    raise RuntimeError("all48 bank catalog identity mismatch")
catalog = json.loads(catalog_bytes)
if (
    catalog.get("status") != "PASS_48_BANK_RECEIPTS_AND_LIVE_SIZES"
    or catalog.get("source_identity", {}).get("revision") != revision
    or catalog.get("source_identity", {}).get("receipt_sha256")
    != "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
    or catalog.get("source_identity", {}).get("manifest_sha256")
    != "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
    or catalog.get("exact_projections_from_receipts") != 73728
    or len(catalog.get("banks", [])) != 48
):
    raise RuntimeError("all48 bank catalog does not bind the frozen original model")
selected_bank = next(
    (item for item in catalog["banks"] if item.get("layer") == layer_id), None
)
if (
    selected_bank is None
    or selected_bank.get("file") != f"layer-{layer_id:02d}-bf16.gguf"
    or selected_bank.get("status") != "PASS_SINGLE_LAYER_BF16_BANK_IDENTITY_ONLY"
    or selected_bank.get("exact_projection_instances") != 1536
    or selected_bank.get("source_identity", {}).get("revision") != revision
    or selected_bank.get("source_identity", {}).get("receipt_sha256")
    != identity["receipt_sha256"]
    or not isinstance(selected_bank.get("sha256"), str)
    or len(selected_bank["sha256"]) != 64
):
    raise RuntimeError(f"layer-{layer_id} bank catalog entry mismatch")
live_bank = bank_root / selected_bank["file"]
if not live_bank.is_file() or live_bank.stat().st_size != selected_bank["bytes"]:
    raise RuntimeError(f"layer-{layer_id} bank file is absent or has wrong live size")

import torch_npu  # noqa: E402
from kt_kernel import kt_kernel_ext  # noqa: E402


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
from sglang.srt.model_loader.weight_utils import default_weight_loader  # noqa: E402
from sglang.srt.models.qwen4_exp import (  # noqa: E402
    Qwen4ExpAttentionDecoderLayer,
    Qwen4ExpLinearDecoderLayer,
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
    kt_weight_path=str(bank_root),
    kt_method="LLAMAFILE",
    kt_cpuinfer=64,
    kt_threadpool_count=2,
    kt_numa_nodes=[6, 7],
    kt_num_gpu_experts=64,
    kt_max_deferred_experts_per_token=0,
    kt_enable_dynamic_expert_update=False,
    kt_expert_placement_strategy="uniform",
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
    distributed_init_method="tcp://127.0.0.1:29876",
    backend="gloo",
    timeout=60,
)
initialize_model_parallel(backend="gloo")
initialize_dp_attention(server_args, model_config)
initialize_moe_config(server_args)
config = model_config.hf_config.text_config
if (
    config.hidden_size != 2560
    or config.hc_count != 4
    or config.num_hidden_layers != 48
    or config.num_experts != 512
    or config.num_experts_per_tok != 10
    or config.shared_expert_intermediate_size != 640
    or config.moe_intermediate_size != 640
    or config.short_conv_layer_ids != [1]
):
    raise RuntimeError("frozen Qwen4 decoder configuration mismatch")
if is_attention != (layer_id in set(config.full_attention_layer_ids)):
    raise RuntimeError("selected decoder layer type disagrees with frozen config")

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
qsa_pool = None
if is_attention:
    qsa_pool = Round9QSAKVPool(
        size=64,
        dtype=torch.bfloat16,
        page_size=4,
        head_num=config.num_key_value_heads,
        head_dim=config.head_dim,
        full_attention_layer_ids=[layer_id],
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
    if not is_attention:
        slot = req.mamba_pool_idx.long().reshape(1)
        state = request_pool.mamba2_layer_cache(layer_id)
        if (
            torch.count_nonzero(state.conv[0][slot]).item()
            or torch.count_nonzero(state.temporal[slot]).item()
        ):
            raise RuntimeError(
                f"reused request slot retained stale layer-{layer_id} GDN state"
            )
    # Adapter fixture only: the scheduler normally owns this physical mapping.
    request_pool.req_to_token[req_slot, :17] = torch.arange(
        4, 21, dtype=torch.int32, device="npu:0"
    )
    return req


def finish_request(req):
    request_pool.free_mamba_cache(req)
    request_pool.free(req)


def batch(req, start, end):
    mode = ForwardMode.EXTEND if end - start > 1 or start == 0 else ForwardMode.DECODE
    value = ForwardBatch(
        forward_mode=mode,
        batch_size=1,
        input_ids=torch.zeros(end - start, dtype=torch.long, device="npu:0"),
        req_pool_indices=torch.tensor([req.req_pool_idx], device="npu:0"),
        seq_lens=torch.tensor([end], device="npu:0"),
        seq_lens_sum=end,
        out_cache_loc=torch.arange(start + 4, end + 4, dtype=torch.int32, device="npu:0"),
    )
    value.req_to_token_pool = request_pool
    value.token_to_kv_pool = qsa_pool
    value.spec_info = None
    value.spec_algorithm = SpeculativeAlgorithm.NONE
    return value


old_default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
ktep_module_present_before_constructor = (
    "sglang.srt.layers.moe.kt_ep_wrapper" in sys.modules
)
try:
    with torch.device("npu:0"):
        layer_class = (
            Qwen4ExpAttentionDecoderLayer
            if is_attention
            else Qwen4ExpLinearDecoderLayer
        )
        suffix = "self_attn" if is_attention else "linear_attn"
        layer = layer_class(
            config,
            layer_id,
            quant_config=None,
            prefix=f"model.layers.{layer_id}.{suffix}",
            alt_stream=None,
        )
finally:
    torch.set_default_dtype(old_default_dtype)
if layer.ple is not None:
    raise RuntimeError("selected decoder layer unexpectedly enables PLE")
if is_attention != hasattr(layer, "indexer"):
    raise RuntimeError("constructed decoder class does not match selected layer type")

# The Qwen4 decoder constructor itself must have imported and registered KTEP
# before constructing its inherited Qwen2 MoE. Import here only for type checks
# and the existing capture hook; it must not make construction pass.
from sglang.srt.layers.moe import kt_ep_wrapper as kt_ep  # noqa: E402

params = dict(layer.named_parameters())
layer_prefix = f"model.language_model.layers.{layer_id}."
checkpoint_keys = sorted(key for key in weight_map if key.startswith(layer_prefix))
common_suffixes = {
        "attn_hyper_connection.block_inject_weight.weight",
        "attn_hyper_connection.hc_norm.weight",
        "attn_hyper_connection.input_mix_weight_down.weight",
        "attn_hyper_connection.input_mix_weight_up.weight",
        "mlp.experts.down_proj",
        "mlp.experts.gate_up_proj",
        "mlp.gate.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert_gate.weight",
        "mlp_hyper_connection.block_inject_weight.weight",
        "mlp_hyper_connection.hc_norm.weight",
        "mlp_hyper_connection.input_mix_weight_down.weight",
        "mlp_hyper_connection.input_mix_weight_up.weight",
}
linear_suffixes = {
    "linear_attn.A_log",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
}
attention_suffixes = {
    "self_attn.indexer.index_qk_proj.weight",
    "self_attn.indexer.k_layernorm.weight",
    "self_attn.indexer.q_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
}
expected_suffixes = common_suffixes | (
    attention_suffixes if is_attention else linear_suffixes
)
expected_checkpoint_keys = {layer_prefix + name for name in expected_suffixes}
if set(checkpoint_keys) != expected_checkpoint_keys:
    raise RuntimeError(f"original layer-{layer_id} checkpoint key set changed")


def original(key):
    with safe_open(str(args.model / weight_map[key]), framework="pt", device="cpu") as f:
        return _owned_cpu_staging(f.get_tensor(key))


loaded = set()
packed = {}
expert_source_shapes = {}
with torch.no_grad():
    for key in checkpoint_keys:
        suffix = key[len(layer_prefix) :]
        if suffix in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
            continue
        if suffix in (
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
        ):
            destination = "mlp.shared_expert.gate_up_proj.weight"
            shard_id = 0 if ".gate_proj." in suffix else 1
            parameter = params[destination]
            source = original(key)
            parameter.weight_loader(parameter, source, shard_id)
            packed.setdefault(destination, set()).add(shard_id)
            loaded.add(destination)
            del source
            continue
        if suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ):
            destination = "qkv_proj.weight"
            shard_id = suffix.removeprefix("self_attn.")[0]
            parameter = params[destination]
            source = original(key)
            parameter.weight_loader(parameter, source, shard_id)
            packed.setdefault(destination, set()).add(shard_id)
            loaded.add(destination)
            del source
            continue
        destination = suffix.removeprefix("self_attn.")
        if destination not in params:
            raise KeyError(f"missing isolated layer destination: {destination}")
        if destination in loaded:
            raise RuntimeError(f"duplicate isolated layer destination: {destination}")
        parameter = params[destination]
        source = original(key)
        loader = getattr(parameter, "weight_loader", default_weight_loader)
        loader(parameter, source)
        loaded.add(destination)
        del source

    expected_packed = {"mlp.shared_expert.gate_up_proj.weight": {0, 1}}
    if is_attention:
        expected_packed["qkv_proj.weight"] = {"q", "k", "v"}
    if packed != expected_packed:
        raise RuntimeError(f"packed projection load was incomplete: {packed}")

    expert_specs = (
        (
            layer_prefix + "mlp.experts.gate_up_proj",
            "mlp.experts.w13_weight",
        ),
        (
            layer_prefix + "mlp.experts.down_proj",
            "mlp.experts.w2_weight",
        ),
    )
    for key, destination in expert_specs:
        parameter = params[destination]
        loader = getattr(parameter, "weight_loader", None)
        if loader is None:
            raise RuntimeError(f"KTEP expert destination lacks loader: {destination}")
        with safe_open(
            str(args.model / weight_map[key]), framework="pt", device="cpu"
        ) as handle:
            source_slice = handle.get_slice(key)
            source_shape = tuple(source_slice.get_shape())
            expert_source_shapes[key] = source_shape
            if source_shape[0] != config.num_experts:
                raise RuntimeError(f"expert source count mismatch: {key}")
            for expert_id in range(config.num_experts):
                expert = source_slice[expert_id]
                if destination.endswith("w13_weight"):
                    if expert.shape[-2] % 2:
                        raise RuntimeError("odd gate/up expert projection width")
                    gate, up = expert.chunk(2, dim=-2)
                    loader(
                        parameter,
                        _owned_cpu_staging(gate),
                        destination,
                        "w1",
                        expert_id,
                    )
                    loader(
                        parameter,
                        _owned_cpu_staging(up),
                        destination,
                        "w3",
                        expert_id,
                    )
                    del gate, up
                else:
                    loader(
                        parameter,
                        _owned_cpu_staging(expert),
                        destination,
                        "w2",
                        expert_id,
                    )
                del expert
        loaded.add(destination)

missing = sorted(set(params) - loaded)
extra = sorted(loaded - set(params))
if missing or extra:
    raise RuntimeError(f"isolated layer load audit failed: missing={missing} extra={extra}")
if not is_attention and layer.linear_attn.A_log.dtype != torch.float32:
    raise RuntimeError(f"layer-{layer_id} GDN A_log did not remain FP32")

postprocessed = []
with torch.no_grad():
    for module_name, module in layer.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
            postprocessed.append(module_name)
            torch.npu.empty_cache()

experts = layer.mlp.experts
quant_method = experts.quant_method
if not isinstance(quant_method, kt_ep.KTEPWrapperMethod):
    raise RuntimeError("actual layer routed experts are not using KTEP")
if (
    quant_method.num_gpu_experts != 64
    or int(quant_method.gpu_experts_mask.sum()) != 64
    or quant_method.wrapper is None
    or quant_method.kt_config.cpuinfer_threads != 64
    or quant_method.kt_config.threadpool_count != 2
    or list(quant_method.kt_config.numa_nodes) != [6, 7]
    or quant_method.kt_config.method != "LLAMAFILE"
    or quant_method.kt_config.weight_path != str(bank_root)
    or quant_method.kt_config.layer_idx != layer_id
    or quant_method.kt_config.max_deferred_experts_per_token != 0
    or quant_method.kt_config.kt_enable_dynamic_expert_update
):
    raise RuntimeError("constructed KTEP layer does not match the bounded runtime profile")

torch.manual_seed(21701 + layer_id)
hidden = torch.randn(17, config.hidden_size, dtype=torch.bfloat16, device="npu:0")
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
        raise AssertionError("decoder path does not cover all 17 tokens")


def relative_l2(actual, expected):
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    )


def tensor_hash(value):
    cpu = value.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


active_capture = None


def capture(stage, tensor, **unused_kwargs):
    if active_capture is None or stage not in (
        "cpu_partial",
        "npu_owned_partial",
        "routed_merge",
    ):
        return
    active_capture[stage] += 1
    if torch.count_nonzero(tensor).item():
        active_capture[stage + "_nonzero"] += 1


def run_path(parts):
    global active_capture
    req = start_request()
    outputs = []
    active_capture = {
        "cpu_partial": 0,
        "cpu_partial_nonzero": 0,
        "npu_owned_partial": 0,
        "npu_owned_partial_nonzero": 0,
        "routed_merge": 0,
        "routed_merge_nonzero": 0,
    }
    try:
        for start, end in ranges(parts):
            forward_batch = batch(req, start, end)
            if is_attention:
                positions = torch.arange(start, end, dtype=torch.long, device="npu:0")
                output, residual = layer(
                    positions=positions,
                    hidden_states=hidden[start:end],
                    residual=None,
                    forward_batch=forward_batch,
                    ple_data=None,
                )
            else:
                output, residual = layer(
                    hidden[start:end],
                    None,
                    forward_batch=forward_batch,
                    ple_data=None,
                )
            if residual is not None:
                raise AssertionError(
                    f"Qwen4 layer-{layer_id} returned an unexpected residual"
                )
            if output.shape != (end - start, config.hc_count * config.hidden_size):
                raise AssertionError(f"Qwen4 layer-{layer_id} GR output shape mismatch")
            if output.dtype != torch.bfloat16 or not torch.isfinite(output).all().item():
                raise AssertionError(
                    f"Qwen4 layer-{layer_id} output dtype/finite gate failed"
                )
            outputs.append(output)
        slot = req.mamba_pool_idx.long()
        if is_attention:
            key_pool = qsa_pool.get_key_buffer(layer_id).reshape(
                -1, config.num_key_value_heads, config.head_dim
            )
            value_pool = qsa_pool.get_value_buffer(layer_id).reshape(
                -1, config.num_key_value_heads, config.head_dim
            )
            state = (
                qsa_pool.get_qsa_key_state_buffer(layer_id)[:4]
                .detach().cpu().clone(),
                qsa_pool.get_qsa_rope_position_buffer(
                    torch.arange(4, device="npu:0")
                ).detach().cpu().clone(),
                qsa_pool.get_qsa_compressed_k_buffer(layer_id)[1:5]
                .detach().cpu().clone(),
                key_pool[4:21].detach().cpu().clone(),
                value_pool[4:21].detach().cpu().clone(),
            )
        else:
            cache = request_pool.mamba2_layer_cache(layer_id)
            state = (
                cache.conv[0][slot].detach().cpu().clone(),
                cache.temporal[slot].detach().cpu().clone(),
            )
        result = torch.cat(outputs).detach().cpu()
        captures = dict(active_capture)
        for stage in ("cpu_partial", "npu_owned_partial", "routed_merge"):
            if captures[stage] == 0 or captures[stage + "_nonzero"] == 0:
                raise AssertionError(f"actual KTEP stage did not run with nonzero output: {stage}")
        return result, state, captures
    finally:
        active_capture = None
        finish_request(req)


report = {
    "status": "RUNNING",
    "scope": (
        f"REAL_LAYER{layer_id}_"
        f"{'QSA' if is_attention else 'GDN'}_GR_ROUTER_TOPK_KTEP_SHARED_"
        "SYNTHETIC_HIDDEN_ONLY"
    ),
    "layer_id": layer_id,
    "layer_type": "QSA_ATTENTION" if is_attention else "GDN_LINEAR_ATTENTION",
    "not_qualified": [
        "scheduler",
        "embedding",
        "multi_layer_decoder",
        "logits",
        "full_model",
        "quality",
        "performance",
    ],
    "source_identity": identity,
    "bank_catalog": {
        "path": str(catalog_path),
        "sha256": catalog_sha256,
        "selected_layer": layer_id,
        "selected_file": selected_bank["file"],
        "selected_sha256_from_catalog": selected_bank["sha256"],
        "selected_live_bytes": live_bank.stat().st_size,
    },
    "physical_device": 15,
    "tokens": 17,
    "paths": PATHS,
    "repeats": 3,
    "relative_l2_gate": 0.01,
    "loader": {
        "checkpoint_source_tensors": len(checkpoint_keys),
        "runtime_parameter_destinations": len(params),
        "loaded_destinations": len(loaded),
        "expert_source_shapes": {key: list(value) for key, value in expert_source_shapes.items()},
        "expert_loading": "per-logical-expert safe_open slice CPU clone through parameter weight_loader",
        "postprocessed_modules": postprocessed,
        "a_log_dtype": (
            str(layer.linear_attn.A_log.dtype) if not is_attention else None
        ),
    },
    "ktep": {
        "quant_method": type(quant_method).__name__,
        "resident_experts": quant_method.num_gpu_experts,
        "logical_experts": config.num_experts,
        "cpu_workers": quant_method.kt_config.cpuinfer_threads,
        "threadpools": quant_method.kt_config.threadpool_count,
        "numa_nodes": quant_method.kt_config.numa_nodes,
        "method": quant_method.kt_config.method,
        "deferred_experts_per_token": quant_method.kt_config.max_deferred_experts_per_token,
        "dynamic_update": quant_method.kt_config.kt_enable_dynamic_expert_update,
        "placement": server_args.kt_expert_placement_strategy,
        "plugin_import_timing": "after_real_qwen4_decoder_constructor",
        "plugin_module_present_before_constructor": (
            ktep_module_present_before_constructor
        ),
        "constructor_created_runner": True,
        "runner_recreated_by_probe": False,
    },
    "rows": [],
    "same_path_hashes": {},
}

try:
    baseline_output = None
    baseline_state = None
    with torch.no_grad(), patch.object(kt_ep, "_r61_capture", side_effect=capture):
        for path, parts in PATHS.items():
            hashes = []
            for repeat in range(report["repeats"]):
                output, state, captures = run_path(parts)
                if baseline_output is None:
                    baseline_output = output
                    baseline_state = state
                output_error = relative_l2(output, baseline_output)
                state_errors = []
                for actual, expected in zip(state, baseline_state):
                    if actual.dtype in (torch.int32, torch.int64):
                        if not torch.equal(actual, expected):
                            raise AssertionError(
                                f"layer-{layer_id} integer request state mismatch"
                            )
                        state_errors.append(0.0)
                    else:
                        state_errors.append(relative_l2(actual, expected))
                state_error = max(state_errors)
                row = {
                    "path": path,
                    "repeat": repeat,
                    "output_relative_l2": output_error,
                    "state_relative_l2": state_error,
                    "ktep_capture": captures,
                }
                report["rows"].append(row)
                if not math.isfinite(max(output_error, state_error)) or max(
                    output_error, state_error
                ) > 0.01:
                    raise AssertionError(row)
                hashes.append(tensor_hash(output))
                print(json.dumps(row), flush=True)
            if len(set(hashes)) != 1:
                raise AssertionError(
                    f"decoder layer-{layer_id} {path} changed across request reuse"
                )
            report["same_path_hashes"][path] = hashes

    if not is_attention:
        # The last free must also clear both selected GDN state tensors before
        # the slot can be admitted for another request.
        reset_req = start_request()
        finish_request(reset_req)

    torch.npu.synchronize()
    report.update(
        status=f"PASS_REAL_LAYER{layer_id}_END_TO_END_COMPONENT",
        rows_count=len(report["rows"]),
        post_run_request_reset=(
            "POOL_FREE_ZEROED_GDN"
            if not is_attention
            else "LOGICAL_ZERO_RESET_EXERCISED_EACH_RUN"
        ),
        peak_npu_allocated=torch.npu.max_memory_allocated(),
    )
except BaseException as exc:
    report.update(status="FAIL", error=repr(exc))
    raise
finally:
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    kt_kernel_ext.shutdown_ascend_callback_worker()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.npu.empty_cache()

print(json.dumps({"status": report["status"], "rows": len(report["rows"])}), flush=True)
