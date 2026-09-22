"""Real f02 init_memory_pool dispatch probe for the bounded Round9 profile.

This allocates tiny NPU pools and calls ModelRunnerKVCacheMixin.init_memory_pool
itself.  The synthetic state dimensions keep the probe bounded; it verifies
pool selection and wiring, not model numerics or full-model memory capacity.
"""
import argparse
import gc
import ctypes
from unittest.mock import patch
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch_npu


torch.npu.set_device(0)
acl = ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
phy = ctypes.c_int32(-1)
assert acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(phy)) == 0 and phy.value == 15
if torch.npu.device_count() != 1:
    raise RuntimeError("dispatch probe requires one isolated logical NPU")

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.round9_qsa_pool import Round9QSAKVPool
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)


FULL_LAYERS = list(range(3, 48, 4))
LINEAR_LAYERS = [layer for layer in range(48) if layer not in FULL_LAYERS]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def server_args():
    return NS(
        attention_backend="ascend",
        disable_radix_cache=True,
        disaggregation_mode="null",
        enable_double_sparsity=False,
        enable_dp_attention=False,
        enable_memory_saver=False,
        enable_mamba_extra_buffer=lambda: False,
        enable_pdmux=False,
        max_mamba_cache_size=1,
        max_running_requests=1,
        max_total_tokens=32,
        mem_fraction_static=0.9,
        page_size=4,
        speculative_algorithm=None,
        speculative_num_draft_tokens=None,
    )


def base_runner(model_config, mamba_config):
    runner = ModelRunnerKVCacheMixin()
    runner.server_args = server_args()
    runner.model_config = model_config
    runner.mambaish_config = mamba_config
    runner.hybrid_gdn_config = mamba_config
    runner.profile_max_num_token = lambda: 32
    runner.spec_algorithm = NS(is_none=lambda: True)
    runner.device = "npu"
    runner.gpu_id = 0
    runner.kv_cache_dtype = torch.bfloat16
    runner.dtype = torch.bfloat16
    runner.page_size = 4
    runner.start_layer = 0
    runner.end_layer = 48
    runner.num_effective_layers = 48
    runner.pp_size = 1
    runner.dp_size = 1
    runner.is_draft_worker = False
    runner.is_hybrid_swa = False
    runner.is_hybrid_swa_compress = False
    runner.enable_hisparse = False
    runner.use_mla_backend = False
    runner.req_to_token_pool = None
    runner.token_to_kv_pool = None
    runner.token_to_kv_pool_allocator = None
    return runner


def qwen4_runner():
    cache_params = NS(
        shape=NS(conv=[(16, 3)], temporal=(2, 4, 4)),
        dtype=NS(conv=torch.bfloat16, temporal=torch.float32),
        layers=LINEAR_LAYERS,
        mamba_cache_per_req=1,
    )
    config = NS(
        model_type="qwen4_exp_text",
        full_attention_layer_ids=FULL_LAYERS,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        short_conv_layer_ids=[1],
        short_conv_state_shape=(16, 9),
        ngram_context_len=2,
        eos_token_id=248044,
        mamba2_cache_params=cache_params,
    )
    hf = NS(
        architectures=["Qwen4ExpForConditionalGeneration"],
        model_type="qwen4_exp_text",
    )
    model = NS(
        context_len=32,
        head_dim=256,
        hf_config=hf,
        is_hybrid_swa=False,
        get_num_kv_heads=lambda _tp: 2,
    )
    return base_runner(model, config)


def ordinary_runner():
    hf = NS(architectures=["LlamaForCausalLM"], model_type="llama")
    model = NS(
        context_len=32,
        head_dim=256,
        hf_config=hf,
        is_hybrid_swa=False,
        get_num_kv_heads=lambda _tp: 2,
    )
    return base_runner(model, None)


def exercise_qwen4_dispatch():
    runner = qwen4_runner()
    with patch("sglang.srt.model_executor.model_runner_kv_cache_mixin.get_attention_tp_size", return_value=1):
        runner.init_memory_pool()
    req_pool = runner.req_to_token_pool
    kv_pool = runner.token_to_kv_pool
    require(type(req_pool) is HybridReqToTokenPool, "Qwen4 request pool is not hybrid")
    require(type(kv_pool) is Round9QSAKVPool, "Qwen4 Ascend did not select Round9QSAKVPool")
    require(kv_pool.full_layer_nums == 12, "Qwen4 QSA pool does not contain 12 attention layers")
    require(set(kv_pool.full_attention_layer_id_mapping) == set(FULL_LAYERS),
            "Qwen4 QSA layer mapping is wrong")
    require(type(kv_pool.full_kv_pool) is NPUMHATokenToKVPool,
            "Round9 ordinary KV storage is not the native NPU pool")
    require(kv_pool.full_kv_pool.layer_num == 12, "ordinary KV allocated a non-attention layer")
    require(kv_pool.mamba_pool is req_pool.mamba_pool, "QSA and request pools do not share Mamba state")

    slot = req_pool.mamba_pool.alloc(1)
    require(slot is not None and slot.numel() == 1, "bounded Mamba slot allocation failed")
    req_pool.mamba2_layer_cache(LINEAR_LAYERS[0]).temporal[slot] = 3
    req_pool.short_conv_layer_cache(1)[slot] = 5
    req_pool.set_ngram_context(
        slot, torch.tensor([[7, 8]], dtype=torch.long, device="npu:0")
    )
    require(torch.count_nonzero(req_pool.mamba2_layer_cache(LINEAR_LAYERS[0]).temporal[slot]).item() > 0,
            "Mamba state is not writable")
    require(torch.count_nonzero(req_pool.short_conv_layer_cache(1)[slot]).item() > 0,
            "PLE short-conv state is not writable")
    require(torch.equal(req_pool.get_ngram_context(slot),
                        torch.tensor([[7, 8]], dtype=torch.long, device="npu:0")),
            "PLE ngram state is not writable")

    ring_location = torch.tensor([0], dtype=torch.long, device="npu:0")
    ring_value = torch.ones((1, 1, 128), dtype=torch.bfloat16, device="npu:0")
    kv_pool.set_qsa_key_state_buffer(FULL_LAYERS[0], ring_location, ring_value)
    require(torch.equal(kv_pool.get_qsa_key_state_buffer(FULL_LAYERS[0])[ring_location], ring_value),
            "QSA side-buffer API is not usable")
    result = {
        "request_pool": type(req_pool).__name__,
        "token_pool": type(kv_pool).__name__,
        "ordinary_kv_pool": type(kv_pool.full_kv_pool).__name__,
        "full_attention_layers": sorted(kv_pool.full_attention_layer_id_mapping),
        "max_total_tokens": runner.max_total_num_tokens,
        "mamba_slots": req_pool.mamba_pool.size,
    }
    del runner, req_pool, kv_pool, slot, ring_location, ring_value
    return result


def exercise_ordinary_ascend_control():
    runner = ordinary_runner()
    with patch("sglang.srt.model_executor.model_runner_kv_cache_mixin.get_attention_tp_size", return_value=1):
        runner.init_memory_pool()
    require(type(runner.req_to_token_pool) is ReqToTokenPool,
            "ordinary Ascend unexpectedly selected a hybrid request pool")
    require(type(runner.token_to_kv_pool) is NPUMHATokenToKVPool,
            "ordinary Ascend did not retain NPUMHATokenToKVPool")
    require(runner.token_to_kv_pool.layer_num == 48,
            "ordinary Ascend pool does not contain all 48 layers")
    require(not hasattr(runner.token_to_kv_pool, "get_qsa_compressed_k_buffer"),
            "ordinary Ascend pool unexpectedly exposes Round9 QSA state")
    result = {
        "request_pool": type(runner.req_to_token_pool).__name__,
        "token_pool": type(runner.token_to_kv_pool).__name__,
        "layers": runner.token_to_kv_pool.layer_num,
        "max_total_tokens": runner.max_total_num_tokens,
    }
    del runner
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

qwen4 = exercise_qwen4_dispatch()
torch.npu.synchronize()
gc.collect()
torch.npu.empty_cache()
ordinary = exercise_ordinary_ascend_control()
torch.npu.synchronize()

mixin_path = Path(inspect.getsourcefile(ModelRunnerKVCacheMixin)).resolve()
qsa_path = Path(inspect.getsourcefile(Round9QSAKVPool)).resolve()
report = {
    "status": "PASS",
    "scope": "REAL_INIT_MEMORY_POOL_DISPATCH_WITH_BOUNDED_SYNTHETIC_STATE_SHAPES",
    "device": str(torch.npu.current_device()),
    "mixin_source": str(mixin_path),
    "mixin_sha256": hashlib.sha256(mixin_path.read_bytes()).hexdigest(),
    "qsa_pool_source": str(qsa_path),
    "qsa_pool_sha256": hashlib.sha256(qsa_path.read_bytes()).hexdigest(),
    "qwen4_ascend": qwen4,
    "ordinary_ascend_negative_control": ordinary,
    "limitations": [
        "synthetic small state dimensions and 32-token pools; TP size supplied as1 without distributed group",
        "dispatch/lifecycle boundary only; no model weights or attention numerics",
        "not a full-model, scheduler, latency, throughput, or memory-peak qualification",
    ],
}
payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
with args.output.open("x") as output:
    output.write(payload)
print(payload, end="")

del ordinary
gc.collect()
torch.npu.empty_cache()
