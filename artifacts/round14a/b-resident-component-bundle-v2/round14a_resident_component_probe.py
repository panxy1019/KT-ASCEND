#!/usr/bin/env python3
"""Real Qwen3.8 component probe for Round14a Q32/Q42/Q512 residency."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import time
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch_npu
from safetensors import safe_open


REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-catalog", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profiles-sha256", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from round14a_resident_runtime import load_profile
    profile = load_profile(args.profiles, args.profiles_sha256, args.strategy)
    selected = profile["rows"][args.layer]
    capacity = len(selected)

    from probe_identity import verify_probe_identity
    identity = verify_probe_identity(
        model=args.model, receipt=args.identity_receipt,
        expected_model=Path("/models") / REVISION,
        expected_receipt_sha256="7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239",
        revision=REVISION,
        manifest_sha256="62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e")
    bank_sha = digest(args.bank)
    catalog = json.loads(args.bank_catalog.read_text())
    if (catalog.get("status") != "PASS_48_CONVERTED_BANK_RECEIPTS_AND_LIVE_SIZES_NOT_MODEL_QUALIFIED" or
            catalog.get("storage") != "MXFP4" or len(catalog.get("banks", [])) != 48):
        raise RuntimeError("component bank catalog identity mismatch")
    bank_manifest = catalog["banks"][args.layer]
    if (bank_manifest.get("sha256") != bank_sha or
            bank_manifest.get("storage") != "MXFP4" or
            bank_manifest.get("layer") != args.layer or
            bank_manifest.get("logical_ids") != list(range(512)) or
            bank_manifest.get("source_identity", {}).get("receipt_sha256") != identity["receipt_sha256"]):
        raise RuntimeError("component bank identity mismatch")

    from kt_kernel import kt_kernel_ext
    if not kt_kernel_ext.llamafile_block32_tail_supported():
        raise RuntimeError("MXFP4 block32 tail capability missing")
    torch.set_num_threads(2)
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.18, 0)
    acl = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so")
    physical = ctypes.c_int32(-1)
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    if (acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0 or
            physical.value != 12 or torch.npu.device_count() != 1):
        raise RuntimeError("component is not isolated to physical12")

    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe import kt_ep_wrapper as ep
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    mask = torch.zeros(512, dtype=torch.bool)
    if selected:
        mask[selected] = True
    layer = torch.nn.Module()
    layer.top_k = 10
    layer.num_experts = 512
    layer.intermediate_size_per_partition = 640
    layer.moe_tp_size = 1
    runner = MoeRunnerConfig(num_experts=512, num_local_experts=capacity,
        hidden_size=2560, intermediate_size_per_partition=640,
        layer_id=args.layer, top_k=10, params_dtype=torch.bfloat16)
    layer.moe_runner_config = runner
    if capacity == 512:
        method = UnquantizedFusedMoEMethod()
    else:
        config = ep.KTConfig(layer_idx=args.layer, gpu_experts_mask=mask,
            cpuinfer_threads=128, threadpool_count=4, numa_nodes=[2, 3, 4, 7],
            weight_path=str(args.bank), chunked_prefill_size=32,
            max_deferred_experts_per_token=0, method="LLAMAFILE", num_layers=48)
        with patch.object(ep, "get_tensor_model_parallel_rank", return_value=0):
            method = ep.KTEPWrapperMethod(UnquantizedFusedMoEMethod(), config)
    with torch.device("npu:0"):
        method.create_weights(layer, 512, 2560, 640, torch.bfloat16)

    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts."
    gate_path = args.model / index[prefix + "gate_up_proj"]
    down_path = args.model / index[prefix + "down_proj"]
    def read_expert(expert):
        with safe_open(str(gate_path), framework="pt", device="cpu") as source:
            gate_up = source.get_slice(prefix + "gate_up_proj")[expert].clone()
        with safe_open(str(down_path), framework="pt", device="cpu") as source:
            down = source.get_slice(prefix + "down_proj")[expert].clone()
        return gate_up, down
    with torch.no_grad():
        for local, expert in enumerate(selected):
            gate_up, down = read_expert(expert)
            layer.w13_weight[local].copy_(gate_up.to("npu:0"))
            layer.w2_weight[local].copy_(down.to("npu:0"))
    method.process_weights_after_loading(layer)
    method.create_moe_runner(layer, runner)
    formats = {name: int(torch_npu.get_npu_format(getattr(layer, name)))
               for name in ("w13_weight", "w2_weight")}
    if capacity > 0 and set(formats.values()) != {29}:
        raise RuntimeError("component did not preserve NZ29")
    expected_shapes = ([capacity, 2560, 1280], [capacity, 640, 2560])
    if list(layer.w13_weight.shape) != expected_shapes[0] or list(layer.w2_weight.shape) != expected_shapes[1]:
        raise RuntimeError("component physical capacity mismatch")
    if capacity == 0 and (layer.w13_weight.numel() or layer.w2_weight.numel()):
        raise RuntimeError("Q0 unexpectedly retained NPU expert storage")

    selected_set = set(selected)
    resident = selected[:10]
    cpu = [expert for expert in range(512) if expert not in selected_set][:10]
    if capacity == 0:
        routes = cpu
    elif capacity == 512:
        routes = resident
    else:
        routes = (resident[:5] + cpu[:5])
    if len(routes) != 10:
        raise RuntimeError("component could not form a ten-route fixture")

    import gguf
    from gguf.quants import dequantize
    native = ctypes.CDLL(kt_kernel_ext.__file__)
    for symbol in ("quantize_row_q8_0", "dequantize_row_q8_0"):
        getattr(native, symbol).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    def q8_activation(value):
        value = value.float().contiguous()
        raw = torch.empty(value.numel() // 32 * 34, dtype=torch.uint8)
        restored = torch.empty_like(value)
        native.quantize_row_q8_0(value.data_ptr(), raw.data_ptr(), value.numel())
        native.dequantize_row_q8_0(raw.data_ptr(), restored.data_ptr(), value.numel())
        return restored
    reader = gguf.GGUFReader(str(args.bank))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    weights = {}
    for expert in routes:
        if expert in selected_set:
            gate_up, down = read_expert(expert)
            gate, up = gate_up.float().chunk(2, 0)
            weights[expert] = (gate, up, down.float(), "BF16")
        else:
            pieces = []
            for name in ("gate", "up", "down"):
                tensor = tensors[f"blk.{args.layer}.ffn_{name}_exps.weight"]
                raw = tensor.data.reshape(512, -1)[expert]
                shape = (2560, 640) if name == "down" else (640, 2560)
                pieces.append(torch.from_numpy(
                    dequantize(raw, tensor.tensor_type).copy()).reshape(shape))
            weights[expert] = (*pieces, "MXFP4")

    torch.manual_seed(131300 + args.layer + capacity)
    hidden = torch.randn(2, 2560).bfloat16()
    route_weights = torch.softmax(torch.randn(2, 10), -1).float()
    route_ids = torch.tensor(routes, dtype=torch.long).repeat(2, 1)
    expected = torch.zeros(2, 2560)
    for column, expert in enumerate(routes):
        gate, up, down, precision = weights[expert]
        reference_hidden = q8_activation(hidden) if precision == "MXFP4" else hidden.float()
        gate_out = (reference_hidden @ gate.T).bfloat16().float()
        up_out = (reference_hidden @ up.T).bfloat16().float()
        activated = (torch.nn.functional.silu(gate_out) * up_out).bfloat16().float()
        if precision == "MXFP4":
            activated = q8_activation(activated)
        value = activated @ down.T
        if precision == "BF16":
            value = value.bfloat16().float()
        expected.add_(value * route_weights[:, column, None].bfloat16().float())
    dispatch = StandardDispatchOutput(hidden_states=hidden.to("npu:0"), hidden_states_scale=None,
        topk_output=StandardTopKOutput(topk_weights=route_weights.to("npu:0"),
            topk_ids=route_ids.to("npu:0"), router_logits=None))
    # Two unmeasured calls establish the native worker/allocator steady state.
    for _ in range(2):
        method.apply(layer, dispatch)
        torch.npu.synchronize()
    hashes = []
    errors = []
    full_parent_seconds = []
    allocated_trace = []
    for _ in range(5):
        torch.npu.synchronize()
        started = time.perf_counter()
        output = method.apply(layer, dispatch).hidden_states
        torch.npu.synchronize()
        full_parent_seconds.append(time.perf_counter() - started)
        actual = output.cpu().float()
        errors.append(float((actual - expected).norm() / expected.norm().clamp_min(1e-30)))
        hashes.append(hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
        allocated_trace.append(int(torch.npu.memory_allocated()))
    if max(errors) > 0.01 or len(set(hashes)) != 1 or not torch.isfinite(output).all():
        raise RuntimeError(f"component numerical/repeat gate failed: {errors}")
    if capacity < 512:
        if (method.num_gpu_experts != capacity or
                method.gpu_index_to_logical.tolist() != selected):
            raise RuntimeError("KTEP logical/reverse mapping mismatch")
        provider = "KTEP_Q0" if capacity == 0 else "KTEP_MIXED"
    else:
        provider = "PURE_NPU"
    result = {
        "schema": 1, "status": "PASS_ROUND14A_RESIDENT_COMPONENT",
        "strategy": args.strategy, "layer": args.layer, "capacity": capacity,
        "provider": provider, "profile_sha256": profile["profile_sha256"],
        "profile_document_sha256": profile["document_sha256"],
        "bank_sha256": bank_sha, "source_identity": identity,
        "numa_nodes": [2, 3, 4, 7], "workers": 128,
        "npu_formats": formats, "physical_shapes": expected_shapes,
        "route_ids": routes, "relative_l2": errors, "repeat_hashes": hashes,
        "full_parent_seconds": full_parent_seconds,
        "full_parent_median_seconds": statistics.median(full_parent_seconds),
        "allocated_trace": allocated_trace,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "allocated_bytes_before_exit": torch.npu.memory_allocated(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], args.strategy, args.layer, capacity, flush=True)


if __name__ == "__main__":
    main()
