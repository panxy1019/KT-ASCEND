#!/usr/bin/env python3
"""Real-weight Q32 BF16/NZ29 resident-MoE NPUGraph feasibility probe."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu
from safetensors import safe_open


REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
LAYER = 0
RESIDENT = 32
HIDDEN = 2560
INTERMEDIATE = 640
TOPK = 10
CYCLES = 20
TIMING_SAMPLES = 15
RELATIVE_L2_GATE = 1e-5
POST_RESET_GROWTH_BYTES = 64 * 1024**2


def tensor_hash(value):
    cpu = value.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def relative_l2(actual, expected):
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    )


def memory_point():
    gc.collect()
    torch.npu.synchronize()
    return {
        "allocated": int(torch.npu.memory_allocated()),
        "reserved": int(torch.npu.memory_reserved()),
        "max_allocated": int(torch.npu.max_memory_allocated()),
        "max_reserved": int(torch.npu.max_memory_reserved()),
    }


def measure(function):
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    began = time.perf_counter_ns()
    start.record()
    result = function()
    end.record()
    torch.npu.synchronize()
    return result, (time.perf_counter_ns() - began) / 1e6, float(start.elapsed_time(end))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=(357,), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from probe_identity import verify_probe_identity
    from round15_b_resident_graph import ResidentMoeDecodeGraph

    identity = verify_probe_identity(
        model=args.model,
        receipt=args.identity_receipt,
        expected_model=Path("/models") / REVISION,
        expected_receipt_sha256=(
            "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
        ),
        revision=REVISION,
        manifest_sha256=(
            "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
        ),
    )
    torch.set_num_threads(2)
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.18, 0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    acl = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so")
    physical = ctypes.c_int32(-1)
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    devices = sorted(Path("/dev").glob("davinci[0-9]*"))
    if (
        acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
        or physical.value != 12
        or torch.npu.device_count() != 1
        or devices != [Path("/dev/davinci12")]
    ):
        raise RuntimeError("Round15 B component is not isolated to physical12")

    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    report = {
        "schema": 1,
        "status": "RUNNING",
        "version": args.version,
        "candidate": "DECODE_BS1_RESIDENT_MOE_NPU_SEGMENT_GRAPH",
        "candidate_default": "OFF",
        "request_scope": "DECODE_ONLY_BATCH1_TP1_Q32_BF16_NZ29_TOPK_IDS_INT32",
        "graph_boundary": "UNQUANTIZED_FUSED_MOE_METHOD_APPLY_ONLY",
        "excluded_from_graph": [
            "CPU_EXPERT_SUBMIT",
            "CPU_EXPERT_WAIT",
            "CPU_NPU_MERGE",
            "PLE",
            "QSA",
            "GDN",
            "KV_STATE",
            "REQUEST_EPOCH",
            "CANCELLATION",
        ],
        "source_identity": identity,
        "physical_device": 12,
        "layer": LAYER,
        "resident_logical_experts": list(range(RESIDENT)),
        "cycles": CYCLES,
        "timing_samples": TIMING_SAMPLES,
        "relative_l2_gate": RELATIVE_L2_GATE,
        "post_reset_growth_limit_bytes": POST_RESET_GROWTH_BYTES,
        "runtime": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "graph_signature": "torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True)",
        },
        "allocator": {},
        "numerical_cycles": [],
        "timings": {},
    }
    graph = None
    try:
        report["allocator"]["initial"] = memory_point()
        runner = MoeRunnerConfig(
            num_experts=512,
            num_local_experts=RESIDENT,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            layer_id=LAYER,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
        )
        layer = torch.nn.Module()
        layer.top_k = TOPK
        layer.num_experts = 512
        layer.num_local_experts = RESIDENT
        layer.intermediate_size_per_partition = INTERMEDIATE
        layer.moe_tp_size = 1
        layer.moe_runner_config = runner
        method = UnquantizedFusedMoEMethod()
        with torch.device("npu:0"):
            method.create_weights(layer, RESIDENT, HIDDEN, INTERMEDIATE, torch.bfloat16)

        weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = f"model.language_model.layers.{LAYER}.mlp.experts."
        gate_path = args.model / weight_map[prefix + "gate_up_proj"]
        down_path = args.model / weight_map[prefix + "down_proj"]
        with safe_open(str(gate_path), framework="pt", device="cpu") as gate_source, safe_open(
            str(down_path), framework="pt", device="cpu"
        ) as down_source, torch.no_grad():
            gate_slice = gate_source.get_slice(prefix + "gate_up_proj")
            down_slice = down_source.get_slice(prefix + "down_proj")
            for expert in range(RESIDENT):
                layer.w13_weight[expert].copy_(gate_slice[expert].clone().to("npu:0"))
                layer.w2_weight[expert].copy_(down_slice[expert].clone().to("npu:0"))
        method.process_weights_after_loading(layer)
        method._kt_return_fp32_contribution = True
        method.create_moe_runner(layer, runner)
        formats = {
            name: int(torch_npu.get_npu_format(getattr(layer, name)))
            for name in ("w13_weight", "w2_weight")
        }
        shapes = {
            name: list(getattr(layer, name).shape)
            for name in ("w13_weight", "w2_weight")
        }
        if formats != {"w13_weight": 29, "w2_weight": 29} or shapes != {
            "w13_weight": [RESIDENT, HIDDEN, 2 * INTERMEDIATE],
            "w2_weight": [RESIDENT, INTERMEDIATE, HIDDEN],
        }:
            raise RuntimeError("Round15 B real resident weight identity mismatch")
        report["resident_weights"] = {
            "formats": formats,
            "shapes": shapes,
            "gate_up_shard": gate_path.name,
            "down_shard": down_path.name,
            "bytes": sum(
                getattr(layer, name).numel() * getattr(layer, name).element_size()
                for name in ("w13_weight", "w2_weight")
            ),
        }

        def fixture(cycle):
            generator = torch.Generator(device="cpu").manual_seed(1534700 + cycle)
            hidden = torch.randn(1, HIDDEN, generator=generator, dtype=torch.bfloat16)
            weights = torch.softmax(
                torch.randn(1, TOPK, generator=generator, dtype=torch.float32), dim=-1
            )
            route_patterns = (
                list(range(10)),
                [31, 29, 27, 25, 23, 21, 19, 17, 15, 13],
                [0, 3, 7, 11, 19, -1, -1, -1, -1, -1],
                [31, 0, 30, 1, 29, 2, -1, -1, 7, 8],
            )
            ids = torch.tensor([route_patterns[cycle % len(route_patterns)]], dtype=torch.int32)
            return StandardDispatchOutput(
                hidden_states=hidden.to("npu:0"),
                hidden_states_scale=None,
                topk_output=StandardTopKOutput(
                    topk_weights=weights.to("npu:0"),
                    topk_ids=ids.to("npu:0"),
                    router_logits=None,
                ),
            )

        report["production_route_dtype_contract"] = {
            "topk_ids_dtype": str(fixture(0).topk_output.topk_ids.dtype),
            "source": "ROUND15_V356_FULL_MODEL_DECODE_OBSERVATION",
            "scope": "EXACT_INT32_ONLY",
        }

        first = fixture(0)
        for _ in range(3):
            method.apply(layer, first)
        torch.npu.synchronize()
        report["allocator"]["before_capture"] = memory_point()
        graph = ResidentMoeDecodeGraph(method, layer, first)
        report["allocator"]["after_capture"] = memory_point()

        output_hashes = []
        for cycle in range(CYCLES):
            dispatch = fixture(cycle)
            eager = method.apply(layer, dispatch).hidden_states
            torch.npu.synchronize()
            eager_cpu = eager.detach().cpu().float()
            graph_output = graph.replay(dispatch).hidden_states
            torch.npu.synchronize()
            graph_cpu = graph_output.detach().cpu().float()
            error = relative_l2(graph_cpu, eager_cpu)
            maximum = float((graph_cpu - eager_cpu).abs().max())
            finite = bool(torch.isfinite(graph_cpu).all())
            eager_hash = tensor_hash(eager)
            graph_hash = tensor_hash(graph_output)
            if not finite or not math.isfinite(error) or error > RELATIVE_L2_GATE:
                raise RuntimeError(
                    f"Round15 B graph numerical mismatch at cycle {cycle}: {error}"
                )
            report["numerical_cycles"].append(
                {
                    "cycle": cycle,
                    "relative_l2": error,
                    "max_abs": maximum,
                    "finite": finite,
                    "eager_hash": eager_hash,
                    "graph_hash": graph_hash,
                    "route_ids": dispatch.topk_output.topk_ids.cpu().reshape(-1).tolist(),
                    "allocator": memory_point(),
                }
            )
            output_hashes.append(graph_hash)
            del dispatch, eager, graph_output, eager_cpu, graph_cpu
        if len(set(output_hashes)) < 4:
            raise RuntimeError("Round15 B changing inputs did not change graph outputs")

        repeat = fixture(777)
        repeat_hashes = []
        for _ in range(3):
            value = graph.replay(repeat).hidden_states
            torch.npu.synchronize()
            repeat_hashes.append(tensor_hash(value))
        if len(set(repeat_hashes)) != 1:
            raise RuntimeError("Round15 B graph repeat hash mismatch")
        report["repeat_hashes"] = repeat_hashes

        timing_dispatch = fixture(999)
        functions = {
            "eager": lambda: method.apply(layer, timing_dispatch).hidden_states,
            "graph": lambda: graph.replay(timing_dispatch).hidden_states,
        }
        for name in ("eager", "graph", "graph", "eager"):
            measure(functions[name])
        host = {"eager": [], "graph": []}
        device = {"eager": [], "graph": []}
        pattern = ("eager", "graph", "graph", "eager")
        step = 0
        while any(len(values) < TIMING_SAMPLES for values in host.values()):
            name = pattern[step % len(pattern)]
            step += 1
            if len(host[name]) >= TIMING_SAMPLES:
                continue
            _output, host_ms, device_ms = measure(functions[name])
            host[name].append(host_ms)
            device[name].append(device_ms)
        eager_host = statistics.median(host["eager"])
        graph_host = statistics.median(host["graph"])
        eager_device = statistics.median(device["eager"])
        graph_device = statistics.median(device["graph"])
        report["timings"] = {
            "host_ms": host,
            "device_ms": device,
            "eager_host_median_ms": eager_host,
            "graph_host_median_ms": graph_host,
            "graph_over_eager_host": graph_host / eager_host,
            "eager_device_median_ms": eager_device,
            "graph_device_median_ms": graph_device,
            "graph_over_eager_device": graph_device / eager_device,
            "candidate_includes_static_input_copies": True,
        }
        before_reset = report["allocator"]["before_capture"]
        graph.reset()
        graph = None
        del repeat, timing_dispatch, first
        report["allocator"]["after_reset"] = memory_point()
        reset_growth = max(
            0,
            report["allocator"]["after_reset"]["allocated"]
            - before_reset["allocated"],
        )
        report["allocator"]["post_reset_growth_bytes"] = reset_growth
        if reset_growth > POST_RESET_GROWTH_BYTES:
            raise RuntimeError(
                f"Round15 B graph reset retained {reset_growth} allocated bytes"
            )
        report["component_gate_pass"] = True
        report["status"] = "PASS_ROUND15_B_GRAPH_COMPONENT_PROCESS"
        report["disposition"] = (
            "OPEN_DIRECTIONAL_MODEL_GATE"
            if graph_host / eager_host <= 0.90 and graph_device / eager_device <= 0.90
            else "NO_COMPONENT_GAIN_CLOSE_B"
        )
    finally:
        if graph is not None:
            graph.reset()
        report["peak_npu_allocated"] = int(torch.npu.max_memory_allocated())
        report["peak_npu_reserved"] = int(torch.npu.max_memory_reserved())
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report.get("disposition"), flush=True)


if __name__ == "__main__":
    main()
