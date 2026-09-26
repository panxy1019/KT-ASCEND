"""Round10 CPU-only C128/MXFP4 probe using frozen real layer-0 activations."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


CAPTURE_SHA256 = "c837641468b9db93c422a3610af7a7ecc1f14d4bbb359841d69ac0dfd655dc6e"
BANK_SHA256 = "e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b"
NATIVE_SHA256 = "614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
NUMA_NODES = [2, 3, 4, 7]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_contract(*, capture_sha256, bank_sha256, native_sha256, nodes):
    if capture_sha256 != CAPTURE_SHA256:
        raise ValueError("real activation fixture changed")
    if bank_sha256 != BANK_SHA256:
        raise ValueError("layer0 MXFP4 bank changed")
    if native_sha256 != NATIVE_SHA256:
        raise ValueError("loaded native extension changed")
    if list(nodes) != NUMA_NODES:
        raise ValueError("C128 NUMA identity changed")
    return True


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "scope": "ROUND10_REAL_ACTIVATION_CPU_ONLY_C128_MXFP4_NATIVE_WORKER",
        "model_revision": MODEL_REVISION,
        "numa_nodes": NUMA_NODES,
        "cpu_workers": 128,
        "threadpool_count": 4,
        "rows": [],
    }
    try:
        import numpy as np
        import torch
        import kt_kernel
        from kt_kernel import KTMoEWrapper
        native = kt_kernel.kt_kernel_ext
        from safetensors import safe_open
        from runtime.lowbit_numpy import encode_blocks, decode_blocks

        torch.set_num_threads(2)
        native_path = Path(native.__file__).resolve(strict=True)
        identities = {
            "capture_sha256": digest(args.capture),
            "bank_sha256": digest(args.bank),
            "native_sha256": digest(native_path),
        }
        validate_contract(**identities, nodes=NUMA_NODES)
        report.update(identities, native_path=str(native_path))
        payload = torch.load(args.capture, map_location="cpu", weights_only=True)
        if payload.get("schema") != 1 or len(payload.get("groups", [])) != 2:
            raise ValueError("capture payload is not the frozen two-group fixture")

        # Independent activation quantization used by the current block32
        # MXFP4 worker.  The exported symbols are outside the Python wrapper.
        lib = ctypes.CDLL(str(native_path))
        for symbol in ("quantize_row_q8_0", "dequantize_row_q8_0"):
            getattr(lib, symbol).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

        def q8_activation(value):
            value = value.float().contiguous()
            if value.numel() % 32:
                raise ValueError("Q8 activation is not block32 aligned")
            raw = torch.empty(value.numel() // 32 * 34, dtype=torch.uint8)
            result = torch.empty_like(value)
            lib.quantize_row_q8_0(value.data_ptr(), raw.data_ptr(), value.numel())
            lib.dequantize_row_q8_0(raw.data_ptr(), result.data_ptr(), value.numel())
            return result

        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=512,
            num_experts_per_tok=10,
            hidden_size=2560,
            moe_intermediate_size=640,
            gpu_experts_mask=torch.zeros(512, dtype=torch.bool),
            cpuinfer_threads=128,
            threadpool_count=4,
            weight_path=str(args.bank),
            chunked_prefill_size=32,
            max_deferred_experts_per_token=0,
            method="LLAMAFILE",
            numa_nodes=NUMA_NODES,
        )
        wrapper.load_weights()

        index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.language_model.layers.0.mlp.experts."
        gate_up_path = args.model / index[prefix + "gate_up_proj"]
        down_path = args.model / index[prefix + "down_proj"]
        with safe_open(str(gate_up_path), framework="pt", device="cpu") as gate_file, safe_open(
            str(down_path), framework="pt", device="cpu"
        ) as down_file:
            gate_view = gate_file.get_slice(prefix + "gate_up_proj")
            down_view = down_file.get_slice(prefix + "down_proj")
            violations = []
            for group_index, group in enumerate(payload["groups"]):
                hidden = group["hidden_states"].contiguous()
                ids = group["topk_logical_ids"].to(torch.int64).contiguous()
                weights = group["topk_weights"].float().contiguous()
                bf16_reference = group["reference_contribution_fp32"].float().contiguous()
                if hidden.shape != (3, 2560) or ids.shape != (3, 10) or weights.shape != (3, 10):
                    raise ValueError("real group dimensions changed")
                expected = torch.zeros_like(bf16_reference)
                quantized_weight_hashes = []
                q8_input = q8_activation(hidden)
                for expert in sorted(set(ids.view(-1).tolist())):
                    gate_up = gate_view[expert].contiguous()
                    gate, up = gate_up.chunk(2, dim=0)
                    down = down_view[expert].contiguous()
                    decoded = []
                    for original in (gate, up, down):
                        encoded = encode_blocks(original.float().numpy(), "MXFP4")
                        quantized_weight_hashes.append(hashlib.sha256(encoded.tobytes()).hexdigest())
                        decoded.append(torch.from_numpy(decode_blocks(encoded, "MXFP4")).reshape(original.shape))
                    for row in range(hidden.shape[0]):
                        positions = (ids[row] == expert).nonzero(as_tuple=False).view(-1)
                        for position in positions.tolist():
                            gate_value = (q8_input[row].float() @ decoded[0].T).bfloat16().float()
                            up_value = (q8_input[row].float() @ decoded[1].T).bfloat16().float()
                            intermediate = (torch.nn.functional.silu(gate_value) * up_value).bfloat16().float()
                            down_value = q8_activation(intermediate) @ decoded[2].T
                            expected[row].add_(down_value * weights[row, position].bfloat16().float())
                    del gate_up, gate, up, down, decoded
                samples = []
                durations = []
                for _ in range(5):
                    started = time.perf_counter_ns()
                    output = wrapper.forward(hidden, ids, weights)
                    durations.append((time.perf_counter_ns() - started) / 1e6)
                    if not torch.isfinite(output).all():
                        raise RuntimeError("native worker returned non-finite output")
                    samples.append(tensor_sha256(output))
                native_error = float((output.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30))
                quantization_error = float((expected.double() - bf16_reference.double()).norm() / bf16_reference.double().norm().clamp_min(1e-30))
                row = {
                    "group": group_index,
                    "tokens": 3,
                    "logical_experts": sorted(set(ids.view(-1).tolist())),
                    "native_vs_independent_mxfp4_relative_l2": native_error,
                    "independent_mxfp4_vs_bf16_relative_l2": quantization_error,
                    "repeat_exact": len(set(samples)) == 1,
                    "output_sha256": samples[0],
                    "reference_sha256": tensor_sha256(expected),
                    "bf16_reference_sha256": tensor_sha256(bf16_reference),
                    "durations_ms": durations,
                    "steady_median_ms": float(np.median(durations[1:])),
                    "weight_block_hashes_sha256": hashlib.sha256(
                        json.dumps(quantized_weight_hashes, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                if native_error > 0.01 or quantization_error > 0.05 or not row["repeat_exact"]:
                    violations.append(row)
        if violations:
            report["precision_violations"] = violations
            raise AssertionError("one or more frozen precision gates failed")
        report["native_path"] = [
            "KTMoEWrapper(LLAMAFILE)",
            "BaseMoEWrapper.forward/submit_forward/sync_forward",
            "CPUInfer task queue",
            "llamafile::MOE forward_many/forward_one",
            "four 160-wide intermediate TP partitions",
        ]
        report["status"] = "PASS_T3_CURRENT_C128_MXFP4_NATIVE_REAL_FIXTURE"
    except BaseException as error:
        report.update(status="FAIL_T3_NATIVE_PROBE", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["ended_at"] = time.time()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
