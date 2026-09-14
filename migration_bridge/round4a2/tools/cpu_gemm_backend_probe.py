#!/usr/bin/env python3
"""Compare isolated CPU CBLAS backends with captured Ascend expert stages."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper


CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112
PROJECTIONS = ("gate", "up", "down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument(
        "--capture-pair",
        action="append",
        required=True,
        help="LABEL:HYBRID_CAPTURE_DIR:NPU_CAPTURE_DIR; may be repeated",
    )
    parser.add_argument(
        "--backend",
        action="append",
        required=True,
        help="NAME:LIBRARY_PATH:THREADS; may be repeated",
    )
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--max-samples-per-expert", default=100, type=int)
    parser.add_argument("--min-samples-per-expert", default=3, type=int)
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    difference = actual - reference
    reference_norm = float(torch.linalg.vector_norm(reference))
    exact = actual == reference
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": (
            float(torch.linalg.vector_norm(difference)) / reference_norm
            if reference_norm
            else 0.0
        ),
        "cosine": float(
            F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
        ),
        "num_diff_elements": int((~exact).sum()),
        "num_exact_elements": int(exact.sum()),
        "num_elements": actual.numel(),
    }


def bf16_bucket_analysis(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual = actual.detach().to(torch.bfloat16).cpu()
    reference = reference.detach().to(torch.bfloat16).cpu()
    difference = (actual.float() - reference.float()).abs()
    positive_inf = torch.full_like(reference, float("inf"))
    negative_inf = torch.full_like(reference, float("-inf"))
    step_up = (torch.nextafter(reference, positive_inf).float() - reference.float()).abs()
    step_down = (reference.float() - torch.nextafter(reference, negative_inf).float()).abs()
    local_step = torch.minimum(step_up, step_down)
    finite_nonzero = torch.isfinite(local_step) & (local_step > 0)
    ratio = torch.zeros_like(difference)
    ratio[finite_nonzero] = difference[finite_nonzero] / local_step[finite_nonzero]
    nonzero = difference > 0
    return {
        "exact": int((~nonzero).sum()),
        "within_1_step": int((nonzero & (ratio <= 1.01)).sum()),
        "within_2_steps": int((nonzero & (ratio > 1.01) & (ratio <= 2.01)).sum()),
        "over_2_steps": int((nonzero & (ratio > 2.01)).sum()),
        "max_local_step_ratio": float(ratio.max()),
    }


def load_weights(
    model_dir: Path, weight_map: dict[str, str], layer: int, expert: int
) -> dict[str, torch.Tensor]:
    keys = {
        projection: (
            f"model.layers.{layer}.mlp.experts.{expert}."
            f"{projection}_proj.weight"
        )
        for projection in PROJECTIONS
    }
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for projection, key in keys.items():
        by_shard.setdefault(weight_map[key], []).append((projection, key))
    weights = {}
    for shard, entries in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as reader:
            for projection, key in entries:
                weights[projection] = reader.get_tensor(key).contiguous()
    return weights


@dataclass(frozen=True)
class CapturePair:
    label: str
    hybrid_dir: Path
    npu_dir: Path


@dataclass(frozen=True)
class Sample:
    label: str
    capture_file: str
    pass_index: int
    token_index: int
    expert: int
    hidden: torch.Tensor
    npu_stages: dict[str, torch.Tensor]


class CBlasBackend:
    def __init__(self, name: str, library: Path, threads: int):
        self.name = name
        self.library = library.resolve()
        self.threads = threads
        self.handle = ctypes.CDLL(str(self.library))
        self.sgemm = self.handle.cblas_sgemm
        self.sgemm.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.sgemm.restype = None
        self._set_threads()

    def _set_threads(self) -> None:
        for symbol in ("openblas_set_num_threads", "bli_thread_set_num_threads"):
            setter = getattr(self.handle, symbol, None)
            if setter is not None:
                setter.argtypes = [ctypes.c_int]
                setter.restype = None
                setter(self.threads)

    def gemm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Return X @ W.T in FP32 using row-major CBLAS SGEMM."""
        x_array = np.ascontiguousarray(x.detach().float().cpu().numpy())
        weight_array = np.ascontiguousarray(weight.detach().float().cpu().numpy())
        if x_array.ndim != 2 or weight_array.ndim != 2:
            raise ValueError("GEMM operands must be rank two")
        m, k = x_array.shape
        n, weight_k = weight_array.shape
        if k != weight_k:
            raise ValueError(f"incompatible GEMM shapes: {x_array.shape}, {weight_array.shape}")
        output = np.empty((m, n), dtype=np.float32)
        self._set_threads()
        self.sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m,
            n,
            k,
            ctypes.c_float(1.0),
            x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k,
            weight_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k,
            ctypes.c_float(0.0),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        return torch.from_numpy(output.copy())

    def expert(self, hidden: torch.Tensor, weights: dict[str, torch.Tensor]) -> dict:
        gate_fp32 = self.gemm(hidden, weights["gate"])
        up_fp32 = self.gemm(hidden, weights["up"])
        gate = gate_fp32.to(torch.bfloat16)
        up = up_fp32.to(torch.bfloat16)
        multiplied = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        down_fp32 = self.gemm(multiplied, weights["down"])
        down = down_fp32.to(torch.bfloat16)
        return {
            "gate_fp32": gate_fp32,
            "up_fp32": up_fp32,
            "gate": gate,
            "up": up,
            "multiplied": multiplied,
            "down_fp32": down_fp32,
            "down": down,
        }


def parse_capture_pair(value: str) -> CapturePair:
    label, hybrid, npu = value.split(":", 2)
    return CapturePair(label=label, hybrid_dir=Path(hybrid), npu_dir=Path(npu))


def parse_backend(value: str) -> tuple[str, Path, int]:
    name, library, threads = value.rsplit(":", 2)
    thread_count = int(threads)
    if thread_count <= 0:
        raise ValueError("backend thread count must be positive")
    return name, Path(library), thread_count


def captured_npu_stages(
    payload: dict, expert: int, hidden: torch.Tensor
) -> dict[str, torch.Tensor] | None:
    counts = payload["expert_tokens"].to(torch.int64)
    start = int(counts[:expert].sum())
    end = start + int(counts[expert])
    routed = payload["routed_input"][start:end]
    matches = torch.where((routed == hidden).all(dim=1))[0]
    if matches.numel() == 0:
        return None
    row = start + int(matches[0])
    gate, up = payload["gmm1_gate_up"][row : row + 1].chunk(2, dim=-1)
    return {
        "gate": gate.cpu(),
        "up": up.cpu(),
        "multiplied": payload["swiglu"][row : row + 1].cpu(),
        "down": payload["gmm2_down"][row : row + 1].cpu(),
    }


def collect_samples(
    capture_pairs: list[CapturePair],
    experts: list[int],
    maximum: int,
    minimum: int,
    layer: int,
) -> list[Sample]:
    samples: dict[int, list[Sample]] = {expert: [] for expert in experts}
    for pair in capture_pairs:
        npu_by_pass = {}
        for path in sorted(pair.npu_dir.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload["layer"]) == layer:
                npu_by_pass[int(payload["pass"])] = payload
        for path in sorted(pair.hybrid_dir.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload["layer"]) != layer:
                continue
            pass_index = int(payload["pass"])
            npu_payload = npu_by_pass.get(pass_index)
            if npu_payload is None:
                continue
            for token_index, hidden in enumerate(payload["hidden_states"].split(1)):
                routed = set(int(value) for value in payload["topk_ids"][token_index].tolist())
                for expert in sorted(routed.intersection(samples)):
                    if len(samples[expert]) >= maximum:
                        continue
                    npu_stages = captured_npu_stages(npu_payload, expert, hidden)
                    if npu_stages is None:
                        continue
                    samples[expert].append(
                        Sample(
                            label=pair.label,
                            capture_file=path.name,
                            pass_index=pass_index,
                            token_index=token_index,
                            expert=expert,
                            hidden=hidden.contiguous(),
                            npu_stages=npu_stages,
                        )
                    )
    missing = {expert: len(rows) for expert, rows in samples.items() if len(rows) < minimum}
    if missing:
        raise RuntimeError(f"insufficient real captured samples: {missing}")
    return [sample for expert in experts for sample in samples[expert]]


def make_llamafile_wrapper(
    model_config: dict, layer: int, experts: list[int], gguf: Path
) -> KTMoEWrapper:
    accelerator_mask = torch.ones(int(model_config["n_routed_experts"]), dtype=torch.bool)
    accelerator_mask[experts] = False
    LlamafileMoEWrapper._gguf_loader_instance = None
    LlamafileMoEWrapper._gguf_loader_path = None
    wrapper = KTMoEWrapper(
        layer_idx=layer,
        num_experts=int(model_config["n_routed_experts"]),
        num_experts_per_tok=int(model_config["num_experts_per_tok"]),
        hidden_size=int(model_config["hidden_size"]),
        moe_intermediate_size=int(model_config["moe_intermediate_size"]),
        gpu_experts_mask=accelerator_mask,
        cpuinfer_threads=16,
        threadpool_count=1,
        weight_path=str(gguf.resolve()),
        chunked_prefill_size=16,
        max_deferred_experts_per_token=0,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    wrapper.load_weights(
        torch.arange(int(model_config["n_routed_experts"]), dtype=torch.int32)
    )
    return wrapper


def run_llamafile(
    wrapper: KTMoEWrapper, sample: Sample, topk: int, repeats: int
) -> tuple[torch.Tensor, dict]:
    ids = torch.full((1, topk), -1, dtype=torch.int64)
    ids[0, 0] = sample.expert
    route_weights = torch.zeros_like(ids, dtype=torch.float32)
    route_weights[0, 0] = 1.0
    outputs = []
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = wrapper.forward(sample.hidden, ids, route_weights).clone()
        durations.append(time.perf_counter() - started)
        outputs.append(output)
    deterministic = all(torch.equal(outputs[0], output) for output in outputs[1:])
    return outputs[0], {
        "deterministic": deterministic,
        "repeat_count": repeats,
        "median_seconds": statistics.median(durations),
        "output_sha256": tensor_sha256(outputs[0]),
    }


def main() -> None:
    args = parse_args()
    if (
        args.max_samples_per_expert <= 0
        or args.min_samples_per_expert <= 0
        or args.min_samples_per_expert > args.max_samples_per_expert
        or args.repeats < 10
    ):
        raise ValueError("sample bounds must be valid and repeats must be >= 10")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))

    experts = [int(value) for value in args.experts.split(",") if value]
    capture_pairs = [parse_capture_pair(value) for value in args.capture_pair]
    backend_specs = [parse_backend(value) for value in args.backend]
    model_dir = args.model_dir.resolve()
    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    weights_by_expert = {
        expert: load_weights(model_dir, weight_map, args.layer, expert)
        for expert in experts
    }
    print(f"loaded_weights={experts}", flush=True)
    samples = collect_samples(
        capture_pairs,
        experts,
        args.max_samples_per_expert,
        args.min_samples_per_expert,
        args.layer,
    )
    print(f"collected_samples={len(samples)}", flush=True)
    backends = [CBlasBackend(*spec) for spec in backend_specs]
    print(f"loaded_backends={[backend.name for backend in backends]}", flush=True)
    wrapper = make_llamafile_wrapper(model_config, args.layer, experts, args.gguf)
    print("llamafile_wrapper_ready", flush=True)

    operand_manifest = {}
    for expert, weights in weights_by_expert.items():
        operand_manifest[str(expert)] = {
            projection: {
                "shape": list(weight.shape),
                "source_dtype": str(weight.dtype),
                "bf16_value_sha256": tensor_sha256(weight.to(torch.bfloat16)),
                "f32_materialized_sha256": tensor_sha256(weight.float()),
            }
            for projection, weight in weights.items()
        }

    rows = []
    for sample_index, sample in enumerate(samples):
        weights = weights_by_expert[sample.expert]
        print(
            f"sample={sample_index} E{sample.expert} stage=llamafile-start",
            flush=True,
        )
        llamafile, llamafile_repeat = run_llamafile(
            wrapper, sample, int(model_config["num_experts_per_tok"]), args.repeats
        )
        print(
            f"sample={sample_index} E{sample.expert} stage=llamafile-done",
            flush=True,
        )
        row = {
            "sample_index": sample_index,
            "source": sample.label,
            "capture_file": sample.capture_file,
            "pass": sample.pass_index,
            "token_index": sample.token_index,
            "expert": sample.expert,
            "hidden_shape": list(sample.hidden.shape),
            "hidden_dtype": str(sample.hidden.dtype),
            "hidden_sha256": tensor_sha256(sample.hidden),
            "llamafile": {
                "repeat": llamafile_repeat,
                "down_vs_npu": metrics(llamafile, sample.npu_stages["down"]),
                "down_bf16_buckets": bf16_bucket_analysis(
                    llamafile, sample.npu_stages["down"]
                ),
            },
            "backends": {},
        }
        for backend in backends:
            print(
                f"sample={sample_index} E{sample.expert} "
                f"stage={backend.name}-start",
                flush=True,
            )
            outputs = []
            durations = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                stages = backend.expert(sample.hidden, weights)
                durations.append(time.perf_counter() - started)
                outputs.append(stages)
            deterministic = all(
                all(torch.equal(outputs[0][stage], output[stage]) for stage in ("gate", "up", "multiplied", "down"))
                for output in outputs[1:]
            )
            stages = outputs[0]
            stage_metrics = {}
            for stage in ("gate", "up", "multiplied", "down"):
                stage_metrics[stage] = {
                    "vs_npu": metrics(stages[stage], sample.npu_stages[stage]),
                    "bf16_buckets": bf16_bucket_analysis(
                        stages[stage], sample.npu_stages[stage]
                    ),
                }
            row["backends"][backend.name] = {
                "deterministic": deterministic,
                "repeat_count": args.repeats,
                "median_seconds": statistics.median(durations),
                "down_sha256": tensor_sha256(stages["down"]),
                "stage_metrics": stage_metrics,
                "down_vs_llamafile": metrics(stages["down"], llamafile),
            }
            print(
                f"sample={sample_index} E{sample.expert} "
                f"stage={backend.name}-done",
                flush=True,
            )
        rows.append(row)
        print(
            f"sample={sample_index} E{sample.expert} "
            + " ".join(
                f"{name}={data['stage_metrics']['down']['vs_npu']['relative_l2']:.6g}"
                for name, data in row["backends"].items()
            )
            + f" llamafile={row['llamafile']['down_vs_npu']['relative_l2']:.6g}",
            flush=True,
        )

    summaries = {}
    for backend in backends:
        down_values = [
            row["backends"][backend.name]["stage_metrics"]["down"]["vs_npu"]["relative_l2"]
            for row in rows
        ]
        summaries[backend.name] = {
            "library": str(backend.library),
            "library_sha256": file_sha256(backend.library),
            "threads": backend.threads,
            "sample_count": len(down_values),
            "median_down_relative_l2": statistics.median(down_values),
            "max_down_relative_l2": max(down_values),
            "all_repeats_deterministic": all(
                row["backends"][backend.name]["deterministic"] for row in rows
            ),
            "median_expert_seconds": statistics.median(
                row["backends"][backend.name]["median_seconds"] for row in rows
            ),
        }
    llamafile_values = [row["llamafile"]["down_vs_npu"]["relative_l2"] for row in rows]
    result = {
        "schema_version": 1,
        "layer": args.layer,
        "experts": experts,
        "max_samples_per_expert": args.max_samples_per_expert,
        "min_samples_per_expert": args.min_samples_per_expert,
        "sample_counts": {
            str(expert): sum(sample.expert == expert for sample in samples)
            for expert in experts
        },
        "repeats": args.repeats,
        "capture_pairs": [
            {
                "label": pair.label,
                "hybrid_dir": str(pair.hybrid_dir.resolve()),
                "npu_dir": str(pair.npu_dir.resolve()),
            }
            for pair in capture_pairs
        ],
        "operand_manifest": operand_manifest,
        "llamafile_summary": {
            "sample_count": len(llamafile_values),
            "median_down_relative_l2": statistics.median(llamafile_values),
            "max_down_relative_l2": max(llamafile_values),
            "all_repeats_deterministic": all(
                row["llamafile"]["repeat"]["deterministic"] for row in rows
            ),
            "median_expert_seconds": statistics.median(
                row["llamafile"]["repeat"]["median_seconds"] for row in rows
            ),
        },
        "backend_summaries": summaries,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "operand_manifest"}}, indent=2))


if __name__ == "__main__":
    main()
