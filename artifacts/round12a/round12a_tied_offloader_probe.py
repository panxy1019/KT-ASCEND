#!/usr/bin/env python3
"""Ascend functional gate for the tied-aware Round12a OffloaderV1."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import time

import torch


OFFLOADER_SHA256 = "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006"


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def relative_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(got.double() - want.double())
    denominator = torch.linalg.vector_norm(want.double()).clamp_min(1e-30)
    return float(numerator / denominator)


class AliasSubmodule(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter, scale: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.register_buffer("scale", scale)


class TiedProbeModule(torch.nn.Module):
    def __init__(self, width: int, device: torch.device | str = "cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(width, width, dtype=torch.bfloat16, device=device)
        )
        self.bias = torch.nn.Parameter(
            torch.randn(width, dtype=torch.bfloat16, device=device)
        )
        self.register_buffer(
            "scale",
            torch.linspace(0.5, 1.5, width, dtype=torch.bfloat16, device=device),
        )
        self.alias = AliasSubmodule(self.weight, self.scale)
        self.identity_observations = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.identity_observations.append(
            {
                "parameter": self.weight is self.alias.weight,
                "buffer": self.scale is self.alias.scale,
            }
        )
        projected = (value @ self.weight.T + value @ self.alias.weight.T) * 0.5
        scale = (self.scale + self.alias.scale) * 0.5
        return torch.tanh(projected + self.bias) * scale


def physical_device_check() -> None:
    library = ctypes.CDLL(
        "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so"
    )
    physical = ctypes.c_int32(-1)
    library.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    if (
        library.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
        or physical.value != 12
        or torch.npu.device_count() != 1
        or list(Path("/dev").glob("davinci[0-9]*")) != [Path("/dev/davinci12")]
    ):
        raise RuntimeError("OffloaderV1 component is not isolated on physical12")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 5 <= args.iterations <= 12:
        raise ValueError("iterations outside frozen component range")

    import torch_npu  # noqa: F401
    from sglang.srt.utils import is_pin_memory_available
    from sglang.srt.utils.offloader import OffloaderV1

    source = Path(
        "/overlay-python/sglang/srt/utils/offloader.py"
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != OFFLOADER_SHA256:
        raise RuntimeError("OffloaderV1 source identity drifted")

    torch.set_num_threads(2)
    torch.manual_seed(120159)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    physical_device_check()
    torch.npu.synchronize()
    allocator_baseline = int(torch.npu.memory_allocated())

    report = {
        "schema": 1,
        "status": "RUNNING",
        "backend": "SGLANG_OFFLOADER_V1_TIED_AWARE",
        "source_sha256": OFFLOADER_SHA256,
        "physical_device": 12,
        "iterations": args.iterations,
        "pin_memory_available": bool(is_pin_memory_available()),
        "allocator_baseline_bytes": allocator_baseline,
        "automatic_retries": 0,
    }
    module = None
    wrapped = None
    offloader = None
    value = None
    reference = None
    state = None
    try:
        width = 2048
        module = TiedProbeModule(width, device="npu:0")
        if not (
            module.weight is module.alias.weight
            and module.scale is module.alias.scale
        ):
            raise RuntimeError("probe aliases were not tied before OffloaderV1")
        value = torch.randn(16, width, dtype=torch.bfloat16, device="npu:0")
        with torch.no_grad():
            reference = module(value)
        reference_identity = list(module.identity_observations)
        module.identity_observations.clear()
        if reference_identity != [{"parameter": True, "buffer": True}]:
            raise RuntimeError("reference forward did not preserve aliases")
        torch.npu.synchronize()
        reference_hash = tensor_hash(reference)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in module.parameters()
        )
        cpu_hashes_before = {
            name: tensor_hash(parameter)
            for name, parameter in module.named_parameters()
        }

        offloader = OffloaderV1(cpu_offload_max_bytes=parameter_bytes + 1)
        wrapped = offloader.maybe_offload_to_cpu(module)
        if wrapped is not module:
            raise RuntimeError("OffloaderV1 unexpectedly replaced the module object")
        parameter_devices = {
            name: str(parameter.device)
            for name, parameter in module.named_parameters()
        }
        buffer_devices = {
            name: str(buffer.device) for name, buffer in module.named_buffers()
        }
        if set(parameter_devices.values()) != {"cpu"}:
            raise RuntimeError(f"OffloaderV1 parameters not on CPU: {parameter_devices}")
        if set(buffer_devices.values()) != {"npu:0"}:
            raise RuntimeError(f"OffloaderV1 buffers moved unexpectedly: {buffer_devices}")
        if offloader._cpu_offload_bytes != parameter_bytes:
            raise RuntimeError("OffloaderV1 byte accounting mismatch")
        state = module.state_dict(keep_vars=True)
        tied_state_groups = {
            "parameter": [
                name for name, tensor in state.items()
                if tensor is module.weight
            ],
            "buffer": [
                name for name, tensor in state.items()
                if tensor is module.scale
            ],
        }
        if tied_state_groups != {
            "parameter": ["weight", "alias.weight"],
            "buffer": ["scale", "alias.scale"],
        }:
            raise RuntimeError(f"unexpected tied state groups: {tied_state_groups}")

        rows = []
        for iteration in range(args.iterations):
            torch.npu.synchronize()
            started = time.perf_counter_ns()
            with torch.no_grad():
                output = module(value)
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            rel_l2 = relative_l2(output, reference)
            output_hash = tensor_hash(output)
            if output_hash != reference_hash or rel_l2 != 0.0:
                raise RuntimeError(
                    f"functional_call output mismatch at iteration {iteration}: {rel_l2}"
                )
            rows.append(
                {
                    "iteration": iteration,
                    "wall_ms": elapsed_ms,
                    "relative_l2": rel_l2,
                    "output_sha256": output_hash,
                    "allocator_bytes": int(torch.npu.memory_allocated()),
                }
            )
            del output
            gc.collect()
            torch.npu.synchronize()

        if module.identity_observations != [
            {"parameter": True, "buffer": True}
        ] * args.iterations:
            raise RuntimeError(
                f"candidate broke tied identity: {module.identity_observations}"
            )

        cpu_hashes_after = {
            name: tensor_hash(parameter)
            for name, parameter in module.named_parameters()
        }
        if cpu_hashes_after != cpu_hashes_before:
            raise RuntimeError("OffloaderV1 mutated CPU parameters")
        if any(parameter.device.type != "cpu" for parameter in module.parameters()):
            raise RuntimeError("OffloaderV1 did not preserve CPU parameter residency")

        report.update(
            status="PASS_ROUND12A_TIED_OFFLOADER_COMPONENT",
            parameter_bytes=parameter_bytes,
            offloader_accounted_bytes=offloader._cpu_offload_bytes,
            parameter_devices=parameter_devices,
            buffer_devices=buffer_devices,
            reference_sha256=reference_hash,
            repeat_hash_exact=len({row["output_sha256"] for row in rows}) == 1,
            max_relative_l2=max(row["relative_l2"] for row in rows),
            tied_state_groups=tied_state_groups,
            tied_identity_preserved=True,
            tie_weights_false_used=False,
            identity_observations=module.identity_observations,
            rows=rows,
        )
    finally:
        del state, reference, value, wrapped, module, offloader
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        report["allocator_after_cleanup_bytes"] = int(torch.npu.memory_allocated())
        report["allocator_reserved_after_cleanup_bytes"] = int(
            torch.npu.memory_reserved()
        )
        if report.get("status", "").startswith("PASS_") and report[
            "allocator_after_cleanup_bytes"
        ] != allocator_baseline:
            report["status"] = "FAIL_ROUND12_P4_OFFLOADER_ALLOCATOR_CLEANUP"
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if report["status"] != "PASS_ROUND12A_TIED_OFFLOADER_COMPONENT":
        raise RuntimeError(report["status"])
    print(report["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
