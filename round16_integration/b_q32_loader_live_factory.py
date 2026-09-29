"""Exact-image factory for the bounded Round16-B V2 loader/layout probe."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from round16.b.resident_packed_loader import (
    PACKED_DOWN_SHAPE,
    PACKED_GATE_UP_SHAPE,
    load_selected_packed_rows,
    planned_row_reads,
    validate_packed_source_metadata,
)
from round16.b.selective_loader import plan_q32
from round16_integration.b_q32_loader_probe import (
    Q32LoaderBinding,
    Q32_V2_CONTEXT_API,
    Q32_V2_FACTORY_API,
)
from round16_integration.b_q32_preallocation_live_factory import (
    _create_raw_q32_binding,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_key(layer_id: int, projection: str) -> tuple[str, ...]:
    base = f"model.language_model.layers.{layer_id}.mlp.experts.{projection}"
    return (base, f"{base}.weight")


def _resolve_checkpoint_tensor(
    checkpoint_root: Path,
    weight_map: dict[str, str],
    candidates: tuple[str, ...],
) -> tuple[str, Path]:
    for key in candidates:
        filename = weight_map.get(key)
        if filename is not None:
            path = checkpoint_root / filename
            if not path.is_file():
                raise ValueError(f"checkpoint shard is missing for {key}")
            return key, path
    raise ValueError(f"checkpoint has no packed tensor among {candidates!r}")


def _read_selected_rows(
    *,
    safe_open: Any,
    torch_module: Any,
    path: Path,
    key: str,
    expected_shape: tuple[int, ...],
    plan: Any,
    projection: str,
) -> tuple[Any, tuple[dict[str, Any], ...], str]:
    reads: list[dict[str, Any]] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        source = handle.get_slice(key)
        shape = tuple(source.get_shape())
        dtype = str(source.get_dtype())
        if shape != expected_shape:
            raise ValueError(f"{projection} checkpoint shape differs from frozen Qwen")
        if dtype.upper() != "BF16":
            raise ValueError(f"{projection} checkpoint dtype is not BF16")
        selected = torch_module.empty(
            (32, *expected_shape[1:]),
            dtype=torch_module.bfloat16,
            device="cpu",
        )
        for physical_slot, logical_id in planned_row_reads(plan):
            row = source[logical_id]
            if tuple(row.shape) != expected_shape[1:]:
                raise ValueError("safetensors row slice materialized an unexpected shape")
            if str(row.dtype).lower() not in ("torch.bfloat16", "bfloat16", "bf16"):
                raise ValueError("safetensors row slice dtype is not BF16")
            selected[physical_slot].copy_(row)
            reads.append(
                {
                    "projection": projection,
                    "logical_expert_id": logical_id,
                    "physical_slot": physical_slot,
                    "materialized_leading_experts": 1,
                    "row_shape": list(row.shape),
                }
            )
            del row
    return selected, tuple(reads), dtype


def create_q32_loader_binding(context: Any) -> Q32LoaderBinding:
    if getattr(context, "round16_api_identity", None) != Q32_V2_CONTEXT_API:
        raise TypeError("V2 live factory received an unverified context contract")
    if not all(
        hasattr(context, name)
        for name in (
            "source_token",
            "snapshot",
            "layer_id",
            "checkpoint_root",
            "v1_receipt_sha256",
        )
    ):
        raise TypeError("V2 live factory context fields are incomplete")

    import torch
    import torch_npu
    from safetensors import safe_open
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ops.fused_moe.fused_moe import (
        AscendUnquantizedFusedMoEMethod,
    )

    # Exact Ascend ModelRunner enables this module-level gate before model
    # construction.  Without it, torch_npu silently downgrades a requested
    # NZ cast to base format even when weight_nz_mode=2.
    torch.npu.config.allow_internal_format = True

    raw_binding = _create_raw_q32_binding(context)
    owner = raw_binding.owner
    plan = plan_q32(context.snapshot.resident_logical_ids)
    gate_selected = None
    down_selected = None
    try:
        checkpoint_root = Path(context.checkpoint_root)
        index_path = checkpoint_root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("checkpoint index has no weight_map")
        gate_key, gate_path = _resolve_checkpoint_tensor(
            checkpoint_root,
            weight_map,
            _checkpoint_key(context.layer_id, "gate_up_proj"),
        )
        down_key, down_path = _resolve_checkpoint_tensor(
            checkpoint_root,
            weight_map,
            _checkpoint_key(context.layer_id, "down_proj"),
        )

        gate_selected, gate_reads, gate_dtype = _read_selected_rows(
            safe_open=safe_open,
            torch_module=torch,
            path=gate_path,
            key=gate_key,
            expected_shape=PACKED_GATE_UP_SHAPE,
            plan=plan,
            projection="gate_up",
        )
        down_selected, down_reads, down_dtype = _read_selected_rows(
            safe_open=safe_open,
            torch_module=torch,
            path=down_path,
            key=down_key,
            expected_shape=PACKED_DOWN_SHAPE,
            plan=plan,
            projection="down",
        )
        validate_packed_source_metadata(
            PACKED_GATE_UP_SHAPE,
            PACKED_DOWN_SHAPE,
        )
        selected_cpu_bytes = (
            int(gate_selected.numel()) * int(gate_selected.element_size())
            + int(down_selected.numel()) * int(down_selected.element_size())
        )

        # Match the exact Ascend BF16 lifecycle: upstream method created the
        # raw weights in V1; the execution method takes ownership before load;
        # its post-load hook runs only after all 96 resident projections load.
        # The real model runner keeps one vLLM config active throughout this
        # lifecycle.  The bounded probe must do the same because both method
        # construction (CustomOp dispatch) and post-load consult that context.
        # S0 stores BF16 routed experts in Ascend FRACTAL_NZ (format 29).
        # vLLM-Ascend's default mode 1 converts only quantized weights, so an
        # otherwise valid standalone BF16 probe would remain ND.  Restore the
        # frozen S0 mode explicitly rather than inheriting ambient environment.
        vllm_config = VllmConfig(additional_config={"weight_nz_mode": 2})
        with set_current_vllm_config(vllm_config):
            # AscendWorker.init_device performs this before ModelRunner builds
            # any MoE method.  The standalone component process must reproduce
            # that exact predecessor rather than relying on a stale singleton.
            ascend_config = init_ascend_config(vllm_config)
            if getattr(ascend_config, "weight_nz_mode", None) != 2:
                raise RuntimeError("V2 did not restore frozen BF16 NZ mode 2")
            owner.quant_method = AscendUnquantizedFusedMoEMethod(owner.moe_config)
            execution_method = type(owner.quant_method).__name__
            coverage = load_selected_packed_rows(
                owner=owner,
                plan=plan,
                selected_gate_up=gate_selected,
                selected_down=down_selected,
                gate_up_weight_name=gate_key,
                down_weight_name=down_key,
            )
            torch.npu.synchronize()
            after_load = int(torch.npu.memory_allocated())
            owner.quant_method.process_weights_after_loading(owner)
        torch.npu.synchronize()
        after_postload = int(torch.npu.memory_allocated())
        peak = int(torch.npu.max_memory_allocated())
        w13_format = int(torch_npu.get_npu_format(owner.w13_weight))
        w2_format = int(torch_npu.get_npu_format(owner.w2_weight))

        # A full Q32 D2H comparison proves every loaded element and the
        # transpose/NZ logical view.  It remains bounded to the selected 300MiB
        # and never creates a logical-512 NPU tensor.
        post_w13 = owner.w13_weight.detach().cpu()
        post_w2 = owner.w2_weight.detach().cpu()
        torch.npu.synchronize()
        exact_checks = {
            "gate_full_q32_exact": bool(
                torch.equal(
                    post_w13[:, :, :640],
                    gate_selected[:, :640, :].transpose(1, 2),
                )
            ),
            "up_full_q32_exact": bool(
                torch.equal(
                    post_w13[:, :, 640:],
                    gate_selected[:, 640:, :].transpose(1, 2),
                )
            ),
            "down_full_q32_exact": bool(
                torch.equal(post_w2, down_selected.transpose(1, 2))
            ),
        }
        del post_w13, post_w2, gate_selected, down_selected
        gate_selected = down_selected = None
        gc.collect()

        files: dict[str, dict[str, Any]] = {}
        for name, path in (("index", index_path), ("gate_up", gate_path), ("down", down_path)):
            stat = path.stat()
            files[name] = {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        files["index"]["sha256"] = _sha256(index_path)

        return Q32LoaderBinding(
            owner=owner,
            torch_module=torch,
            coverage=tuple(coverage),
            construction_method=raw_binding.construction_method,
            execution_method=execution_method,
            checkpoint_keys={"gate_up": gate_key, "down": down_key},
            checkpoint_files=files,
            checkpoint_shapes={
                "gate_up": PACKED_GATE_UP_SHAPE,
                "down": PACKED_DOWN_SHAPE,
            },
            checkpoint_dtypes={"gate_up": gate_dtype, "down": down_dtype},
            source_row_reads=gate_reads + down_reads,
            selected_cpu_bytes=selected_cpu_bytes,
            load_mode="SAFETENSORS_CPU_ROW_SLICE_NATIVE_PARAMETER_LOADER",
            postload_order="UPSTREAM_CREATE_LOAD_THEN_ASCEND_POSTLOAD",
            exact_value_checks=exact_checks,
            w13_npu_format=w13_format,
            w2_npu_format=w2_format,
            worker_internal_format_assignment_issued=True,
            allocator_before_bytes=raw_binding.memory_before_bytes,
            allocator_after_load_bytes=after_load,
            allocator_after_postload_bytes=after_postload,
            allocator_peak_bytes=peak,
            cleanup=raw_binding.cleanup,
        )
    except BaseException:
        if gate_selected is not None:
            del gate_selected
        if down_selected is not None:
            del down_selected
        raw_binding.cleanup()
        raise


create_q32_loader_binding.round16_api_identity = Q32_V2_FACTORY_API


__all__ = ("create_q32_loader_binding",)
