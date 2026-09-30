"""Bounded layer0 Ascend probe for one Physical-Q32 fixed-slot rewrite.

The device path reuses the frozen Round18 native-partial owner/loader.  It
never constructs a full model or more than one layer.  The frozen owner is
currently expected to be post-load Ascend NZ29; that format is inspected on
the real tensor and rejected before mutation because no format-specific
in-place writer has been qualified.  A future mutable ND/BF16 owner can take
the complete replace/verify/restore path already defined here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from round16.b.q32_hybrid_contract import validate_v2_receipt
from round16_integration.b_q32_loader_live_factory import create_q32_loader_binding
from round16_integration.b_q32_loader_probe import (
    ASCEND_NZ_FORMAT,
    Q32V2FactoryContext,
)
from round16_integration.p1_allgather_q32_probe import (
    load_snapshot_manifest,
    load_source_preflight_token,
)
from round18.fixed_slot_write import (
    FixedSlotBacking,
    FixedSlotWriter,
    ProjectionSource,
    SlotNotPublishedError,
    SourceExpert,
)
from round18.native_partial_contract import LOGICAL_EXPERTS, NativePartialCase
from round18.native_runner_seam import execution_map_from_owner
from round18_integration.native_partial_probe import (
    MAX_RELATIVE_L2,
    _enter_bounded_forward_context,
    _load_bf16_reference,
    _run_case,
    _write_json,
)


R18_NATIVE_FIXED_SLOT_PROBE_API = "round18:native-fixed-slot-write-device-probe:v1"
ASCEND_ND_FORMAT = 2
_BF16_DTYPES = frozenset(("bf16", "bfloat16", "torch.bfloat16"))
_EXPECTED_W13_SHAPE = (32, 2560, 1280)
_EXPECTED_W2_SHAPE = (32, 640, 2560)


class InjectedMidWriteFailure(RuntimeError):
    """A deliberate pre-copy failure used to prove non-publication."""


def _tensor_inspection(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    get_npu_format: Callable[[Any], Any],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    shape = tuple(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", ""))
    device = str(getattr(getattr(value, "device", None), "type", ""))
    try:
        pointer = int(value.data_ptr())
    except (AttributeError, TypeError, ValueError):
        pointer = -1
        failures.append(f"{name} has no stable data_ptr")
    try:
        npu_format = int(get_npu_format(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} NPU format is not an integer") from error
    if shape != expected_shape:
        failures.append(f"{name} shape differs from frozen Physical-Q32")
    if dtype.lower() not in _BF16_DTYPES:
        failures.append(f"{name} dtype is not BF16")
    if device.lower() != "npu":
        failures.append(f"{name} is not on NPU")
    if npu_format != ASCEND_ND_FORMAT:
        failures.append(
            f"{name} format {npu_format} is preprocessed; only mutable ND format "
            f"{ASCEND_ND_FORMAT} is qualified"
        )
    return (
        {
            "shape": list(shape),
            "dtype": dtype,
            "device_type": device,
            "npu_format": npu_format,
            "data_ptr": pointer,
        },
        failures,
    )


def inspect_postload_backing(
    owner: Any, *, get_npu_format: Callable[[Any], Any]
) -> dict[str, Any]:
    """Inspect actual post-load tensors without allocating or mutating them."""

    if not callable(get_npu_format):
        raise TypeError("get_npu_format must be callable")
    w13 = getattr(owner, "w13_weight", None)
    w2 = getattr(owner, "w2_weight", None)
    if w13 is None or w2 is None or w13 is w2:
        raise ValueError("owner must expose distinct w13_weight and w2_weight")
    w13_row, w13_failures = _tensor_inspection(
        w13,
        name="w13_weight",
        expected_shape=_EXPECTED_W13_SHAPE,
        get_npu_format=get_npu_format,
    )
    w2_row, w2_failures = _tensor_inspection(
        w2,
        name="w2_weight",
        expected_shape=_EXPECTED_W2_SHAPE,
        get_npu_format=get_npu_format,
    )
    failures = w13_failures + w2_failures
    return {
        "eligible": not failures,
        "eligible_format": ASCEND_ND_FORMAT,
        "known_preprocessed_format": ASCEND_NZ_FORMAT,
        "w13": w13_row,
        "w2": w2_row,
        "data_ptrs": {"w13": w13_row["data_ptr"], "w2": w2_row["data_ptr"]},
        "failure_reason": "; ".join(failures) if failures else None,
    }


def build_ineligible_backing_receipt(
    *,
    inspection: Mapping[str, Any],
    allocator: Mapping[str, int],
    format_wall_seconds: float,
) -> dict[str, Any]:
    if inspection.get("eligible") is not False:
        raise ValueError("ineligible receipt requires a rejected backing inspection")
    return {
        "schema": 1,
        "api": R18_NATIVE_FIXED_SLOT_PROBE_API,
        "status": "FAIL_FIXED_SLOT_POSTLOAD_FORMAT_INELIGIBLE",
        "device_executed": True,
        "mutation_started": False,
        "owner_scope": "SINGLE_LAYER_PHYSICAL_Q32",
        "initial_owner_construction": "REUSED_ROUND16_Q32_LOADER_FACTORY",
        "postload_backing": dict(inspection),
        "allocator": dict(allocator),
        "format_inspection_wall_seconds": float(format_wall_seconds),
        "copy_wall_seconds": None,
        "publish_wall_seconds": None,
        "published_generations": [],
        "data_ptr_stable": None,
        "full_parent_rebuild": False,
        "parent_tensor_assignments": 0,
        "performance_claim": None,
    }


class MeasuredSlotHooks:
    """Measure device hooks and inject one failure before a selected copy."""

    def __init__(
        self,
        *,
        copy_projection: Callable[..., Any],
        publish_barrier: Callable[..., Any],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not callable(copy_projection) or not callable(publish_barrier):
            raise TypeError("measured slot hooks require callable device hooks")
        if not callable(clock):
            raise TypeError("measured slot hook clock must be callable")
        self._copy_projection = copy_projection
        self._publish_barrier = publish_barrier
        self._clock = clock
        self._fail_once_projection: str | None = None
        self.copy_attempts = 0
        self.copy_wall_seconds = 0.0
        self.publish_wall_seconds = 0.0

    def arm_failure_once(self, projection: str = "up") -> None:
        if projection not in ("gate", "up", "down"):
            raise ValueError("failure projection must be gate, up, or down")
        if self._fail_once_projection is not None:
            raise RuntimeError("a midwrite failure is already armed")
        self._fail_once_projection = projection

    def copy_projection(self, **kwargs: Any) -> Any:
        started = self._clock()
        self.copy_attempts += 1
        try:
            if kwargs.get("projection") == self._fail_once_projection:
                self._fail_once_projection = None
                raise InjectedMidWriteFailure(
                    f"injected failure before {kwargs.get('projection')} copy"
                )
            return self._copy_projection(**kwargs)
        finally:
            self.copy_wall_seconds += self._clock() - started

    def publish_barrier(self, **kwargs: Any) -> Any:
        started = self._clock()
        try:
            return self._publish_barrier(**kwargs)
        finally:
            self.publish_wall_seconds += self._clock() - started


def _require_verification(value: Mapping[str, Any], phase: str) -> dict[str, Any]:
    result = dict(value)
    if result.get("passed") is not True:
        raise RuntimeError(f"native {phase} verification did not pass")
    return result


def run_fixed_slot_cycle(
    *,
    writer: FixedSlotWriter,
    hooks: MeasuredSlotHooks,
    replacement_source: SourceExpert,
    original_source: SourceExpert,
    verify_replacement: Callable[[], Mapping[str, Any]],
    verify_restored: Callable[[], Mapping[str, Any]],
    read_data_ptrs: Callable[[], Mapping[str, int]],
    read_allocator: Callable[[], Mapping[str, int]],
) -> dict[str, Any]:
    """Prove failure gating, replacement, native compute, and restoration."""

    if writer.copy_projection != hooks.copy_projection:
        # Bound-method identity is not stable across attribute access; compare
        # the owning instance instead.
        if getattr(writer.copy_projection, "__self__", None) is not hooks:
            raise ValueError("writer is not wired to the supplied measured hooks")
    if writer.published_generation != 0:
        raise ValueError("bounded fixed-slot cycle requires a fresh writer")
    pointers_before = dict(read_data_ptrs())

    hooks.arm_failure_once("up")
    try:
        writer.write(source=replacement_source, generation=1)
    except InjectedMidWriteFailure:
        pass
    else:
        raise RuntimeError("midwrite failure injection did not fire")
    try:
        writer.acquire(generation=1)
    except SlotNotPublishedError:
        midwrite_unpublished = True
    else:
        raise RuntimeError("partially written generation became visible")

    attempts_before_stale = hooks.copy_attempts
    try:
        writer.write(source=replacement_source, generation=2)
    except ValueError:
        skipped_rejected = hooks.copy_attempts == attempts_before_stale
    else:
        raise RuntimeError("skipped generation was accepted after partial failure")
    if not skipped_rejected:
        raise RuntimeError("skipped generation reached the copy hook")

    replacement_publish = writer.write(source=replacement_source, generation=1)
    attempts_before_stale = hooks.copy_attempts
    try:
        writer.write(source=replacement_source, generation=1)
    except ValueError:
        stale_rejected = hooks.copy_attempts == attempts_before_stale
    else:
        raise RuntimeError("stale generation was accepted after publication")
    if not stale_rejected:
        raise RuntimeError("stale generation reached the copy hook")

    replacement_error: BaseException | None = None
    replacement_traceback = None
    replacement_check: dict[str, Any] = {}
    try:
        replacement_check = _require_verification(
            verify_replacement(), "replacement"
        )
    except BaseException as error:
        replacement_error = error
        replacement_traceback = error.__traceback__

    try:
        restore_publish = writer.write(source=original_source, generation=2)
        restore_check = _require_verification(verify_restored(), "restore")
    except BaseException as restore_error:
        if replacement_error is not None:
            raise restore_error from replacement_error
        raise
    pointers_after = dict(read_data_ptrs())
    if pointers_after != pointers_before:
        raise RuntimeError("fixed-slot data_ptr changed during replace/restore")
    if replacement_error is not None:
        raise replacement_error.with_traceback(replacement_traceback)

    return {
        "schema": 1,
        "api": R18_NATIVE_FIXED_SLOT_PROBE_API,
        "status": "PASS_FIXED_SLOT_REPLACE_RESTORE",
        "device_executed": True,
        "mutation_started": True,
        "physical_slot": writer.backing.physical_slot,
        "replacement_logical_expert_id": replacement_source.logical_expert_id,
        "restored_logical_expert_id": original_source.logical_expert_id,
        "published_generations": [
            replacement_publish.generation,
            restore_publish.generation,
        ],
        "midwrite_failure_unpublished": midwrite_unpublished,
        "skipped_generation_rejected": skipped_rejected,
        "stale_generation_rejected": stale_rejected,
        "replacement_verification": replacement_check,
        "restore_verification": restore_check,
        "data_ptrs_before": pointers_before,
        "data_ptrs_after": pointers_after,
        "data_ptr_stable": True,
        "copy_attempts": hooks.copy_attempts,
        "copy_wall_seconds": hooks.copy_wall_seconds,
        "publish_wall_seconds": hooks.publish_wall_seconds,
        "allocator": dict(read_allocator()),
        "owner_scope": "SINGLE_LAYER_PHYSICAL_Q32",
        "initial_owner_construction": "REUSED_ROUND16_Q32_LOADER_FACTORY",
        "full_parent_rebuild": False,
        "parent_tensor_assignments": 0,
        "performance_claim": None,
    }


def _load_one_bf16_source(
    *, model_root: Path, layer_id: int, logical_expert_id: int
) -> SourceExpert:
    from safetensors import safe_open

    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    base = f"model.language_model.layers.{layer_id}.mlp.experts."
    gate_key = base + "gate_up_proj"
    down_key = base + "down_proj"
    with safe_open(
        str(model_root / weight_map[gate_key]), framework="pt", device="cpu"
    ) as gate_file, safe_open(
        str(model_root / weight_map[down_key]), framework="pt", device="cpu"
    ) as down_file:
        gate_up = gate_file.get_slice(gate_key)[logical_expert_id]
        down = down_file.get_slice(down_key)[logical_expert_id]
        if str(gate_up.dtype).lower() not in _BF16_DTYPES or str(
            down.dtype
        ).lower() not in _BF16_DTYPES:
            raise ValueError("fixed-slot checkpoint source is not BF16")
        gate, up = gate_up.chunk(2, dim=0)
        return SourceExpert(
            logical_expert_id=logical_expert_id,
            gate=ProjectionSource(
                "gate", (640, 2560), "bfloat16", gate.contiguous()
            ),
            up=ProjectionSource("up", (640, 2560), "bfloat16", up.contiguous()),
            down=ProjectionSource(
                "down", (2560, 640), "bfloat16", down.contiguous()
            ),
        )


def _source_from_saved_slot(
    *, owner: Any, physical_slot: int, logical_expert_id: int
) -> SourceExpert:
    w13 = owner.w13_weight[physical_slot].detach().cpu()
    w2 = owner.w2_weight[physical_slot].detach().cpu()
    return SourceExpert(
        logical_expert_id=logical_expert_id,
        gate=ProjectionSource(
            "gate", (640, 2560), "bfloat16", w13[:, :640].T.contiguous()
        ),
        up=ProjectionSource(
            "up", (640, 2560), "bfloat16", w13[:, 640:].T.contiguous()
        ),
        down=ProjectionSource(
            "down", (2560, 640), "bfloat16", w2.T.contiguous()
        ),
    )


def _verification_case(target_logical_id: int, plan: Any) -> NativePartialCase:
    nonresident = [
        expert_id
        for expert_id, slot in enumerate(plan.logical_to_physical)
        if slot == -1 and expert_id >= 32
    ][:9]
    if target_logical_id >= 32 or len(nonresident) != 9:
        raise RuntimeError(
            "bounded native reference currently requires a resident logical ID <32 "
            "and nine nonresident IDs >=32"
        )
    return NativePartialCase(
        "fixed_slot_single_resident",
        ((target_logical_id, *nonresident),),
    )


def _require_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND18_ENABLE_FIXED_SLOT_WRITE") != "1":
        raise RuntimeError("ROUND18_ENABLE_FIXED_SLOT_WRITE=1 is required")


def run_device_probe(parsed: argparse.Namespace) -> dict[str, Any]:
    """Run one real layer0 owner, failing closed on the frozen NZ29 backing."""

    source_token = load_source_preflight_token(parsed.source_token)
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    v2_receipt, v2_sha256 = validate_v2_receipt(parsed.v2_receipt)
    if tuple(v2_receipt["resident_logical_ids"]) != snapshot.resident_logical_ids:
        raise ValueError("fixed-slot V2 receipt and S0 resident maps differ")
    if parsed.layer != 0:
        raise ValueError("fixed-slot probe is restricted to layer0")
    if not (0 <= parsed.physical_slot < 32):
        raise ValueError("physical slot must be in [0,32)")
    if not (0 <= parsed.replacement_expert < LOGICAL_EXPERTS):
        raise ValueError("replacement expert must be in [0,512)")

    import torch
    import torch_npu
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.10, 0)
    torch.npu.config.allow_internal_format = True
    if not hasattr(torch.ops._C_ascend, "moe_gating_top_k"):
        raise RuntimeError("frozen Ascend moe_gating_top_k op was not registered")

    context = Q32V2FactoryContext(
        source_token=source_token,
        snapshot=snapshot,
        layer_id=parsed.layer,
        checkpoint_root=parsed.model_root,
        v1_receipt_sha256=str(v2_receipt["v1_receipt_sha256"]),
    )
    binding = create_q32_loader_binding(context)
    config_scope = None
    forward_scope = None
    prior_extra: dict[str, Any] = {}
    distributed_initialized = False
    model_parallel_initialized = False
    allocator_before = int(torch.npu.memory_allocated())
    torch.npu.reset_peak_memory_stats()
    format_started = time.perf_counter()
    try:
        owner = binding.owner
        inspection = inspect_postload_backing(
            owner, get_npu_format=torch_npu.get_npu_format
        )
        format_wall = time.perf_counter() - format_started
        if not inspection["eligible"]:
            torch.npu.synchronize()
            return {
                **build_ineligible_backing_receipt(
                    inspection=inspection,
                    allocator={
                        "before_bytes": allocator_before,
                        "after_bytes": int(torch.npu.memory_allocated()),
                        "peak_bytes": int(torch.npu.max_memory_allocated()),
                    },
                    format_wall_seconds=format_wall,
                ),
                "layer": parsed.layer,
                "physical_slot": parsed.physical_slot,
                "replacement_logical_expert_id": parsed.replacement_expert,
                "v2_receipt_sha256": v2_sha256,
            }

        plan = owner.round16_selective_plan
        original_logical_id = plan.resident_logical_ids[parsed.physical_slot]
        if parsed.replacement_expert == original_logical_id:
            raise ValueError(
                "replacement expert must differ from the original slot owner"
            )
        replacement_source = _load_one_bf16_source(
            model_root=parsed.model_root,
            layer_id=parsed.layer,
            logical_expert_id=parsed.replacement_expert,
        )
        save_started = time.perf_counter()
        torch.npu.synchronize()
        original_source = _source_from_saved_slot(
            owner=owner,
            physical_slot=parsed.physical_slot,
            logical_expert_id=original_logical_id,
        )
        torch.npu.synchronize()
        save_original_wall = time.perf_counter() - save_started
        gate_destination = owner.w13_weight[parsed.physical_slot, :, :640]
        up_destination = owner.w13_weight[parsed.physical_slot, :, 640:]
        down_destination = owner.w2_weight[parsed.physical_slot]
        backing = FixedSlotBacking(
            physical_slot=parsed.physical_slot,
            physical_slot_count=32,
            dtype=str(owner.w13_weight.dtype),
            backing_format="BF16_FIXED_Q32",
            gate_destination=gate_destination,
            up_destination=up_destination,
            down_destination=down_destination,
        )

        def copy_projection(**kwargs: Any) -> bool:
            source = kwargs["source"].T.contiguous().to("npu:0")
            with torch.no_grad():
                kwargs["destination"].copy_(source)
            del source
            return True

        def publish_barrier(**_kwargs: Any) -> bool:
            torch.npu.synchronize()
            return True

        hooks = MeasuredSlotHooks(
            copy_projection=copy_projection,
            publish_barrier=publish_barrier,
        )
        writer = FixedSlotWriter(
            backing=backing,
            copy_projection=hooks.copy_projection,
            publish_barrier=hooks.publish_barrier,
        )
        vllm_config = VllmConfig(additional_config={"weight_nz_mode": 2})
        config_scope = set_current_vllm_config(vllm_config)
        config_scope.__enter__()
        ascend_config = init_ascend_config(vllm_config)
        if getattr(ascend_config, "weight_nz_mode", None) != 2:
            raise RuntimeError("fixed-slot probe did not restore frozen weight mode")
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29928",
            backend="hccl",
        )
        distributed_initialized = True
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            prefill_context_model_parallel_size=1,
            decode_context_model_parallel_size=1,
            backend="hccl",
        )
        model_parallel_initialized = True
        forward_scope, _forward_state = _enter_bounded_forward_context()
        comm = AllGatherCommImpl(owner.moe_config)
        for name in (
            "moe_comm_method",
            "moe_comm_type",
            "flash_comm_v1_enabled",
            "in_profile_run",
        ):
            prior_extra[name] = getattr(_EXTRA_CTX, name, None)
        _EXTRA_CTX.moe_comm_method = comm
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        _EXTRA_CTX.flash_comm_v1_enabled = False
        _EXTRA_CTX.in_profile_run = False
        expert_map = execution_map_from_owner(owner, plan)
        case = _verification_case(original_logical_id, plan)
        replacement_reference = _load_bf16_reference(
            model_root=parsed.model_root,
            layer_id=parsed.layer,
            resident_ids=(parsed.replacement_expert,),
        )[parsed.replacement_expert]
        original_reference = _load_bf16_reference(
            model_root=parsed.model_root,
            layer_id=parsed.layer,
            resident_ids=(original_logical_id,),
        )[original_logical_id]

        def verify(reference: tuple[Any, Any, Any], seed: int) -> dict[str, Any]:
            row = _run_case(
                case=case,
                owner=owner,
                comm=comm,
                expert_map=expert_map,
                reference_weights={original_logical_id: reference},
                seed=seed,
                repeats=1,
            )
            return {
                "passed": row["passed"],
                "relative_l2": row["rows"][0]["relative_l2"],
                "hash": row["hash"],
                "selector_calls": row["rows"][0]["selector_calls"],
            }

        def data_ptrs() -> dict[str, int]:
            return {
                "w13": int(owner.w13_weight.data_ptr()),
                "w2": int(owner.w2_weight.data_ptr()),
                "gate": int(gate_destination.data_ptr()),
                "up": int(up_destination.data_ptr()),
                "down": int(down_destination.data_ptr()),
            }

        result = run_fixed_slot_cycle(
            writer=writer,
            hooks=hooks,
            replacement_source=replacement_source,
            original_source=original_source,
            verify_replacement=lambda: verify(replacement_reference, 18901),
            verify_restored=lambda: verify(original_reference, 18902),
            read_data_ptrs=data_ptrs,
            read_allocator=lambda: {
                "before_bytes": allocator_before,
                "after_bytes": int(torch.npu.memory_allocated()),
                "peak_bytes": int(torch.npu.max_memory_allocated()),
            },
        )
        result.update(
            {
                "layer": parsed.layer,
                "original_logical_expert_id": original_logical_id,
                "postload_backing": inspection,
                "format_inspection_wall_seconds": format_wall,
                "save_original_wall_seconds": save_original_wall,
                "relative_l2_limit": MAX_RELATIVE_L2,
                "v2_receipt_sha256": v2_sha256,
                "full_model": "NOT_RUN",
                "all_layer_bank": "NOT_RUN",
            }
        )
        return result
    finally:
        for name, value in prior_extra.items():
            setattr(_EXTRA_CTX, name, value)
        if forward_scope is not None:
            forward_scope.__exit__(None, None, None)
        if model_parallel_initialized:
            destroy_model_parallel()
        if distributed_initialized:
            destroy_distributed_environment()
        if config_scope is not None:
            config_scope.__exit__(None, None, None)
        binding.cleanup()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-device", action="store_true")
    parser.add_argument("--source-token", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--v2-receipt", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--physical-slot", type=int, default=0)
    parser.add_argument("--replacement-expert", type=int, default=32)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if not parsed.enable_device:
        raise ValueError("native fixed-slot probe requires --enable-device")
    _require_enable(os.environ)
    result = run_device_probe(parsed)
    _write_json(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "MeasuredSlotHooks",
    "R18_NATIVE_FIXED_SLOT_PROBE_API",
    "build_ineligible_backing_receipt",
    "inspect_postload_backing",
    "main",
    "run_device_probe",
    "run_fixed_slot_cycle",
)
