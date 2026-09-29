"""Model-construction-scoped Q32 hook for the frozen Qwen FusedMoE binding.

The installer performs no device work.  It temporarily replaces only the
``qwen3_next`` module's already-patched ``FusedMoE`` global while one target
model is being constructed.  The original binding is restored on every exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock, get_ident
from types import MappingProxyType
from typing import Any, Iterator

from .model_plugin import TARGET_ARCHITECTURE
from .selective_loader import SelectiveLoadPlan


SELECTIVE_INSTALLER_API = "round16-b:model-scoped-selective-preallocation:v1"
HYBRID_RUNNER_API = "round16-b:matched-hybrid-runner:v1"
ASCEND_FUSED_MOE_FORWARDING_API = (
    "vllm-ascend@550ab66:_ascend_FusedMoE:routed-experts-forwarding:v1"
)
ROUTED_EXPERTS_Q32_API = "round16-b:routed-experts-q32-constructor:v1"
SOURCE_BOUND_HYBRID_RUNNER_API = (
    "round16-b:source-bound-hybrid-runner:offline-contract-v1"
)
QWEN_MODULE_NAME = "vllm.model_executor.models.qwen3_next"

_CONSTRUCTION_LOCK = RLock()


class ModelScopedSelectivePreallocationInstaller:
    """Temporarily inject a verified Q32 RoutedExperts owner into Qwen MoE."""

    api_identity = SELECTIVE_INSTALLER_API

    def __init__(
        self,
        *,
        qwen_module: Any,
        ascend_fused_moe: Any,
        ascend_api_identity: str,
        routed_experts_cls: type,
        routed_experts_api_identity: str,
        hybrid_runner_cls: type | None = None,
        hybrid_runner_api_identity: str | None = None,
        hybrid_runner_args: Any = None,
        expected_layer_count: int,
    ) -> None:
        if getattr(qwen_module, "__name__", None) != QWEN_MODULE_NAME:
            raise ValueError("qwen module identity does not match frozen qwen3_next")
        if not callable(ascend_fused_moe) or getattr(
            ascend_fused_moe, "__name__", None
        ) != "_ascend_FusedMoE":
            raise ValueError("frozen donor _ascend_FusedMoE was not supplied")
        if ascend_api_identity != ASCEND_FUSED_MOE_FORWARDING_API:
            raise ValueError("Ascend FusedMoE forwarding API identity mismatch")
        if (
            not isinstance(routed_experts_cls, type)
            or routed_experts_api_identity != ROUTED_EXPERTS_Q32_API
            or getattr(routed_experts_cls, "round16_q32_api_identity", None)
            != ROUTED_EXPERTS_Q32_API
        ):
            raise ValueError(
                "CHECKPOINT_SLICE_BINDING_UNRESOLVED: verified device "
                "RoutedExperts Q32 constructor is unavailable"
            )
        if type(expected_layer_count) is not int or expected_layer_count <= 0:
            raise ValueError("expected MoE layer count must be a positive integer")
        pair_values = (
            hybrid_runner_cls,
            hybrid_runner_api_identity,
            hybrid_runner_args,
        )
        pair_supplied = any(value is not None for value in pair_values)
        if pair_supplied:
            if (
                not isinstance(hybrid_runner_cls, type)
                or hybrid_runner_api_identity != SOURCE_BOUND_HYBRID_RUNNER_API
                or getattr(
                    hybrid_runner_cls,
                    "round16_source_bound_runner_api_identity",
                    None,
                )
                != SOURCE_BOUND_HYBRID_RUNNER_API
            ):
                raise ValueError(
                    "verified source-bound Hybrid runner class is unavailable"
                )
            if not isinstance(hybrid_runner_args, dict):
                from collections.abc import Mapping

                if not isinstance(hybrid_runner_args, Mapping):
                    raise TypeError("hybrid_runner_args must be a mapping")
            copied_runner_args = dict(hybrid_runner_args)
            if "round16_selective_plan" in copied_runner_args:
                raise ValueError(
                    "hybrid runner base args cannot override the scoped Q32 plan"
                )
            if any(type(name) is not str or not name for name in copied_runner_args):
                raise ValueError("hybrid runner argument names must be strings")
            frozen_runner_args: MappingProxyType | None = MappingProxyType(
                copied_runner_args
            )
        else:
            frozen_runner_args = None
        self._qwen_module = qwen_module
        self._ascend_fused_moe = ascend_fused_moe
        self._routed_experts_cls = routed_experts_cls
        self._expected_layer_count = expected_layer_count
        self._hybrid_runner_cls = hybrid_runner_cls
        self._hybrid_runner_args = frozen_runner_args
        self.routed_experts_binding_verified = True
        self.source_bound_runner_pairing_verified = pair_supplied
        # The frozen factory passes its original 512-local moe_config and
        # ExpertMapManager to the default AscendMoERunner after RoutedExperts
        # construction.  This scoped hook alone cannot safely claim a Q32
        # runtime.  Native registration must require a separately verified,
        # paired Hybrid runner seam.
        self.runtime_runner_binding_verified = False
        self.hybrid_runner_api_identity = (
            SOURCE_BOUND_HYBRID_RUNNER_API if pair_supplied else None
        )

    @staticmethod
    def _q32_args(plan: SelectiveLoadPlan) -> MappingProxyType:
        return MappingProxyType(
            {
                "round16_selective_plan": plan,
                "round16_resident_logical_ids": plan.resident_logical_ids,
                "round16_logical_to_physical": plan.logical_to_physical,
                "round16_logical_expert_count": plan.logical_expert_count,
                "round16_physical_expert_count": plan.physical_expert_count,
            }
        )

    @staticmethod
    def _validate_consumed_layer(
        routed_experts: Any,
        plan: SelectiveLoadPlan,
    ) -> None:
        if getattr(routed_experts, "round16_selective_plan", None) is not plan:
            raise RuntimeError("MoE layer did not consume the identical Q32 plan")
        if (
            getattr(routed_experts, "round16_resident_logical_ids", None)
            != plan.resident_logical_ids
        ):
            raise RuntimeError("MoE layer resident expert identity mismatch")
        if (
            getattr(routed_experts, "round16_logical_to_physical", None)
            != plan.logical_to_physical
        ):
            raise RuntimeError("MoE layer logical-to-physical mapping mismatch")
        if (
            getattr(routed_experts, "round16_logical_expert_count", None)
            != plan.logical_expert_count
            or getattr(routed_experts, "round16_physical_expert_count", None)
            != plan.physical_expert_count
        ):
            raise RuntimeError("MoE layer logical/physical expert count mismatch")
        expert_map_manager = getattr(routed_experts, "expert_map_manager", None)
        mapper = getattr(expert_map_manager, "map_global_to_local", None)
        if not callable(mapper):
            raise RuntimeError("MoE layer has no live ExpertMapManager mapper")
        for logical_id, expected_physical_id in enumerate(
            plan.logical_to_physical
        ):
            observed_physical_id = mapper(logical_id)
            if (
                type(observed_physical_id) is not int
                or observed_physical_id != expected_physical_id
            ):
                raise RuntimeError(
                    "MoE layer live ExpertMapManager mapper differs from the "
                    f"Q32 plan at logical expert {logical_id}"
                )
        weights = (
            ("w13_weight", getattr(routed_experts, "w13_weight", None)),
            ("w2_weight", getattr(routed_experts, "w2_weight", None)),
        )
        if weights[0][1] is weights[1][1]:
            raise RuntimeError("MoE layer w13/w2 allocations must be distinct")
        for name, weight in weights:
            try:
                shape = tuple(weight.shape)
            except (AttributeError, TypeError) as error:
                raise RuntimeError(f"MoE layer {name} shape is unavailable") from error
            if (
                not shape
                or any(type(dimension) is not int for dimension in shape)
                or shape[0] != plan.physical_expert_count
            ):
                raise RuntimeError(
                    f"MoE layer {name} physical expert dimension must be "
                    f"exactly {plan.physical_expert_count}"
                )
            dtype = str(getattr(weight, "dtype", "")).lower()
            if dtype not in ("bf16", "bfloat16", "torch.bfloat16"):
                raise RuntimeError(f"MoE layer {name} must use bfloat16 dtype")

    @staticmethod
    def _validate_consumed_runner(
        runner: Any,
        routed_experts: Any,
        plan: SelectiveLoadPlan,
    ) -> None:
        if getattr(runner, "round16_selective_plan", None) is not plan:
            raise RuntimeError("Hybrid runner did not consume the identical Q32 plan")
        if getattr(runner, "routed_experts", None) is not routed_experts:
            raise RuntimeError("Hybrid runner does not retain the Q32 routed owner")
        if getattr(runner, "round16_routed_owner", None) is not routed_experts:
            raise RuntimeError("Hybrid runner/owner identity binding differs")
        if getattr(runner, "round16_runner_owner_binding_verified", None) is not True:
            raise RuntimeError("Hybrid runner/owner binding was not acknowledged")

    @contextmanager
    def activate(
        self,
        *,
        plan: SelectiveLoadPlan,
        architecture: str,
        model_class: type,
    ) -> Iterator[Any]:
        del model_class
        if not isinstance(plan, SelectiveLoadPlan):
            raise TypeError("selective preallocation requires a SelectiveLoadPlan")
        if architecture != TARGET_ARCHITECTURE:
            raise ValueError("selective preallocation is limited to the target model")

        owner_thread = get_ident()
        consumed_layers: list[tuple[str, Any, Any]] = []
        frozen_args = self._q32_args(plan)
        frozen_runner_args = (
            MappingProxyType(
                {
                    **dict(self._hybrid_runner_args),
                    "round16_selective_plan": plan,
                }
            )
            if self._hybrid_runner_args is not None
            else None
        )

        with _CONSTRUCTION_LOCK:
            original = getattr(self._qwen_module, "FusedMoE", None)
            if original is not self._ascend_fused_moe:
                raise RuntimeError(
                    "qwen3_next.FusedMoE binding changed before target construction"
                )

            def scoped_fused_moe(*args: Any, **kwargs: Any) -> Any:
                if get_ident() != owner_thread:
                    raise RuntimeError(
                        "target Qwen FusedMoE construction escaped its owner thread"
                    )
                if kwargs.get("routed_experts_cls") is not None or kwargs.get(
                    "routed_experts_args"
                ) is not None:
                    raise RuntimeError(
                        "Qwen FusedMoE call already supplies a routed expert override"
                    )
                if self._hybrid_runner_cls is not None and (
                    "runner_cls" in kwargs or "runner_args" in kwargs
                ):
                    raise RuntimeError(
                        "Qwen FusedMoE call already supplies a runner override"
                    )
                forwarded = dict(kwargs)
                forwarded["routed_experts_cls"] = self._routed_experts_cls
                forwarded["routed_experts_args"] = frozen_args
                if self._hybrid_runner_cls is not None:
                    forwarded["runner_cls"] = self._hybrid_runner_cls
                    forwarded["runner_args"] = frozen_runner_args
                runner = self._ascend_fused_moe(*args, **forwarded)
                if self._hybrid_runner_cls is not None and not isinstance(
                    runner, self._hybrid_runner_cls
                ):
                    raise RuntimeError(
                        "native FusedMoE factory did not consume the paired Hybrid runner"
                    )
                routed_experts = getattr(runner, "routed_experts", None)
                if routed_experts is None:
                    raise RuntimeError(
                        "CHECKPOINT_SLICE_BINDING_UNRESOLVED: native runner does "
                        "not expose the constructed RoutedExperts owner"
                    )
                self._validate_consumed_layer(routed_experts, plan)
                if self._hybrid_runner_cls is not None:
                    self._validate_consumed_runner(runner, routed_experts, plan)
                prefix = forwarded.get("prefix", "")
                consumed_layers.append((prefix, runner, routed_experts))
                return runner

            def acknowledge(native_model: Any) -> bool:
                if native_model is None:
                    raise RuntimeError("native target model construction did not complete")
                if len(consumed_layers) != self._expected_layer_count:
                    raise RuntimeError(
                        "Q32 construction expected "
                        f"{self._expected_layer_count} MoE layers but observed "
                        f"{len(consumed_layers)}"
                    )
                prefixes = tuple(prefix for prefix, _runner, _layer in consumed_layers)
                if len(set(prefixes)) != len(prefixes):
                    raise RuntimeError("Q32 construction observed duplicate MoE layer prefixes")
                for _prefix, runner, routed_experts in consumed_layers:
                    self._validate_consumed_layer(routed_experts, plan)
                    checkpoint_validator = getattr(
                        routed_experts,
                        "round16_validate_checkpoint_coverage",
                        None,
                    )
                    if checkpoint_validator is not None:
                        if not callable(checkpoint_validator):
                            raise RuntimeError(
                                "Q32 checkpoint coverage validator is not callable"
                            )
                        checkpoint_validator()
                    if self._hybrid_runner_cls is not None:
                        self._validate_consumed_runner(
                            runner, routed_experts, plan
                        )
                return True

            self._qwen_module.FusedMoE = scoped_fused_moe
            try:
                yield acknowledge
            finally:
                self._qwen_module.FusedMoE = original


__all__ = (
    "ASCEND_FUSED_MOE_FORWARDING_API",
    "HYBRID_RUNNER_API",
    "ModelScopedSelectivePreallocationInstaller",
    "QWEN_MODULE_NAME",
    "ROUTED_EXPERTS_Q32_API",
    "SOURCE_BOUND_HYBRID_RUNNER_API",
    "SELECTIVE_INSTALLER_API",
)
