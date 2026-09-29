"""Offline-bound adapter for the frozen vLLM-Ascend 0.26 source shape.

This module has no vLLM, torch, or device imports.  It records the exact
offline source identity and validates injected runtime objects before a model
registry is mutated.  Selective allocation is deliberately an injected,
model-scoped context: the saved donor source has no verified native hook for
changing expert allocation before ``RoutedExperts`` creates its parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from threading import Lock
from typing import Any, Callable, Sequence

from .model_plugin import TARGET_ARCHITECTURE
from .native_source_preflight import (
    SourcePreflightToken,
    validate_exact_source_preflight_token,
)
from .selective_preallocation_installer import HYBRID_RUNNER_API, SELECTIVE_INSTALLER_API
from .selective_loader import SelectiveLoadPlan, plan_q32


NATIVE_BOUND_IMPLEMENTATION = (
    "round16.b.native_vllm_adapter:"
    "BoundAscendQwen4ExpForConditionalGeneration"
)


@dataclass(frozen=True)
class NativeVllmApiIdentity:
    schema: int
    image: str
    image_id: str
    repository_digest: str
    vllm_version: str
    vllm_commit: str
    vllm_ascend_commit: str
    model_revision: str
    model_config_remote_raw_path: str
    model_config_remote_raw_sha256: str
    model_config_sha256: str
    model_config_architectures: tuple[str, ...]
    model_config_num_hidden_layers: int
    model_config_num_experts: int
    model_config_num_experts_per_tok: int
    model_config_dtype: str
    model_config_max_position_embeddings: int
    target_architecture: str
    target_implementation: str
    entrypoints: tuple[tuple[str, str], ...]
    source_archive: str
    source_archive_sha256: str
    source_root: str
    upstream_vllm_source_root: str
    upstream_vllm_source_sha256: tuple[tuple[str, str], ...]
    source_sha256: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "image": self.image,
            "image_id": self.image_id,
            "repository_digest": self.repository_digest,
            "vllm_version": self.vllm_version,
            "vllm_commit": self.vllm_commit,
            "vllm_ascend_commit": self.vllm_ascend_commit,
            "model_revision": self.model_revision,
            "model_config_remote_raw_path": self.model_config_remote_raw_path,
            "model_config_remote_raw_sha256": (
                self.model_config_remote_raw_sha256
            ),
            "model_config_sha256": self.model_config_sha256,
            "model_config_architectures": list(self.model_config_architectures),
            "model_config_num_hidden_layers": self.model_config_num_hidden_layers,
            "model_config_num_experts": self.model_config_num_experts,
            "model_config_num_experts_per_tok": (
                self.model_config_num_experts_per_tok
            ),
            "model_config_dtype": self.model_config_dtype,
            "model_config_max_position_embeddings": (
                self.model_config_max_position_embeddings
            ),
            "target_architecture": self.target_architecture,
            "target_implementation": self.target_implementation,
            "entrypoints": dict(self.entrypoints),
            "source_archive": self.source_archive,
            "source_archive_sha256": self.source_archive_sha256,
            "source_root": self.source_root,
            "upstream_vllm_source_root": self.upstream_vllm_source_root,
            "upstream_vllm_source_sha256": dict(
                self.upstream_vllm_source_sha256
            ),
            "source_sha256": dict(self.source_sha256),
        }


FROZEN_API_IDENTITY = NativeVllmApiIdentity(
    schema=1,
    image="quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler",
    image_id="sha256:7e3ab7e56a54259d3ecec91073346f9ca8b2a4c1f58e15826e3ed6ff67036f00",
    repository_digest="sha256:38e4a59458c82d2972dc08ea3aa209784cc9fb925c417a5559d422f2bbebf830",
    vllm_version="0.26.0",
    vllm_commit="568afb3a13806beb53bb2e6bd518269357b237c0",
    vllm_ascend_commit="550ab66e7feb89f6e2ba49394f6ffb153189ac2b",
    model_revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
    model_config_remote_raw_path=(
        "/home/admin/Documents/ChatGPT/KT/kt-round16-integration/artifacts/"
        "round16-b/QWEN_MODEL_CONFIG_REMOTE_RAW.txt"
    ),
    model_config_remote_raw_sha256=(
        "ed9343b42a0e9d7c1bc377ba953d470da079858a8e3abbf3b306deaad4a4cdf9"
    ),
    model_config_sha256=(
        "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
    ),
    model_config_architectures=(TARGET_ARCHITECTURE,),
    model_config_num_hidden_layers=48,
    model_config_num_experts=512,
    model_config_num_experts_per_tok=10,
    model_config_dtype="bfloat16",
    model_config_max_position_embeddings=262144,
    target_architecture=TARGET_ARCHITECTURE,
    target_implementation=(
        "vllm_ascend.models.qwen4_exp:"
        "AscendQwen4ExpForConditionalGeneration"
    ),
    entrypoints=(
        ("registry", "vllm.ModelRegistry.register_model"),
        (
            "model_class",
            "vllm_ascend.models.qwen4_exp.model:"
            "AscendQwen4ExpForConditionalGeneration",
        ),
        ("worker_class", "vllm_ascend.worker.worker:NPUWorker"),
        (
            "model_runner_class",
            "vllm_ascend.worker.model_runner_v1:NPUModelRunner",
        ),
        ("model_loader", "vllm.model_executor.model_loader:get_model"),
        (
            "weight_loader",
            "Qwen4ExpForConditionalGeneration.load_weights -> "
            "AutoWeightsLoader.load_weights",
        ),
        (
            "moe_method_class",
            "vllm_ascend.ops.fused_moe.fused_moe:"
            "AscendUnquantizedFusedMoEMethod",
        ),
        ("moe_apply", "AscendUnquantizedFusedMoEMethod.apply"),
    ),
    source_archive=(
        "/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round15/"
        "upstream-image-source/round15-offline-vllm-source-20260928.tar.gz"
    ),
    source_archive_sha256=(
        "005d731a31cc11854878efbd2c2a68d1840c72afba83b048422678942773eaef"
    ),
    source_root="round15-offline-vllm-source-20260928/vllm_ascend",
    upstream_vllm_source_root=(
        "/home/admin/Documents/ChatGPT/KT/kt-round16-b-vllm/artifacts/"
        "round16-b/upstream-vllm-0.26"
    ),
    upstream_vllm_source_sha256=(
        (
            "registry.py",
            "27f6675a032ba82780c4b6038c05e6aad3988879b44f3582f6d5a273847da2f0",
        ),
        (
            "layer.py",
            "c888133b47507d6dc0a8e74671ef6e6b2c60c53e615e27098e354ad5b961d725",
        ),
        (
            "qwen3_next.py",
            "b642c10eb68978ca0df25f92ea866add08b31e654d99e78eaac0195d8bc6c74b",
        ),
        (
            "model_utils.py",
            "f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2",
        ),
        (
            "moe_runner.py",
            "3c000e17f726c9995a9a950db952415a4b2198d0778e3cedda96355421edfdae",
        ),
        (
            "routed_experts.py",
            "aa8dd75adc7c2e69f1b1c999aaac9acd4db4a206afa45081e1ffd9c4c4fdb76c",
        ),
        (
            "expert_map_manager.py",
            "308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b",
        ),
        (
            "unquantized_fused_moe_method.py",
            "76f66e7e37680d7bea541159528d3ca302be73b88beb0f5bf4847deae786dab1",
        ),
        (
            "weight_utils.py",
            "0f7dd5854cb850ee1aede33e11b22f37ea9c53f60d3477c5115efd20d69417d9",
        ),
    ),
    source_sha256=(
        (
            "models/__init__.py",
            "820c9fe4f3a107699f02e5d6e95fbd8312a056dcbdea376a324eb0d31879b9bd",
        ),
        (
            "models/qwen4_exp/__init__.py",
            "878eb7ee4b5aec2e766ace3a08ff9d80a023aad2cb32e953ae44b0827b84fa92",
        ),
        (
            "models/qwen4_exp/model.py",
            "b06a3123073234efb39958c502b53bee23384d1a7eb61f40d12b3d8b55481133",
        ),
        (
            "models/qwen4_exp/nvidia/model.py",
            "a38101a1c4a3cb078b0169d3c4b7b6c4b39ba41db8283a29cf8fcc5eb7ac9689",
        ),
        (
            "platform.py",
            "c711042261d5283be5aa644c3a5a00fcb34f70c0156f4b866700f38a927a1276",
        ),
        (
            "worker/worker.py",
            "eb0bb11de6dffecbed135979655cf032cccef3f3ea4fe3b93254933d47e0f8c0",
        ),
        (
            "worker/model_runner_v1.py",
            "de9f108bb1365c7eb5967c0cd098a005510fe627f7a27e0732d120a0f33e2608",
        ),
        (
            "ops/fused_moe/fused_moe.py",
            "2d03cf6963f4e7edd529751383c779c407560011760c86478cf0598661d47f7c",
        ),
        (
            "patch/platform/patch_fused_moe.py",
            "007875c5e3c188e33714cca007bd3e365b8ceb0d5fca8415866c673861267eaf",
        ),
        (
            "patch/worker/patch_process_weights_after_loading.py",
            "85c1fbf7e7ebaa52901e2173728be5dfacdcf796cebeb92f6d0ad9dd930c1d09",
        ),
        (
            "ops/fused_moe/moe_comm_method.py",
            "043f3c28518254347589f7af03793810262d0b29a9a82ca17dfd5fe5f9559f2e",
        ),
        (
            "ops/fused_moe/prepare_finalize.py",
            "c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd",
        ),
        (
            "quantization/quant_type.py",
            "2faa4432100452f762aac28b17cb6167d8ed62955063abb9c4f56647fb0f8257",
        ),
        (
            "ops/fused_moe/token_dispatcher.py",
            "36fa3658532b330f701c441bab279bed320fea067a2ddf75ee4cfa6a703e0d9b",
        ),
        (
            "ops/fused_moe/moe_runtime_args.py",
            "c1d52390f5619ce184207038d131abbe2d6745f527badc82363b0394f80371c2",
        ),
        (
            "ascend_forward_context.py",
            "af4a6e79ffcf6803c2a2a41dd598618398f090ea3a97253642c340ea26295c9f",
        ),
    ),
)


@dataclass(frozen=True)
class NativeEntrypoints:
    """Objects resolved by the real worker after its own version preflight."""

    model_class: type
    worker_class: type
    model_runner_class: type
    model_loader: Callable[..., Any]
    moe_method_class: type


@dataclass(frozen=True)
class NativePluginReceipt:
    status: str
    registered: bool
    architecture: str = TARGET_ARCHITECTURE
    implementation: str = NATIVE_BOUND_IMPLEMENTATION
    plan: SelectiveLoadPlan | None = None
    detail: str = ""


@dataclass(frozen=True)
class RuntimeContextCapacity:
    """Launch-time state contract for 128 input plus 128 generated tokens."""

    max_input_tokens: int
    max_output_tokens: int
    max_model_len: int
    state_capacity: int


def _runtime_context_is_sufficient(value: object) -> bool:
    if not isinstance(value, RuntimeContextCapacity):
        return False
    dimensions = (
        value.max_input_tokens,
        value.max_output_tokens,
        value.max_model_len,
        value.state_capacity,
    )
    if any(type(dimension) is not int for dimension in dimensions):
        return False
    return (
        value.max_input_tokens == 128
        and value.max_output_tokens == 128
        and value.max_model_len >= 256
        and value.state_capacity >= 256
        and value.max_model_len
        >= value.max_input_tokens + value.max_output_tokens
        and value.state_capacity
        >= value.max_input_tokens + value.max_output_tokens
    )


def _object_path(value: Any) -> str:
    return f"{getattr(value, '__module__', '')}:{getattr(value, '__name__', '')}"


def _parameters(value: Any) -> dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(value).parameters)
    except (TypeError, ValueError) as error:
        raise TypeError(f"cannot inspect API shape for {value!r}") from error


def _require_parameters(value: Any, names: Sequence[str]) -> None:
    parameters = _parameters(value)
    missing = [name for name in names if name not in parameters]
    if missing:
        raise TypeError(f"API shape is missing parameters: {', '.join(missing)}")


def _validate_entrypoints(entrypoints: NativeEntrypoints) -> None:
    if not isinstance(entrypoints, NativeEntrypoints):
        raise TypeError("entrypoints must be NativeEntrypoints")
    expected_paths = (
        (
            entrypoints.model_class,
            "vllm_ascend.models.qwen4_exp.model:"
            "AscendQwen4ExpForConditionalGeneration",
        ),
        (entrypoints.worker_class, "vllm_ascend.worker.worker:NPUWorker"),
        (
            entrypoints.model_runner_class,
            "vllm_ascend.worker.model_runner_v1:NPUModelRunner",
        ),
        (
            entrypoints.model_loader,
            "vllm.model_executor.model_loader:get_model",
        ),
        (
            entrypoints.moe_method_class,
            "vllm_ascend.ops.fused_moe.fused_moe:"
            "AscendUnquantizedFusedMoEMethod",
        ),
    )
    for value, expected in expected_paths:
        if _object_path(value) != expected:
            raise TypeError(
                f"API object path mismatch: expected {expected}, got {_object_path(value)}"
            )

    model_init = entrypoints.model_class.__init__
    _require_parameters(model_init, ("self", "vllm_config", "prefix"))
    model_parameters = _parameters(model_init)
    if model_parameters["vllm_config"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise TypeError("native model vllm_config must remain keyword-only")
    if model_parameters["prefix"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise TypeError("native model prefix must remain keyword-only")
    if model_parameters["prefix"].default != "model":
        raise TypeError("native conditional model prefix default must be 'model'")
    if not callable(getattr(entrypoints.model_class, "load_weights", None)):
        raise TypeError("native conditional model load_weights is unavailable")

    _require_parameters(
        entrypoints.worker_class.__init__,
        (
            "self",
            "vllm_config",
            "local_rank",
            "rank",
            "distributed_init_method",
            "is_driver_worker",
            "kwargs",
        ),
    )
    for method_name in ("init_device", "load_model"):
        if not callable(getattr(entrypoints.worker_class, method_name, None)):
            raise TypeError(f"native worker {method_name} is unavailable")

    _require_parameters(
        entrypoints.model_runner_class.__init__,
        ("self", "vllm_config", "device"),
    )
    if not callable(getattr(entrypoints.model_runner_class, "load_model", None)):
        raise TypeError("native model runner load_model is unavailable")
    _require_parameters(entrypoints.model_loader, ("vllm_config",))

    moe_method = entrypoints.moe_method_class
    if not callable(getattr(moe_method, "process_weights_after_loading", None)):
        raise TypeError("native MoE process_weights_after_loading is unavailable")
    apply = getattr(moe_method, "apply", None)
    if not callable(apply):
        raise TypeError("native MoE apply is unavailable")
    _require_parameters(
        apply,
        (
            "self",
            "layer",
            "x",
            "top_k",
            "router_logits",
            "renormalize",
            "routed_scaling_factor",
            "expert_map",
            "log2phy",
        ),
    )


def _validate_installer(installer: Any) -> None:
    if getattr(installer, "api_identity", None) != SELECTIVE_INSTALLER_API:
        raise TypeError("selective allocation installer API identity mismatch")
    if not callable(getattr(installer, "activate", None)):
        raise TypeError("selective allocation installer activate() is unavailable")
    if getattr(installer, "routed_experts_binding_verified", None) is not True:
        raise RuntimeError(
            "CHECKPOINT_SLICE_BINDING_UNRESOLVED: the injected installer has "
            "no verified RoutedExperts Q32 constructor/weight binding"
        )
    if (
        getattr(installer, "runtime_runner_binding_verified", None) is not True
        or getattr(installer, "hybrid_runner_api_identity", None)
        != HYBRID_RUNNER_API
    ):
        raise _RuntimeRunnerBindingUnresolved(
            "RUNTIME_RUNNER_BINDING_UNRESOLVED: a physical-Q32 RoutedExperts "
            "owner must be paired with a source-verified custom Hybrid runner"
        )
    _require_parameters(
        installer.activate,
        ("plan", "architecture", "model_class"),
    )


class _RuntimeRunnerBindingUnresolved(RuntimeError):
    pass


def _build_bound_native_model_class(
    *,
    entrypoints: NativeEntrypoints,
    installer: Any,
    plan: SelectiveLoadPlan,
    hybrid_bridge_factory: Callable[[Any], Any],
) -> type:
    native_base = entrypoints.model_class

    def bound_init(self: Any, *args: Any, **kwargs: Any) -> None:
        activation = installer.activate(
            plan=plan,
            architecture=TARGET_ARCHITECTURE,
            model_class=native_base,
        )
        if not callable(getattr(activation, "__enter__", None)) or not callable(
            getattr(activation, "__exit__", None)
        ):
            raise TypeError("selective allocation activation must be a context manager")
        with activation as acknowledge:
            if not callable(acknowledge):
                raise TypeError("selective allocation activation must yield an acknowledgement")
            native_base.__init__(self, *args, **kwargs)
            if acknowledge(self) is not True:
                raise RuntimeError(
                    "native model did not acknowledge the verified Q32 allocation binding"
                )
        bridge = hybrid_bridge_factory(self)
        if not callable(bridge):
            raise TypeError("verified hybrid bridge must be callable")
        object.__setattr__(self, "hybrid_expert_bridge", bridge)

    def native_model(self: Any) -> Any:
        return self

    def run_hybrid_experts(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self.hybrid_expert_bridge(*args, **kwargs)

    return type(
        "BoundAscendQwen4ExpForConditionalGeneration",
        (native_base,),
        {
            "__init__": bound_init,
            "native_model": property(native_model),
            "run_hybrid_experts": run_hybrid_experts,
            "__module__": __name__,
        },
    )


@dataclass(frozen=True)
class _NativeRegistrationState:
    registry: Any
    model_class: type
    native_base: type
    installer: Any
    plan: SelectiveLoadPlan
    hybrid_bridge_factory: Callable[[Any], Any]


_NATIVE_REGISTRATION_LOCK = Lock()
_NATIVE_REGISTRATIONS: dict[int, _NativeRegistrationState] = {}


def register_frozen_native_plugin(
    registry: Any,
    *,
    enabled: bool = False,
    identity: NativeVllmApiIdentity = FROZEN_API_IDENTITY,
    entrypoints: NativeEntrypoints | None = None,
    installer: Any | None = None,
    resident_logical_ids: Sequence[int] | None = None,
    hybrid_bridge_factory: Callable[[Any], Any] | None = None,
    source_preflight_token: SourcePreflightToken | None = None,
    runtime_context_capacity: RuntimeContextCapacity | None = None,
) -> NativePluginReceipt:
    """Bind the frozen source shape only after every model-scoped gate passes.

    ``enabled`` defaults to false.  A caller must inject objects from the
    frozen image and an allocation installer that is already proven to take
    effect before native model construction.  The adapter never imports or
    constructs a worker, model runner, KV cache, or device runtime.
    """

    if type(enabled) is not bool:
        return NativePluginReceipt("INVALID_ENABLE_FLAG", False)
    if not enabled:
        return NativePluginReceipt("DISABLED_DEFAULT_OFF", False)
    if identity != FROZEN_API_IDENTITY:
        return NativePluginReceipt(
            "FROZEN_API_IDENTITY_MISMATCH",
            False,
            detail="injected image/version/source identity is not the frozen 0.26 identity",
        )
    if entrypoints is None:
        return NativePluginReceipt(
            "VERSION_API_UNRESOLVED", False, detail="native entrypoints were not injected"
        )
    try:
        _validate_entrypoints(entrypoints)
    except (TypeError, ValueError) as error:
        return NativePluginReceipt("VERSION_API_SHAPE_MISMATCH", False, detail=str(error))
    if installer is None:
        return NativePluginReceipt(
            "CHECKPOINT_SLICE_BINDING_UNRESOLVED",
            False,
            detail=(
                "the model-scoped preallocation seam is source-resolved, but a "
                "verified device RoutedExperts Q32 constructor/weight binding "
                "was not injected"
            ),
        )
    try:
        _validate_installer(installer)
    except _RuntimeRunnerBindingUnresolved as error:
        return NativePluginReceipt(
            "RUNTIME_RUNNER_BINDING_UNRESOLVED", False, detail=str(error)
        )
    except RuntimeError as error:
        return NativePluginReceipt(
            "CHECKPOINT_SLICE_BINDING_UNRESOLVED", False, detail=str(error)
        )
    except (TypeError, ValueError) as error:
        return NativePluginReceipt("VERSION_API_SHAPE_MISMATCH", False, detail=str(error))
    if not callable(hybrid_bridge_factory):
        return NativePluginReceipt(
            "VERSION_API_SHAPE_MISMATCH",
            False,
            detail="hybrid bridge factory must be callable",
        )
    if resident_logical_ids is None:
        return NativePluginReceipt(
            "INVALID_Q32_PLAN", False, detail="resident logical ids are required"
        )
    try:
        plan = plan_q32(resident_logical_ids)
    except (TypeError, ValueError) as error:
        return NativePluginReceipt("INVALID_Q32_PLAN", False, detail=str(error))
    if source_preflight_token is None:
        return NativePluginReceipt(
            "SOURCE_PREFLIGHT_REQUIRED",
            False,
            plan=plan,
            detail="registry mutation requires observed exact-source digests/API checks",
        )
    if not validate_exact_source_preflight_token(source_preflight_token):
        return NativePluginReceipt(
            "SOURCE_PREFLIGHT_REJECTED",
            False,
            plan=plan,
            detail="preflight token does not bind the frozen source/config digests",
        )
    if not _runtime_context_is_sufficient(runtime_context_capacity):
        return NativePluginReceipt(
            "CONTEXT_CAPACITY_DEVICE_MODEL_PENDING",
            False,
            plan=plan,
            detail=(
                "launch must prove 128 input + 128 output with max_model_len "
                "and state capacity both at least 256"
            ),
        )

    register_model = getattr(registry, "register_model", None)
    if not callable(register_model):
        return NativePluginReceipt(
            "REGISTRY_API_REJECTED",
            False,
            plan=plan,
            detail="registry does not expose register_model",
        )
    bound_model_class = _build_bound_native_model_class(
        entrypoints=entrypoints,
        installer=installer,
        plan=plan,
        hybrid_bridge_factory=hybrid_bridge_factory,
    )
    key = id(registry)
    with _NATIVE_REGISTRATION_LOCK:
        existing = _NATIVE_REGISTRATIONS.get(key)
        if existing is not None:
            if existing.registry is not registry:
                return NativePluginReceipt(
                    "REGISTRY_API_REJECTED",
                    False,
                    plan=plan,
                    detail="registry identity collision",
                )
            if (
                existing.native_base is entrypoints.model_class
                and existing.installer is installer
                and existing.plan == plan
                and existing.hybrid_bridge_factory is hybrid_bridge_factory
            ):
                return NativePluginReceipt("ALREADY_REGISTERED", False, plan=plan)
            return NativePluginReceipt(
                "REGISTRY_ALREADY_BOUND_DIFFERENT_NATIVE_ADAPTER",
                False,
                plan=plan,
            )
        try:
            register_model(TARGET_ARCHITECTURE, bound_model_class)
        except (TypeError, ValueError, RuntimeError) as error:
            return NativePluginReceipt(
                "REGISTRY_API_REJECTED", False, plan=plan, detail=str(error)
            )
        _NATIVE_REGISTRATIONS[key] = _NativeRegistrationState(
            registry=registry,
            model_class=bound_model_class,
            native_base=entrypoints.model_class,
            installer=installer,
            plan=plan,
            hybrid_bridge_factory=hybrid_bridge_factory,
        )
    return NativePluginReceipt("REGISTERED_SOURCE_BOUND_OFFLINE", True, plan=plan)


__all__ = (
    "FROZEN_API_IDENTITY",
    "HYBRID_RUNNER_API",
    "NATIVE_BOUND_IMPLEMENTATION",
    "NativeEntrypoints",
    "NativePluginReceipt",
    "NativeVllmApiIdentity",
    "RuntimeContextCapacity",
    "SELECTIVE_INSTALLER_API",
    "register_frozen_native_plugin",
)
