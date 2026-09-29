"""Explicit, model-scoped registration for the Round16-B bridge.

The module deliberately does not import vLLM, torch, or torch_npu at import
time.  A worker may inject its model registry after its own runtime preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from threading import Lock
from typing import Any, Callable

from round16.common.contracts import SupportDecision


TARGET_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
MODEL_IMPLEMENTATION = (
    "round16.b.model_plugin:KTHybridQwen4ExpForConditionalGeneration"
)


@dataclass(frozen=True)
class RegistrationReceipt:
    status: str
    architecture: str
    implementation: str
    registered: bool


@dataclass(frozen=True)
class VerifiedModelBinding:
    """Explicit proof that a worker resolved its model construction boundary."""

    api_identity: str
    native_factory: Callable[..., Any]
    hybrid_bridge_factory: Callable[[Any], Any]

    def __post_init__(self) -> None:
        if not self.api_identity.strip():
            raise ValueError("verified model binding requires a non-empty API identity")
        if not callable(self.native_factory) or not callable(self.hybrid_bridge_factory):
            raise TypeError("verified model factories must be callable")


class KTHybridQwen4ExpForConditionalGeneration:
    """Thin model-level holder whose native model is supplied by the worker.

    Round16-B does not take platform or KV-cache ownership.  The actual vLLM
    model factory remains version-specific and is therefore injected only
    after a worker preflight has resolved it.
    """

    _verified_binding: VerifiedModelBinding | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        binding = type(self)._verified_binding
        if binding is None:
            raise RuntimeError(
                "VERSION_API_UNRESOLVED: no verified native model binding"
            )
        self.native_model = binding.native_factory(*args, **kwargs)
        self.hybrid_expert_bridge = binding.hybrid_bridge_factory(self.native_model)
        if not callable(self.hybrid_expert_bridge):
            raise TypeError("verified hybrid bridge must be callable")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.native_model(*args, **kwargs)

    def run_hybrid_experts(self, *args: Any, **kwargs: Any) -> Any:
        return self.hybrid_expert_bridge(*args, **kwargs)


class ModelPluginRegistrar:
    """Process-local, thread-safe registrar with per-registry idempotency."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._registered: dict[int, tuple[Any, VerifiedModelBinding]] = {}

    @staticmethod
    def _bound_model_class(binding: VerifiedModelBinding) -> type:
        return type(
            "VerifiedKTHybridQwen4ExpForConditionalGeneration",
            (KTHybridQwen4ExpForConditionalGeneration,),
            {
                "_verified_binding": binding,
                "__module__": __name__,
            },
        )

    def register(
        self,
        registry: Any,
        *,
        architecture: str = TARGET_ARCHITECTURE,
        binding: VerifiedModelBinding | None = None,
    ) -> RegistrationReceipt:
        if architecture != TARGET_ARCHITECTURE:
            raise ValueError(
                f"Round16-B only supports target architecture {TARGET_ARCHITECTURE}"
            )
        if binding is None:
            return RegistrationReceipt(
                "VERSION_API_UNRESOLVED",
                architecture,
                MODEL_IMPLEMENTATION,
                False,
            )
        if not isinstance(binding, VerifiedModelBinding):
            raise TypeError("binding must be a VerifiedModelBinding")
        register_model = getattr(registry, "register_model", None)
        if not callable(register_model):
            raise TypeError("model registry does not expose register_model")
        key = id(registry)
        with self._lock:
            registered = self._registered.get(key)
            if registered is not None:
                registered_registry, registered_binding = registered
                if registered_registry is not registry:
                    raise RuntimeError("registry identity collision")
                if registered_binding is not binding:
                    raise RuntimeError("registry already has a different verified binding")
                return RegistrationReceipt(
                    "ALREADY_REGISTERED",
                    architecture,
                    MODEL_IMPLEMENTATION,
                    False,
                )
            register_model(architecture, self._bound_model_class(binding))
            self._registered[key] = (registry, binding)
            return RegistrationReceipt(
                "REGISTERED",
                architecture,
                MODEL_IMPLEMENTATION,
                True,
            )


def _resolve_native_registry(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> tuple[SupportDecision, Any | None]:
    try:
        vllm = import_module("vllm")
    except (ImportError, ModuleNotFoundError) as error:
        return (
            SupportDecision.reject(
                "VERSION_API_UNRESOLVED",
                f"vLLM model registry import unavailable: {error}",
            ),
            None,
        )
    registry = getattr(vllm, "ModelRegistry", None)
    if registry is None or not callable(getattr(registry, "register_model", None)):
        return (
            SupportDecision.reject(
                "VERSION_API_UNRESOLVED",
                "vLLM ModelRegistry.register_model is unavailable",
            ),
            None,
        )
    return SupportDecision.accept("vLLM ModelRegistry.register_model resolved"), registry


def preflight_vllm_registry(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> SupportDecision:
    decision, _ = _resolve_native_registry(import_module)
    return decision


_DEFAULT_REGISTRAR = ModelPluginRegistrar()


def register(
    registry: Any | None = None,
    *,
    binding: VerifiedModelBinding | None = None,
) -> RegistrationReceipt:
    """Register only the target model; return an explicit unresolved receipt."""

    if binding is None:
        return RegistrationReceipt(
            "VERSION_API_UNRESOLVED",
            TARGET_ARCHITECTURE,
            MODEL_IMPLEMENTATION,
            False,
        )
    if registry is None:
        decision, registry = _resolve_native_registry()
        if not decision.supported:
            return RegistrationReceipt(
                decision.reason_code,
                TARGET_ARCHITECTURE,
                MODEL_IMPLEMENTATION,
                False,
            )
    return _DEFAULT_REGISTRAR.register(registry, binding=binding)


__all__ = (
    "KTHybridQwen4ExpForConditionalGeneration",
    "MODEL_IMPLEMENTATION",
    "ModelPluginRegistrar",
    "RegistrationReceipt",
    "TARGET_ARCHITECTURE",
    "VerifiedModelBinding",
    "preflight_vllm_registry",
    "register",
)
