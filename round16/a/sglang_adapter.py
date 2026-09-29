"""SGLang-facing adapter with no import-time SGLang or device dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from round16.a.provider_manifest import decide_support
from round16.common.contracts import (
    Phase,
    ProviderResult,
    ProviderSpec,
    StateLease,
    SupportDecision,
    TensorSpec,
)


class MetadataContractError(ValueError):
    """Raised before execution when scheduler metadata does not own the lease."""


class UnsupportedSpecError(RuntimeError):
    """Raised when a rejected request has no configured eager fallback."""


class ProviderCloseError(RuntimeError):
    """Reports every dependency that failed during a best-effort close pass."""

    def __init__(self, errors: tuple[Exception, ...]) -> None:
        self.errors = errors
        summary = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"failed to close {len(errors)} provider dependency: {summary}")


@dataclass
class ImmediateCompletion:
    """A synchronous completion boundary used by offline and eager adapters."""

    completion_id: str
    _done: bool = True

    @property
    def done(self) -> bool:
        return self._done

    def wait(self) -> None:
        self._done = True


@dataclass(frozen=True)
class AdaptedMetadata:
    provider_spec: ProviderSpec
    lease_identity: tuple[Any, ...]
    observed_inputs: tuple[TensorSpec, ...]
    runtime_stream_kind: str
    source: Any


@dataclass(frozen=True)
class BackendResult:
    output: Any
    completion: Any
    state_writes: tuple[str, ...] = ()
    aliases_input: bool = False


_LEASE_FIELDS = (
    "request_id",
    "request_epoch",
    "allocation_generation",
    "batch_generation",
    "layer_id",
    "token_range",
    "completion_id",
)


def _read(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise MetadataContractError(f"metadata field {name!r} is required")
        return source[name]
    try:
        return getattr(source, name)
    except AttributeError as exc:
        raise MetadataContractError(f"metadata field {name!r} is required") from exc


def _freeze_tensor_spec(value: Any) -> TensorSpec:
    if isinstance(value, TensorSpec):
        fields = {
            "name": value.name,
            "shape": value.shape,
            "dtype": value.dtype,
            "layout": value.layout,
            "stride": value.stride,
            "device_type": value.device_type,
        }
    elif isinstance(value, Mapping):
        fields = dict(value)
    else:
        raise MetadataContractError("provider input spec must be TensorSpec or a mapping")
    try:
        fields["shape"] = tuple(fields["shape"])
        if fields.get("stride") is not None:
            fields["stride"] = tuple(fields["stride"])
        return TensorSpec(**fields)
    except (KeyError, TypeError, ValueError) as exc:
        raise MetadataContractError(f"invalid provider input spec: {exc}") from exc


def _freeze_spec(value: Any) -> ProviderSpec:
    if isinstance(value, ProviderSpec):
        fields = {
            "operation": value.operation,
            "phase": value.phase,
            "inputs": value.inputs,
            "state_format": value.state_format,
            "stream_kind": value.stream_kind,
            "metadata_version": value.metadata_version,
        }
    elif isinstance(value, Mapping):
        fields = value
    else:
        raise MetadataContractError("provider_spec must be ProviderSpec or a mapping")
    try:
        inputs = tuple(_freeze_tensor_spec(item) for item in fields["inputs"])
        phase = fields["phase"]
        return ProviderSpec(
            operation=fields["operation"],
            phase=phase if isinstance(phase, Phase) else Phase(phase),
            inputs=inputs,
            state_format=fields["state_format"],
            stream_kind=fields["stream_kind"],
            metadata_version=fields["metadata_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MetadataContractError(f"invalid provider_spec: {exc}") from exc


def _input_values(inputs: Any) -> tuple[Any, ...]:
    if isinstance(inputs, (tuple, list)):
        return tuple(inputs)
    return (inputs,)


def _input_identity_objects(inputs: Any) -> tuple[Any, ...]:
    """Return the runtime container and every top-level tensor it owns."""
    if isinstance(inputs, Mapping):
        return (inputs, *tuple(inputs.values()))
    if isinstance(inputs, (tuple, list)):
        return (inputs, *tuple(inputs))
    return (inputs,)


def _aliases_runtime_input(
    output: Any,
    input_identities: tuple[Any, ...],
) -> bool:
    """Check a result against the input identities frozen before execution."""
    return any(output is candidate for candidate in input_identities)


def _runtime_field(value: Any, field: str, index: int) -> Any:
    try:
        observed = getattr(value, field)
    except AttributeError as exc:
        if field == "device_type":
            try:
                observed = getattr(value.device, "type")
            except AttributeError:
                pass
            else:
                return observed
        raise MetadataContractError(
            f"runtime input[{index}] field {field!r} is required"
        ) from exc
    return observed() if callable(observed) else observed


def _symbol(value: Any) -> str:
    text = str(value)
    return text.removeprefix("torch.")


def _exact_int_tuple(value: Any, field: str, index: int) -> tuple[int, ...]:
    try:
        frozen = tuple(value)
    except TypeError as exc:
        raise MetadataContractError(
            f"runtime input[{index}] {field} must be an iterable of exact ints"
        ) from exc
    for position, dimension in enumerate(frozen):
        if type(dimension) is not int:
            raise MetadataContractError(
                f"runtime input[{index}] {field}[{position}] must have type int, "
                f"got {type(dimension).__name__}"
            )
    return frozen


def _observe_input(value: Any, index: int, logical_name: Any) -> TensorSpec:
    try:
        if not isinstance(logical_name, str) or not logical_name:
            raise ValueError("name must be a non-empty string")
        shape = _exact_int_tuple(_runtime_field(value, "shape", index), "shape", index)
        dtype = _symbol(_runtime_field(value, "dtype", index))
        layout = _symbol(_runtime_field(value, "layout", index))
        if layout == "strided":
            layout = "ND"
        stride_value = _runtime_field(value, "stride", index)
        stride = (
            None
            if stride_value is None
            else _exact_int_tuple(stride_value, "stride", index)
        )
        device_type = _symbol(_runtime_field(value, "device_type", index))
        return TensorSpec(
            name=logical_name,
            shape=shape,
            dtype=dtype,
            layout=layout,
            stride=stride,
            device_type=device_type,
        )
    except (TypeError, ValueError) as exc:
        raise MetadataContractError(f"invalid runtime input[{index}]: {exc}") from exc


def _observe_inputs(
    inputs: Any,
    declared: tuple[TensorSpec, ...],
) -> tuple[TensorSpec, ...]:
    if isinstance(inputs, Mapping):
        named_values = tuple(inputs.items())
    else:
        values = _input_values(inputs)
        named_values = tuple(
            (declared[index].name, value)
            for index, value in enumerate(values)
            if index < len(declared)
        )
    actual_count = len(inputs) if isinstance(inputs, Mapping) else len(_input_values(inputs))
    if actual_count != len(declared):
        raise MetadataContractError(
            f"runtime input count {actual_count} does not match ProviderSpec count {len(declared)}"
        )
    return tuple(
        _observe_input(value, index, logical_name)
        for index, (logical_name, value) in enumerate(named_values)
    )


def _match_observed_inputs(
    declared: tuple[TensorSpec, ...],
    observed: tuple[TensorSpec, ...],
) -> None:
    if len(observed) != len(declared):
        raise MetadataContractError(
            f"runtime input count {len(observed)} does not match ProviderSpec count {len(declared)}"
        )
    fields = ("name", "shape", "dtype", "layout", "stride", "device_type")
    for index, (actual, expected) in enumerate(zip(observed, declared)):
        for field in fields:
            actual_value = getattr(actual, field)
            expected_value = getattr(expected, field)
            if actual_value != expected_value:
                raise MetadataContractError(
                    f"runtime input[{index}] {field}={actual_value!r} does not match "
                    f"ProviderSpec value {expected_value!r}"
                )


def adapt_metadata(
    metadata: Any,
    state_lease: StateLease,
    inputs: Any,
) -> AdaptedMetadata:
    """Freeze scheduler metadata and prove it matches runtime input and lease."""
    spec = _freeze_spec(_read(metadata, "provider_spec"))
    runtime_stream_kind = _read(metadata, "runtime_stream_kind")
    if runtime_stream_kind != spec.stream_kind:
        raise MetadataContractError(
            f"runtime_stream_kind={runtime_stream_kind!r} does not match "
            f"ProviderSpec value {spec.stream_kind!r}"
        )
    observed_inputs = _observe_inputs(inputs, spec.inputs)
    _match_observed_inputs(spec.inputs, observed_inputs)
    actual = tuple(_read(metadata, field) for field in _LEASE_FIELDS)
    expected = state_lease.identity
    for field, observed, wanted in zip(_LEASE_FIELDS, actual, expected):
        if observed != wanted:
            raise MetadataContractError(
                f"metadata {field}={observed!r} does not match lease value {wanted!r}"
            )
    return AdaptedMetadata(
        spec,
        actual,
        observed_inputs,
        runtime_stream_kind,
        metadata,
    )


class KTAscendProvider:
    """Provider façade whose real runtime and fallback are explicitly injected."""

    def __init__(self, backend: Any | None = None, *, fallback: Any | None = None) -> None:
        self._backend = backend
        self._fallback = fallback
        self._closed = False
        self._closed_dependency_ids: set[int] = set()

    def supports(self, spec: ProviderSpec) -> SupportDecision:
        if self._closed:
            return SupportDecision.reject(
                "PROVIDER_CLOSED",
                "provider lifecycle is closed",
                fallback_allowed=False,
            )
        return decide_support(spec, backend_available=self._backend is not None)

    def run(
        self,
        inputs: Any,
        metadata: Any,
        state_lease: StateLease,
    ) -> ProviderResult:
        if self._closed:
            raise RuntimeError("provider is closed")

        input_identities = _input_identity_objects(inputs)
        adapted = adapt_metadata(metadata, state_lease, inputs)
        decision = self.supports(adapted.provider_spec)
        if not decision.supported:
            if decision.fallback_allowed and self._fallback is not None:
                result = self._fallback.run(inputs, metadata, state_lease)
                self._validate_result_lease(result, state_lease, "fallback")
                if _aliases_runtime_input(result.output, input_identities):
                    raise RuntimeError("fallback output may not alias provider inputs")
                return result
            raise UnsupportedSpecError(f"{decision.reason_code}: {decision.detail}")

        backend_result = self._backend.run(
            adapted.provider_spec.operation,
            inputs,
            adapted,
            state_lease,
        )
        if not isinstance(backend_result, BackendResult):
            raise TypeError("backend must return BackendResult")
        if backend_result.aliases_input or _aliases_runtime_input(
            backend_result.output, input_identities
        ):
            raise RuntimeError("backend output may not alias provider inputs")
        result = ProviderResult(
            output=backend_result.output,
            completion=backend_result.completion,
            lease=state_lease,
            state_writes=tuple(backend_result.state_writes),
        )
        self._validate_result_lease(result, state_lease, "backend")
        return result

    @staticmethod
    def _validate_result_lease(
        result: Any,
        state_lease: StateLease,
        origin: str,
    ) -> None:
        if not isinstance(result, ProviderResult):
            raise TypeError(f"{origin} must return ProviderResult")
        if result.lease.identity != state_lease.identity:
            raise RuntimeError(f"{origin} result returned a different state lease")

    def close(self) -> None:
        self._closed = True
        attempted_ids: set[int] = set()
        errors: list[Exception] = []
        for dependency in (self._backend, self._fallback):
            if dependency is None:
                continue
            dependency_id = id(dependency)
            if (
                dependency_id in attempted_ids
                or dependency_id in self._closed_dependency_ids
            ):
                continue
            attempted_ids.add(dependency_id)
            close = getattr(dependency, "close", None)
            try:
                if close is not None:
                    close()
            except Exception as exc:
                errors.append(exc)
            else:
                self._closed_dependency_ids.add(dependency_id)
        if errors:
            raise ProviderCloseError(tuple(errors))
