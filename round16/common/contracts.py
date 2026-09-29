from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Any, Protocol, Sequence, runtime_checkable


CONTRACT_VERSION = "round16-contract-v2"


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    stride: tuple[int, ...] | None = None
    device_type: str = "cpu"

    def __post_init__(self) -> None:
        frozen_shape = tuple(self.shape)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in frozen_shape):
            raise TypeError("tensor shape dimensions must be integers")
        frozen_stride = None if self.stride is None else tuple(self.stride)
        if frozen_stride is not None and any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in frozen_stride
        ):
            raise TypeError("tensor stride entries must be integers")
        object.__setattr__(self, "shape", frozen_shape)
        object.__setattr__(self, "stride", frozen_stride)
        if type(self.name) is not str or not self.name:
            raise TypeError("tensor name must be a non-empty string")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor shape dimensions must be non-negative")
        if self.stride is not None and len(self.stride) != len(self.shape):
            raise ValueError("tensor stride rank must match shape rank")
        if any(
            type(value) is not str or not value
            for value in (self.dtype, self.layout, self.device_type)
        ):
            raise TypeError(
                "dtype, layout, and device_type must be non-empty strings"
            )

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class ProviderSpec:
    operation: str
    phase: Phase
    inputs: tuple[TensorSpec, ...]
    state_format: str
    stream_kind: str
    metadata_version: str

    def __post_init__(self) -> None:
        frozen_inputs = tuple(self.inputs)
        if any(not isinstance(value, TensorSpec) for value in frozen_inputs):
            raise TypeError("provider inputs must contain TensorSpec values")
        object.__setattr__(self, "inputs", frozen_inputs)
        if type(self.operation) is not str or not self.operation:
            raise TypeError("provider operation must be a non-empty string")
        if not self.inputs:
            raise ValueError("provider inputs must be non-empty")
        if not isinstance(self.phase, Phase):
            raise TypeError("provider phase must be a Phase")
        if any(
            type(value) is not str or not value
            for value in (
                self.state_format,
                self.stream_kind,
                self.metadata_version,
            )
        ):
            raise TypeError(
                "provider state, stream, and metadata identities must be "
                "non-empty strings"
            )


@dataclass(frozen=True)
class StateLease:
    request_id: str
    request_epoch: int
    allocation_generation: int
    batch_generation: int
    layer_id: int
    token_range: tuple[int, int]
    completion_id: str
    writable_state: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        generations = (
            self.request_epoch,
            self.allocation_generation,
            self.batch_generation,
            self.layer_id,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in generations):
            raise TypeError("lease generations and layer_id must be non-negative integers")
        frozen_range = tuple(self.token_range)
        if len(frozen_range) != 2:
            raise ValueError("token_range must contain exactly two entries")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in frozen_range
        ):
            raise TypeError("token_range entries must be non-negative integers")
        frozen_writable = tuple(self.writable_state)
        if any(not isinstance(value, str) or not value for value in frozen_writable):
            raise TypeError("writable_state entries must be non-empty strings")
        object.__setattr__(self, "token_range", frozen_range)
        object.__setattr__(self, "writable_state", frozen_writable)
        if any(
            type(value) is not str or not value
            for value in (self.request_id, self.completion_id)
        ):
            raise TypeError(
                "request_id and completion_id must be non-empty strings"
            )
        if min(generations) < 0:
            raise ValueError("lease generations and layer_id must be non-negative")
        start, end = self.token_range
        if start < 0 or end < start:
            raise ValueError("token_range must be non-negative and ordered")
        if len(set(self.writable_state)) != len(self.writable_state):
            raise ValueError("writable_state contains duplicate entries")

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.request_id,
            self.request_epoch,
            self.allocation_generation,
            self.batch_generation,
            self.layer_id,
            self.token_range,
            self.completion_id,
        )


@dataclass(frozen=True)
class RouteBatch:
    topk_ids: tuple[tuple[int, ...], ...]
    topk_weights: tuple[tuple[float, ...], ...]
    valid_rows: tuple[bool, ...]

    @classmethod
    def freeze(
        cls,
        topk_ids: Sequence[Sequence[int]],
        topk_weights: Sequence[Sequence[float]],
        *,
        valid_rows: Sequence[bool] | None = None,
    ) -> "RouteBatch":
        frozen_ids = tuple(tuple(value for value in row) for row in topk_ids)
        frozen_weights = tuple(tuple(value for value in row) for row in topk_weights)
        if len(frozen_ids) != len(frozen_weights):
            raise ValueError("topk ids and weights must have the same row count")
        frozen_valid = (
            tuple(True for _ in frozen_ids)
            if valid_rows is None
            else tuple(value for value in valid_rows)
        )
        return cls(frozen_ids, frozen_weights, frozen_valid)

    def __post_init__(self) -> None:
        frozen_ids = tuple(tuple(row) for row in self.topk_ids)
        raw_weights = tuple(tuple(row) for row in self.topk_weights)
        frozen_valid = tuple(self.valid_rows)
        if any(
            isinstance(expert_id, bool) or not isinstance(expert_id, int)
            for row in frozen_ids
            for expert_id in row
        ):
            raise TypeError("expert ids must be integers")
        if any(
            isinstance(weight, bool) or not isinstance(weight, Real)
            for row in raw_weights
            for weight in row
        ):
            raise TypeError("route weights must be real numbers")
        if any(not isinstance(value, bool) for value in frozen_valid):
            raise TypeError("valid_rows entries must be booleans")
        frozen_weights = tuple(
            tuple(float(weight) for weight in row) for row in raw_weights
        )
        object.__setattr__(self, "topk_ids", frozen_ids)
        object.__setattr__(self, "topk_weights", frozen_weights)
        object.__setattr__(self, "valid_rows", frozen_valid)
        if len(self.topk_ids) != len(self.topk_weights):
            raise ValueError("topk ids and weights must have the same row count")
        if len(self.valid_rows) != len(self.topk_ids):
            raise ValueError("valid_rows must have the same row count as routes")
        widths = {len(row) for row in self.topk_ids}
        weight_widths = {len(row) for row in self.topk_weights}
        if len(widths) > 1 or widths != weight_widths:
            raise ValueError("routes must have a consistent top-k width")
        if any(expert_id < -1 for row in self.topk_ids for expert_id in row):
            raise ValueError("expert ids may only use -1 as the padding sentinel")
        if any(not isfinite(weight) for row in self.topk_weights for weight in row):
            raise ValueError("route weights must be finite")

    @property
    def token_count(self) -> int:
        return len(self.topk_ids)

    @property
    def top_k(self) -> int:
        return len(self.topk_ids[0]) if self.topk_ids else 0


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    reason_code: str
    detail: str
    fallback_allowed: bool
    requires_device: bool = False

    @classmethod
    def accept(cls, detail: str = "") -> "SupportDecision":
        return cls(True, "SUPPORTED", detail, False, False)

    @classmethod
    def reject(
        cls,
        reason_code: str,
        detail: str,
        *,
        fallback_allowed: bool = True,
    ) -> "SupportDecision":
        return cls(False, reason_code, detail, fallback_allowed, False)

    @classmethod
    def device_required(cls, detail: str) -> "SupportDecision":
        return cls(False, "DEVICE_REQUIRED_NOT_RUN", detail, True, True)


@runtime_checkable
class CompletionBoundary(Protocol):
    completion_id: str

    @property
    def done(self) -> bool: ...

    def wait(self) -> None: ...


@dataclass(frozen=True)
class ProviderResult:
    output: Any
    completion: CompletionBoundary
    lease: StateLease
    state_writes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        frozen_writes = tuple(self.state_writes)
        if any(not isinstance(value, str) or not value for value in frozen_writes):
            raise TypeError("state_writes entries must be non-empty strings")
        object.__setattr__(self, "state_writes", frozen_writes)
        if not isinstance(self.lease, StateLease):
            raise TypeError("lease must be a StateLease")
        if not isinstance(self.completion, CompletionBoundary):
            raise TypeError("completion must implement CompletionBoundary")
        if self.completion.completion_id != self.lease.completion_id:
            raise ValueError("completion_id does not match the active lease")
        unleased = set(self.state_writes).difference(self.lease.writable_state)
        if unleased:
            raise ValueError(f"state writes are not leased: {sorted(unleased)}")


@runtime_checkable
class Provider(Protocol):
    def supports(self, spec: ProviderSpec) -> SupportDecision: ...

    def run(
        self,
        inputs: Any,
        metadata: Any,
        state_lease: StateLease,
    ) -> ProviderResult: ...

    def close(self) -> None: ...
