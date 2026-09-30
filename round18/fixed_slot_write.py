"""Generation-gated writes into one frozen Physical-Q32 expert slot.

This module contains no torch or device imports.  The caller supplies the
three already-resolved destination views and the device copy/barrier hooks.
The writer never allocates or replaces the parent Q32 tensors.

The contract is intentionally narrower than a production expert pool.  It
qualifies only dense BF16 checkpoint rows written into a mutable BF16 Q32
backing.  Quantized and post-load/preprocessed backings need a separate,
format-specific writer and are rejected before any destination write.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Callable


FIXED_SLOT_WRITE_API = "round18:physical-q32-fixed-slot-write:v1"
ELIGIBLE_SOURCE_FORMAT = "BF16_CHECKPOINT_ROWS"
ELIGIBLE_BACKING_FORMAT = "BF16_FIXED_Q32"
PROJECTIONS = ("gate", "up", "down")
SOURCE_SHAPES = {
    "gate": (640, 2560),
    "up": (640, 2560),
    "down": (2560, 640),
}
_BF16_DTYPES = frozenset(("bf16", "bfloat16", "torch.bfloat16"))


class UnsupportedBackingError(ValueError):
    """The source or destination format is not qualified for mutation."""


class SlotNotPublishedError(RuntimeError):
    """The requested slot generation is absent, being written, or invalid."""


class SlotState(str, Enum):
    EMPTY = "EMPTY"
    WRITING = "WRITING"
    PUBLISHED = "PUBLISHED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ProjectionSource:
    projection: str
    shape: tuple[int, ...]
    dtype: str
    payload: Any
    source_format: str = ELIGIBLE_SOURCE_FORMAT

    def __post_init__(self) -> None:
        if self.projection not in PROJECTIONS:
            raise ValueError("source projection must be gate, up, or down")
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise TypeError("source projection shape must be iterable") from error
        if shape != SOURCE_SHAPES[self.projection]:
            raise ValueError(
                f"{self.projection} source shape must be "
                f"{SOURCE_SHAPES[self.projection]}"
            )
        object.__setattr__(self, "shape", shape)
        if type(self.dtype) is not str or not self.dtype:
            raise TypeError("source projection dtype must be a non-empty string")
        if type(self.source_format) is not str or not self.source_format:
            raise TypeError("source format must be a non-empty string")
        if self.payload is None:
            raise ValueError("source projection payload is missing")


@dataclass(frozen=True)
class SourceExpert:
    logical_expert_id: int
    gate: ProjectionSource
    up: ProjectionSource
    down: ProjectionSource

    def __post_init__(self) -> None:
        if type(self.logical_expert_id) is not int or not (
            0 <= self.logical_expert_id < 512
        ):
            raise ValueError("logical expert ID must be in [0,512)")
        for name in PROJECTIONS:
            value = getattr(self, name)
            if not isinstance(value, ProjectionSource) or value.projection != name:
                raise ValueError("source expert must contain exact gate/up/down rows")

    def projection(self, name: str) -> ProjectionSource:
        if name not in PROJECTIONS:
            raise ValueError("projection must be gate, up, or down")
        return getattr(self, name)


@dataclass(frozen=True)
class FixedSlotBacking:
    """Three fixed subviews of one existing physical slot.

    ``gate_destination`` and ``up_destination`` must be views into the two
    halves of the existing w13 slot; ``down_destination`` must be the existing
    w2 slot.  The device adapter creates these views once.  This contract never
    accepts a parent tensor factory or an assignment hook.
    """

    physical_slot: int
    physical_slot_count: int
    dtype: str
    backing_format: str
    gate_destination: Any
    up_destination: Any
    down_destination: Any

    def __post_init__(self) -> None:
        if type(self.physical_slot_count) is not int or self.physical_slot_count != 32:
            raise ValueError("fixed-slot backing requires Physical-Q32")
        if type(self.physical_slot) is not int or not (
            0 <= self.physical_slot < self.physical_slot_count
        ):
            raise ValueError("physical slot is outside the Q32 backing")
        if type(self.dtype) is not str or not self.dtype:
            raise TypeError("destination dtype must be a non-empty string")
        if type(self.backing_format) is not str or not self.backing_format:
            raise TypeError("destination backing format must be non-empty")
        if any(getattr(self, f"{name}_destination") is None for name in PROJECTIONS):
            raise ValueError("fixed slot must expose all three destination views")

    def destination(self, projection: str) -> Any:
        if projection not in PROJECTIONS:
            raise ValueError("projection must be gate, up, or down")
        return getattr(self, f"{projection}_destination")


@dataclass(frozen=True)
class PublishedSlot:
    api: str
    logical_expert_id: int
    physical_slot: int
    generation: int
    projections_written: tuple[str, ...]
    backing_format: str


def _is_bf16(dtype: str) -> bool:
    return dtype.strip().lower() in _BF16_DTYPES


class FixedSlotWriter:
    """Write and publish one existing slot with strict generation ordering."""

    def __init__(
        self,
        *,
        backing: FixedSlotBacking,
        copy_projection: Callable[..., Any],
        publish_barrier: Callable[..., Any],
    ) -> None:
        if not isinstance(backing, FixedSlotBacking):
            raise TypeError("backing must be a FixedSlotBacking")
        if not callable(copy_projection) or not callable(publish_barrier):
            raise TypeError("copy and publish barrier hooks must be callable")
        self.backing = backing
        self.copy_projection = copy_projection
        self.publish_barrier = publish_barrier
        self._lock = RLock()
        self._state = SlotState.EMPTY
        self._published_generation = 0
        self._published_logical_expert_id: int | None = None
        self._failed_generation: int | None = None

    @property
    def state(self) -> SlotState:
        with self._lock:
            return self._state

    @property
    def published_generation(self) -> int:
        with self._lock:
            return self._published_generation

    def _validate_eligibility(self, source: SourceExpert) -> None:
        if not isinstance(source, SourceExpert):
            raise TypeError("source must be one complete SourceExpert")
        if (
            self.backing.backing_format != ELIGIBLE_BACKING_FORMAT
            or not _is_bf16(self.backing.dtype)
        ):
            raise UnsupportedBackingError(
                "only mutable BF16_FIXED_Q32 backing is qualified; "
                "W8A8 and preprocessed backing are ineligible"
            )
        for projection in PROJECTIONS:
            row = source.projection(projection)
            if row.source_format != ELIGIBLE_SOURCE_FORMAT or not _is_bf16(row.dtype):
                raise UnsupportedBackingError(
                    "fixed-slot source must be dense BF16 checkpoint rows"
                )

    def _expected_generation(self) -> int:
        if self._state is SlotState.INVALID and self._failed_generation is not None:
            return self._failed_generation
        return self._published_generation + 1

    def write(self, *, source: SourceExpert, generation: int) -> PublishedSlot:
        """Overwrite all three fixed views and atomically publish metadata.

        The device copy hook must return exactly ``True`` for each complete
        projection.  The barrier must return exactly ``True`` after making all
        three writes visible to the eventual consumer stream.  Any exception
        or negative acknowledgement invalidates the slot; no generation can be
        acquired until the same generation is fully rewritten.
        """

        self._validate_eligibility(source)
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive integer")

        with self._lock:
            if self._state is SlotState.WRITING:
                raise RuntimeError("fixed slot already has an active write")
            expected = self._expected_generation()
            if generation != expected:
                raise ValueError(f"expected generation {expected}, got {generation}")

            self._state = SlotState.WRITING
            try:
                for projection in PROJECTIONS:
                    row = source.projection(projection)
                    acknowledged = self.copy_projection(
                        projection=projection,
                        source=row.payload,
                        source_shape=row.shape,
                        destination=self.backing.destination(projection),
                        logical_expert_id=source.logical_expert_id,
                        physical_slot=self.backing.physical_slot,
                        generation=generation,
                    )
                    if acknowledged is not True:
                        raise RuntimeError(
                            f"device writer did not acknowledge {projection} projection"
                        )
                barrier_complete = self.publish_barrier(
                    physical_slot=self.backing.physical_slot,
                    logical_expert_id=source.logical_expert_id,
                    generation=generation,
                )
                if barrier_complete is not True:
                    raise RuntimeError("device publish barrier was not acknowledged")
            except BaseException:
                self._state = SlotState.INVALID
                self._failed_generation = generation
                raise

            # This is the sole publication point.  It executes only after the
            # complete gate/up/down write and the caller's device barrier.
            self._published_generation = generation
            self._published_logical_expert_id = source.logical_expert_id
            self._failed_generation = None
            self._state = SlotState.PUBLISHED
            return PublishedSlot(
                api=FIXED_SLOT_WRITE_API,
                logical_expert_id=source.logical_expert_id,
                physical_slot=self.backing.physical_slot,
                generation=generation,
                projections_written=PROJECTIONS,
                backing_format=self.backing.backing_format,
            )

    def acquire(self, *, generation: int) -> FixedSlotBacking:
        """Return the stable backing only for the exact published generation."""

        with self._lock:
            if (
                self._state is not SlotState.PUBLISHED
                or type(generation) is not int
                or generation != self._published_generation
            ):
                raise SlotNotPublishedError(
                    f"slot generation {generation!r} is not published"
                )
            return self.backing


__all__ = (
    "ELIGIBLE_BACKING_FORMAT",
    "ELIGIBLE_SOURCE_FORMAT",
    "FIXED_SLOT_WRITE_API",
    "FixedSlotBacking",
    "FixedSlotWriter",
    "PROJECTIONS",
    "ProjectionSource",
    "PublishedSlot",
    "SlotNotPublishedError",
    "SlotState",
    "SourceExpert",
    "UnsupportedBackingError",
)
