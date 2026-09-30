"""Bounded contract for parentless native Ascend expert-list mutation.

The Round18 Phase-II probe owns one independent ``w13`` tensor and one
independent ``w2`` tensor per physical expert slot.  It never creates a
stacked Q32 parent, never takes row views of such a parent, and never rebuilds
the slot list during mutation.  This module intentionally has no torch or
vLLM imports so publication and graph-ordering rules can be verified offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


INDEPENDENT_EXPERT_LIST_API = "round18:ep1-independent-expert-list:v1"
COMPONENT_GRAPH_API = "round18:component-npugraph-mutation:v1"
LOGICAL_EXPERTS = 512
W13_CONSUMER_SHAPE = (2560, 1280)
W2_CONSUMER_SHAPE = (640, 2560)


def _shape(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} has no valid shape") from error
    return result


def _pointer(value: Any, name: str) -> int:
    method = getattr(value, "data_ptr", None)
    if not callable(method):
        raise ValueError(f"{name} has no data_ptr for identity verification")
    pointer = int(method())
    if pointer <= 0:
        raise ValueError(f"{name} has an invalid data_ptr")
    return pointer


def _validate_destination(value: Any, name: str, expected_shape: tuple[int, ...]) -> int:
    if _shape(value, name) != expected_shape:
        raise ValueError(f"{name} must have consumer shape {expected_shape}")
    if getattr(value, "_base", None) is not None:
        raise ValueError(f"{name} must own independent storage, not a parent view")
    if "bfloat16" not in str(getattr(value, "dtype", "")):
        raise ValueError(f"{name} must use BF16")
    device = getattr(getattr(value, "device", None), "type", None)
    if device != "npu":
        raise ValueError(f"{name} must be resident on NPU")
    return _pointer(value, name)


@dataclass(frozen=True)
class IndependentExpertPayload:
    """One logical expert already converted to native consumer orientation."""

    logical_expert_id: int
    w13_source: Any
    w2_source: Any

    def __post_init__(self) -> None:
        if type(self.logical_expert_id) is not int or not (
            0 <= self.logical_expert_id < LOGICAL_EXPERTS
        ):
            raise ValueError("logical expert ID is outside the 512-expert domain")
        if self.w13_source is None or self.w2_source is None:
            raise ValueError("both w13 and w2 source projections are required")


@dataclass(frozen=True)
class IndependentExpertSlot:
    """Two independent native-consumer tensors for one fixed physical slot."""

    physical_slot: int
    w13_destination: Any
    w2_destination: Any

    def __post_init__(self) -> None:
        if type(self.physical_slot) is not int or self.physical_slot < 0:
            raise ValueError("physical slot must be a non-negative integer")
        w13_pointer = _validate_destination(
            self.w13_destination, "w13_destination", W13_CONSUMER_SHAPE
        )
        w2_pointer = _validate_destination(
            self.w2_destination, "w2_destination", W2_CONSUMER_SHAPE
        )
        if w13_pointer == w2_pointer:
            raise ValueError("w13 and w2 must use distinct storage")

    def pointers(self) -> tuple[int, int]:
        return (
            _pointer(self.w13_destination, "w13_destination"),
            _pointer(self.w2_destination, "w2_destination"),
        )


class IndependentListPool:
    """Publish fixed-slot mutations only after both projections are ready.

    ``expert_map`` is allocated once and mutated in place.  The callback is
    responsible for copying that same logical map into the frozen runtime's
    device tensor, or may itself be the device-backed list implementation.
    Internal publication is rolled back if the callback rejects the update.
    Projection bytes are deliberately not rolled back after a failed write;
    the old mapping remains authoritative, so the partially written slot is
    unreachable until a complete retry succeeds.
    """

    def __init__(
        self,
        *,
        slots: Sequence[IndependentExpertSlot],
        initial_slot_to_logical: Sequence[int],
        copy_projection: Callable[..., bool],
        ready_barrier: Callable[..., bool],
        publish_map: Callable[..., bool],
    ) -> None:
        self.slots = tuple(slots)
        if not self.slots:
            raise ValueError("at least one independent expert slot is required")
        if tuple(slot.physical_slot for slot in self.slots) != tuple(
            range(len(self.slots))
        ):
            raise ValueError("physical slots must be dense and ordered from zero")
        initial = tuple(initial_slot_to_logical)
        if len(initial) != len(self.slots) or len(set(initial)) != len(initial):
            raise ValueError("initial slot-to-logical mapping is not one-to-one")
        if any(type(value) is not int or not 0 <= value < LOGICAL_EXPERTS for value in initial):
            raise ValueError("initial logical expert ID is invalid")
        pointers = [pointer for slot in self.slots for pointer in slot.pointers()]
        if len(set(pointers)) != len(pointers):
            raise ValueError("independent expert list contains aliased slot storage")
        for callback, name in (
            (copy_projection, "copy_projection"),
            (ready_barrier, "ready_barrier"),
            (publish_map, "publish_map"),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")

        self._copy_projection = copy_projection
        self._ready_barrier = ready_barrier
        self._publish_map = publish_map
        self._slot_to_logical = list(initial)
        self.expert_map = [-1] * LOGICAL_EXPERTS
        for physical_slot, logical_id in enumerate(initial):
            self.expert_map[logical_id] = physical_slot
        self._initial_pointers = self.pointer_snapshot()
        self.published_generation = 0

    def pointer_snapshot(self) -> tuple[tuple[int, int], ...]:
        return tuple(slot.pointers() for slot in self.slots)

    def replace(
        self,
        *,
        physical_slot: int,
        payload: IndependentExpertPayload,
        generation: int,
    ) -> dict[str, Any]:
        if type(generation) is not int or generation != self.published_generation + 1:
            raise ValueError("mutation must publish the next generation")
        if type(physical_slot) is not int or not 0 <= physical_slot < len(self.slots):
            raise ValueError("physical slot is outside the fixed pool")
        if self.expert_map[payload.logical_expert_id] not in (-1, physical_slot):
            raise ValueError("logical expert is already published in another slot")

        slot = self.slots[physical_slot]
        before_pointers = self.pointer_snapshot()
        for projection, source, destination in (
            ("w13", payload.w13_source, slot.w13_destination),
            ("w2", payload.w2_source, slot.w2_destination),
        ):
            copied = self._copy_projection(
                projection=projection,
                source=source,
                destination=destination,
                physical_slot=physical_slot,
                logical_expert_id=payload.logical_expert_id,
                generation=generation,
            )
            if copied is not True:
                raise RuntimeError(f"{projection} copy did not complete")
        if self._ready_barrier(
            physical_slot=physical_slot,
            logical_expert_id=payload.logical_expert_id,
            generation=generation,
        ) is not True:
            raise RuntimeError("two-projection ready barrier did not complete")

        replaced = self._slot_to_logical[physical_slot]
        old_map = list(self.expert_map)
        self.expert_map[replaced] = -1
        self.expert_map[payload.logical_expert_id] = physical_slot
        try:
            published = self._publish_map(
                mapping=self.expert_map,
                physical_slot=physical_slot,
                logical_expert_id=payload.logical_expert_id,
                generation=generation,
            )
            if published is not True:
                raise RuntimeError("expert map publication did not complete")
        except BaseException:
            self.expert_map[:] = old_map
            raise

        self._slot_to_logical[physical_slot] = payload.logical_expert_id
        self.published_generation = generation
        after_pointers = self.pointer_snapshot()
        if before_pointers != after_pointers or after_pointers != self._initial_pointers:
            raise RuntimeError("independent expert-list data_ptr changed during mutation")
        return {
            "api": INDEPENDENT_EXPERT_LIST_API,
            "physical_slot": physical_slot,
            "logical_expert_id": payload.logical_expert_id,
            "replaced_logical_expert": replaced,
            "published_generation": generation,
            "data_ptr_stable": True,
            "projection_ready_before_publish": True,
            "expert_map_entries": LOGICAL_EXPERTS,
            "parent_constructed": False,
            "parent_rebuilt": False,
            "raw_pointer_copy": False,
        }


class ComponentGraphLedger:
    """Order-only contract for a bounded manual NPUGraph component probe."""

    def __init__(self) -> None:
        self._phase = "idle"
        self._captured_generation: int | None = None
        self._replayed_generation: int | None = None
        self._consumable_generation: int | None = None
        self._capture_output_rejected = False
        self._replay_count = 0

    def begin_capture(self) -> None:
        if self._phase != "idle":
            raise RuntimeError("capture can begin only from idle")
        self._phase = "capturing"

    def end_capture(self, *, output_generation: int) -> None:
        if self._phase != "capturing":
            raise RuntimeError("capture was not active")
        self._captured_generation = int(output_generation)
        self._consumable_generation = None
        self._phase = "captured_needs_replay"

    def record_replay(self, *, output_generation: int) -> None:
        if self._phase not in ("captured_needs_replay", "replay_boundary"):
            raise RuntimeError("replay is outside the legal component boundary")
        self._replayed_generation = int(output_generation)
        self._consumable_generation = int(output_generation)
        self._replay_count += 1
        self._phase = "replayed"

    def consume_output(self, *, generation: int) -> None:
        if self._phase == "captured_needs_replay":
            self._capture_output_rejected = True
            raise RuntimeError("capture output is not a real consumable output")
        if self._phase != "replayed" or self._consumable_generation != generation:
            raise RuntimeError("output generation was not produced by legal replay")
        self._phase = "replay_boundary"

    def begin_external_mutation(self) -> None:
        if self._phase != "replay_boundary":
            raise RuntimeError("mutation requires a completed replay boundary")
        self._phase = "external_mutation"

    def end_external_mutation(self, *, new_generation: int) -> None:
        if self._phase != "external_mutation":
            raise RuntimeError("external mutation was not active")
        if self._replayed_generation is not None and new_generation <= self._replayed_generation:
            raise ValueError("mutation generation must advance")
        self._consumable_generation = None
        self._phase = "replay_boundary"

    def receipt(self) -> dict[str, Any]:
        return {
            "api": COMPONENT_GRAPH_API,
            "scope": "COMPONENT_NPUGRAPH_ONLY",
            "full_model_graph_claim": False,
            "piecewise_used": False,
            "mutation_inside_replay": False,
            "capture_output_rejected": self._capture_output_rejected,
            "replay_count": self._replay_count,
            "last_replayed_generation": self._replayed_generation,
        }


def validate_component_graph_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("api") != COMPONENT_GRAPH_API:
        raise ValueError("component graph receipt API changed")
    if receipt.get("scope") != "COMPONENT_NPUGRAPH_ONLY":
        raise ValueError("manual graph probe must remain component-scoped")
    if receipt.get("full_model_graph_claim") is not False:
        raise ValueError("component graph cannot claim full-model graph qualification")
    if receipt.get("piecewise_used") is not False:
        raise ValueError("PIECEWISE is forbidden for this probe")
    if receipt.get("mutation_inside_replay") is not False:
        raise ValueError("expert mutation must remain outside graph replay")
    if receipt.get("capture_output_rejected") is not True:
        raise ValueError("capture output rejection was not demonstrated")
    if int(receipt.get("replay_count", 0)) < 2:
        raise ValueError("component graph probe requires replay before and after mutation")


__all__ = (
    "COMPONENT_GRAPH_API",
    "INDEPENDENT_EXPERT_LIST_API",
    "ComponentGraphLedger",
    "IndependentExpertPayload",
    "IndependentExpertSlot",
    "IndependentListPool",
    "validate_component_graph_receipt",
)
