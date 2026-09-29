"""Q32-first selective expert allocation with transactional rollback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


LOGICAL_EXPERT_COUNT = 512
RESIDENT_EXPERT_COUNT = 32


@dataclass(frozen=True)
class SelectiveLoadPlan:
    resident_logical_ids: tuple[int, ...]
    logical_to_physical: tuple[int, ...]
    logical_expert_count: int = LOGICAL_EXPERT_COUNT

    def __post_init__(self) -> None:
        try:
            resident = tuple(self.resident_logical_ids)
            mapping = tuple(self.logical_to_physical)
        except TypeError as error:
            raise ValueError("selection plan ids and mapping must be iterable") from error
        object.__setattr__(self, "resident_logical_ids", resident)
        object.__setattr__(self, "logical_to_physical", mapping)
        _validate_plan(self)

    @property
    def physical_expert_count(self) -> int:
        return len(self.resident_logical_ids)


class RollbackCleanupError(Exception):
    """Aggregate every allocator release failure without hiding the load error."""

    def __init__(self, errors: Sequence[BaseException]) -> None:
        self.errors = tuple(errors)
        details = "; ".join(str(error) for error in self.errors)
        super().__init__(
            f"selective load rollback had {len(self.errors)} cleanup error(s): {details}"
        )


@dataclass(frozen=True)
class _LoadSpec:
    output_name: str
    source_names: tuple[str, ...]
    allocation_shape: tuple[int, ...]


def plan_q32(resident_logical_ids: Sequence[int]) -> SelectiveLoadPlan:
    resident = tuple(resident_logical_ids)
    if any(type(expert_id) is not int for expert_id in resident):
        raise ValueError("resident logical expert ids must be integers")
    if len(resident) != RESIDENT_EXPERT_COUNT:
        raise ValueError("Round16-B selective load requires exactly 32 resident experts")
    if len(set(resident)) != len(resident):
        raise ValueError("resident logical expert ids must be unique")
    if any(expert_id < 0 or expert_id >= LOGICAL_EXPERT_COUNT for expert_id in resident):
        raise ValueError("resident logical expert id is outside [0, 512)")
    mapping = [-1] * LOGICAL_EXPERT_COUNT
    for slot, expert_id in enumerate(resident):
        mapping[expert_id] = slot
    return SelectiveLoadPlan(resident, tuple(mapping))


def _validate_plan(plan: SelectiveLoadPlan) -> None:
    if not isinstance(plan, SelectiveLoadPlan):
        raise ValueError("selection plan must be a SelectiveLoadPlan")
    if (
        type(plan.logical_expert_count) is not int
        or plan.logical_expert_count != LOGICAL_EXPERT_COUNT
    ):
        raise ValueError("selection plan must preserve exactly 512 logical experts")

    resident = plan.resident_logical_ids
    if len(resident) != RESIDENT_EXPERT_COUNT:
        raise ValueError("selection plan must contain exactly 32 resident experts")
    if any(type(expert_id) is not int for expert_id in resident):
        raise ValueError("selection plan resident logical ids must be integers")
    if len(set(resident)) != RESIDENT_EXPERT_COUNT:
        raise ValueError("selection plan resident logical ids must be unique")
    if any(
        expert_id < 0 or expert_id >= LOGICAL_EXPERT_COUNT
        for expert_id in resident
    ):
        raise ValueError("selection plan resident logical id is outside [0, 512)")

    mapping = plan.logical_to_physical
    if len(mapping) != LOGICAL_EXPERT_COUNT:
        raise ValueError("selection plan logical-to-physical map must have 512 entries")
    if any(type(physical_id) is not int for physical_id in mapping):
        raise ValueError("selection plan physical ids must be integers")
    expected = [-1] * LOGICAL_EXPERT_COUNT
    for physical_id, logical_id in enumerate(resident):
        expected[logical_id] = physical_id
    if tuple(mapping) != tuple(expected):
        raise ValueError(
            "selection plan mapping must exactly match resident logical ids and "
            "continuous physical ids [0, 32)"
        )


def _source_shape(source: Any) -> tuple[int, ...]:
    try:
        shape = tuple(getattr(source, "shape", ()))
    except TypeError as error:
        raise ValueError("source shape dimensions must be integers") from error
    if any(type(value) is not int for value in shape):
        raise ValueError("source shape dimensions must be integers")
    if not shape or shape[0] != LOGICAL_EXPERT_COUNT:
        raise ValueError("source expert bank must have a leading logical dimension of 512")
    return shape


def _read_selected(source: Any, expert_ids: tuple[int, ...]) -> tuple[Any, ...]:
    reader = getattr(source, "read_experts", None)
    if callable(reader):
        rows = reader(expert_ids)
    else:
        rows = tuple(source[expert_id] for expert_id in expert_ids)
    rows = tuple(rows)
    if len(rows) != len(expert_ids):
        raise ValueError("selective expert read returned the wrong row count")
    return rows


def _fuse_rows(gate_rows: Sequence[Any], up_rows: Sequence[Any]) -> tuple[Any, ...]:
    fused = []
    for gate, up in zip(gate_rows, up_rows):
        try:
            fused.append(tuple(gate) + tuple(up))
        except TypeError:
            fused.append((gate, up))
    return tuple(fused)


class SelectiveExpertLoader:
    """Allocate only planned physical experts and preserve parameter aliases."""

    def __init__(self, allocator: Any) -> None:
        for method in ("allocate", "write", "release"):
            if not callable(getattr(allocator, method, None)):
                raise TypeError(f"allocator must expose {method}()")
        self._allocator = allocator

    def _plan_specs(
        self,
        sources: Mapping[str, Any],
        fused_gate_up: tuple[str, str, str] | None,
    ) -> tuple[tuple[_LoadSpec, ...], dict[str, str]]:
        source_shapes = {name: _source_shape(source) for name, source in sources.items()}
        specs: list[_LoadSpec] = []
        aliases: dict[str, str] = {}
        consumed: set[str] = set()
        consumed_source_ids: set[int] = set()
        if fused_gate_up is not None:
            gate_name, up_name, output_name = fused_gate_up
            if gate_name not in sources or up_name not in sources:
                raise ValueError("fused gate/up sources are missing")
            if output_name in sources and output_name not in (gate_name, up_name):
                raise ValueError(
                    "name ownership conflict: fused output shadows an unconsumed source"
                )
            gate_source_id = id(sources[gate_name])
            up_source_id = id(sources[up_name])
            fused_source_ids = {gate_source_id, up_source_id}
            source_aliases = [
                name
                for name, source in sources.items()
                if name not in (gate_name, up_name) and id(source) in fused_source_ids
            ]
            if gate_source_id == up_source_id or source_aliases:
                raise ValueError(
                    "fused gate/up source alias requires an unfused explicit load plan"
                )
            gate_shape = source_shapes[gate_name]
            up_shape = source_shapes[up_name]
            if len(gate_shape) != len(up_shape) or gate_shape[2:] != up_shape[2:]:
                raise ValueError("fused gate/up source shapes are incompatible")
            specs.append(
                _LoadSpec(
                    output_name,
                    (gate_name, up_name),
                    (
                        RESIDENT_EXPERT_COUNT,
                        gate_shape[1] + up_shape[1],
                        *gate_shape[2:],
                    ),
                )
            )
            consumed.update((gate_name, up_name))
            consumed_source_ids.update((gate_source_id, up_source_id))

        canonical_by_source: dict[int, str] = {}
        for name, source in sources.items():
            if name in consumed:
                continue
            source_id = id(source)
            if source_id in consumed_source_ids:
                continue
            canonical = canonical_by_source.get(source_id)
            if canonical is not None:
                aliases[name] = canonical
                continue
            canonical_by_source[source_id] = name
            shape = source_shapes[name]
            specs.append(
                _LoadSpec(name, (name,), (RESIDENT_EXPERT_COUNT, *shape[1:]))
            )
        if any(spec.allocation_shape[0] != RESIDENT_EXPERT_COUNT for spec in specs):
            raise RuntimeError("full-expert temporary allocation plans are forbidden")
        return tuple(specs), aliases

    @staticmethod
    def _validate_aliases(
        specs: Sequence[_LoadSpec],
        source_aliases: Mapping[str, str],
        explicit_aliases: Mapping[str, str],
    ) -> None:
        output_names = tuple(spec.output_name for spec in specs)
        if len(set(output_names)) != len(output_names):
            raise ValueError("name ownership conflict: output names must be unique")
        loaded_names = set(output_names)
        planned_names = loaded_names | set(source_aliases)

        for alias, canonical in source_aliases.items():
            if alias in loaded_names or canonical not in loaded_names:
                raise ValueError(
                    "name ownership conflict: source alias hides an output allocation"
                )
        for alias, canonical in explicit_aliases.items():
            if alias in planned_names:
                raise ValueError(
                    "name ownership conflict: explicit alias hides a planned allocation"
                )
            if canonical not in loaded_names:
                raise ValueError(f"alias target {canonical!r} was not loaded")

    def load(
        self,
        plan: SelectiveLoadPlan,
        sources: Mapping[str, Any],
        *,
        fused_gate_up: tuple[str, str, str] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        _validate_plan(plan)
        resident_logical_ids = plan.resident_logical_ids
        source_snapshot = dict(sources)
        explicit_aliases = dict(aliases or {})
        specs, source_aliases = self._plan_specs(source_snapshot, fused_gate_up)
        self._validate_aliases(specs, source_aliases, explicit_aliases)
        loaded: dict[str, Any] = {}
        allocated: list[Any] = []
        allocation_owners: dict[int, str] = {}
        try:
            for spec in specs:
                allocation = self._allocator.allocate(spec.output_name, spec.allocation_shape)
                allocated.append(allocation)
                allocation_id = id(allocation)
                if allocation_id in allocation_owners:
                    owner = allocation_owners[allocation_id]
                    raise RuntimeError(
                        "allocator ownership conflict: "
                        f"{spec.output_name!r} reused allocation owned by {owner!r}"
                    )
                allocation_owners[allocation_id] = spec.output_name
                rows = [
                    _read_selected(source_snapshot[name], resident_logical_ids)
                    for name in spec.source_names
                ]
                data = rows[0] if len(rows) == 1 else _fuse_rows(rows[0], rows[1])
                self._allocator.write(allocation, data)
                loaded[spec.output_name] = allocation
            for alias, canonical in {**source_aliases, **explicit_aliases}.items():
                loaded[alias] = loaded[canonical]
            return loaded
        except BaseException as primary_error:
            released: set[int] = set()
            cleanup_errors: list[BaseException] = []
            for allocation in reversed(allocated):
                allocation_id = id(allocation)
                if allocation_id not in released:
                    released.add(allocation_id)
                    try:
                        self._allocator.release(allocation)
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                aggregate = RollbackCleanupError(cleanup_errors)
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(str(aggregate))
                raise primary_error from aggregate
            raise


__all__ = (
    "LOGICAL_EXPERT_COUNT",
    "RESIDENT_EXPERT_COUNT",
    "RollbackCleanupError",
    "SelectiveExpertLoader",
    "SelectiveLoadPlan",
    "plan_q32",
)
