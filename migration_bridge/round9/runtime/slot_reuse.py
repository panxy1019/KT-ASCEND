"""Host-only planner for opt-in same-layer private shadow-slot reuse."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotPlan:
    moves: tuple
    remapped_ids: tuple


def _identity_dict(identity):
    if identity is None:
        return None
    return {
        'request_id': identity.request_id,
        'placement_epoch': identity.placement_epoch,
        'allocation_epoch': identity.allocation_epoch,
    }


class SameLayerSlotReuse:
    """Plan changed-slot copies and publish them after the caller's sync."""

    def __init__(self, *, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('slot reuse capacity must be a positive integer')
        self.capacity = capacity
        self._slots = {}
        self._last_used = {}
        self._scope = None
        self._pending = None
        self._requested_slots = 0
        self._cache_hits = 0
        self._changed_slots = 0

    def _clear_committed(self):
        self._slots = {}
        self._last_used = {}
        self._scope = None

    @staticmethod
    def _scope_for(layer, identity):
        return (
            identity.request_id, identity.placement_epoch,
            layer, identity.allocation_epoch,
        )

    def prepare(self, layer, logical_experts, remapped_ids, identity):
        if self._pending is not None:
            raise RuntimeError('a shadow-slot plan is already pending')
        logical = tuple(logical_experts)
        if (type(layer) is not int or layer < 0 or not logical or
                len(logical) > self.capacity or
                any(type(expert) is not int or expert < 0 for expert in logical) or
                tuple(sorted(set(logical))) != logical):
            raise ValueError('logical experts must be sorted, unique and bounded')
        routes = tuple(tuple(row) for row in remapped_ids)
        if any(any(type(index) is not int or not 0 <= index < len(logical)
                   for index in row) for row in routes):
            raise ValueError('group remapped IDs are out of range')

        incoming = self._scope_for(layer, identity)
        continuous = (
            self._scope is not None and
            incoming[:3] == self._scope[:3] and
            incoming[3] == self._scope[3] + 1
        )
        if not continuous:
            self._clear_committed()

        requested = set(logical)
        slots = dict(self._slots)
        last_used = dict(self._last_used)
        hits = requested.intersection(slots)
        misses = requested.difference(slots)
        need = max(0, len(set(slots).union(requested)) - self.capacity)
        victims = sorted(
            set(slots).difference(requested),
            key=lambda expert: (last_used[expert], expert),
        )[:need]
        for expert in victims:
            slots.pop(expert)
            last_used.pop(expert)
        free = sorted(set(range(self.capacity)).difference(slots.values()))
        if len(free) < len(misses):
            raise RuntimeError('shadow-slot plan exceeds fixed capacity')
        moves = tuple(zip(free, sorted(misses)))
        for physical, expert in moves:
            slots[expert] = physical
        for expert in requested:
            last_used[expert] = identity.allocation_epoch
        remapped = tuple(
            tuple(slots[logical[index]] for index in row) for row in routes
        )
        plan = SlotPlan(moves, remapped)
        self._pending = (identity, incoming, slots, last_used, plan)
        self._requested_slots += len(requested)
        self._cache_hits += len(hits)
        self._changed_slots += len(misses)
        return plan

    def publish(self, identity):
        if self._pending is None or self._pending[0] != identity:
            raise RuntimeError('stale shadow-slot publication')
        unused, scope, slots, last_used, unused_plan = self._pending
        self._scope = scope
        self._slots = slots
        self._last_used = last_used
        self._pending = None

    def publish_if_pending(self, identity):
        if self._pending is None:
            return False
        self.publish(identity)
        return True

    def invalidate(self):
        self._pending = None
        self._clear_committed()

    def snapshot(self):
        scope = None
        if self._scope is not None:
            scope = {
                'request_id': self._scope[0],
                'placement_epoch': self._scope[1],
                'logical_layer': self._scope[2],
                'allocation_epoch': self._scope[3],
            }
        return {
            'enabled': True,
            'committed_scope': scope,
            'committed_slots': dict(sorted(self._slots.items())),
            'pending_identity': _identity_dict(
                self._pending[0] if self._pending is not None else None
            ),
            'requested_slots': self._requested_slots,
            'cache_hits': self._cache_hits,
            'changed_slots': self._changed_slots,
        }
