"""Nonblocking CPU-side accounting for shared host-work resources.

The budget limits outstanding allocation bytes; it does not measure or promise
DRAM/H2D bandwidth. Admission is immediate and has no waiter queue, so the
priority policy makes no starvation-freedom or fairness claim. Callers drive
progress by cancelling optional prefetch, completing in-flight work, releasing
its allocation, and retrying admission.

Request IDs must identify one live request lifecycle. They must not be reused
for another logical request until ``retire_request`` succeeds; retirement is
allowed only after every allocation owned by that request has been released.
"""
from dataclasses import dataclass
from enum import IntEnum
import threading

from .framework_config import BufferIdentity


_RESOURCE_NAMES = ('host_bytes', 'h2d_bytes', 'pinned_bytes')


class Priority(IntEnum):
    """Admission importance, highest first."""

    PLE_CURRENT = 3
    EXPERT_MIGRATION = 2
    PLE_PREFETCH = 1


@dataclass(frozen=True, slots=True)
class Allocation:
    """Immutable identity and credits for one admitted unit of work."""

    identity: BufferIdentity
    priority: Priority
    host_bytes: int
    h2d_bytes: int
    pinned_bytes: int

    @property
    def allocation_epoch(self):
        return self.identity.allocation_epoch


def _byte_count(value):
    return type(value) is int and value >= 0


class HostWorkBudget:
    """Atomically admit bounded host work without blocking caller threads."""

    def __init__(
        self,
        *,
        enabled=False,
        host_bytes=0,
        h2d_bytes=0,
        pinned_bytes=0,
        reserve_host_bytes=0,
        reserve_h2d_bytes=0,
        reserve_pinned_bytes=0,
    ):
        capacities = {
            'host_bytes': host_bytes,
            'h2d_bytes': h2d_bytes,
            'pinned_bytes': pinned_bytes,
        }
        reserves = {
            'host_bytes': reserve_host_bytes,
            'h2d_bytes': reserve_h2d_bytes,
            'pinned_bytes': reserve_pinned_bytes,
        }
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        if any(not _byte_count(value) for value in (*capacities.values(), *reserves.values())):
            raise ValueError('capacities and reserves must be nonnegative integers')
        if any(reserves[name] > capacities[name] for name in _RESOURCE_NAMES):
            raise ValueError('reserve cannot exceed capacity')

        self.enabled = enabled
        self._capacity = capacities
        self._reserve = reserves
        self._used = dict.fromkeys(_RESOURCE_NAMES, 0)
        self._allocations = {}
        self._latest_placement = {}
        self._next_epoch = 0
        self._lock = threading.Lock()

    def try_acquire(
        self,
        *,
        request_id,
        placement_epoch,
        priority,
        host_bytes,
        h2d_bytes,
        pinned_bytes,
    ):
        """Return an allocation immediately, or ``None`` when credits are unavailable."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('request_id must be a nonempty string')
        if type(placement_epoch) is not int or placement_epoch < 0:
            raise ValueError('placement_epoch must be a nonnegative integer')
        if type(priority) is not Priority:
            raise ValueError('priority must be a Priority')
        requested = {
            'host_bytes': host_bytes,
            'h2d_bytes': h2d_bytes,
            'pinned_bytes': pinned_bytes,
        }
        if any(not _byte_count(value) for value in requested.values()):
            raise ValueError('credits must be nonnegative integers')
        if not any(requested.values()):
            raise ValueError('at least one credit must be requested')

        with self._lock:
            if not self.enabled:
                return None
            latest = self._latest_placement.get(request_id)
            if latest is not None and placement_epoch < latest:
                raise ValueError('stale placement epoch')

            # Optional prefetch cannot consume the reserve. Correctness traffic
            # may use the full budget but is still never overcommitted.
            limits = self._capacity
            if priority is Priority.PLE_PREFETCH:
                limits = {
                    name: self._capacity[name] - self._reserve[name]
                    for name in _RESOURCE_NAMES
                }
            if any(self._used[name] + requested[name] > limits[name]
                   for name in _RESOURCE_NAMES):
                return None

            epoch = self._next_epoch
            self._next_epoch += 1
            allocation = Allocation(
                identity=BufferIdentity(request_id, placement_epoch, epoch),
                priority=priority,
                **requested,
            )
            self._allocations[epoch] = {
                'allocation': allocation,
                'in_flight': False,
                'cancelled': False,
            }
            self._latest_placement[request_id] = placement_epoch
            for name in _RESOURCE_NAMES:
                self._used[name] += requested[name]
            return allocation

    def mark_in_flight(self, allocation):
        """Record that an allocation now owns buffers used by asynchronous work."""
        with self._lock:
            state = self._state(allocation)
            if state['cancelled']:
                raise RuntimeError('cancelled allocation cannot start')
            if state['in_flight']:
                raise RuntimeError('allocation is already in flight')
            state['in_flight'] = True

    def complete(self, allocation):
        """Record transfer completion; credits remain held until ``release``."""
        with self._lock:
            state = self._state(allocation)
            if not state['in_flight']:
                raise RuntimeError('allocation is not in flight')
            state['in_flight'] = False

    def cancel(self, allocation):
        """Idempotently mark live work cancelled without reclaiming its credits."""
        with self._lock:
            state = self._state(allocation)
            if state['cancelled']:
                return
            state['cancelled'] = True

    def cancel_prefetches(self):
        """Cancel all live optional prefetches and return their allocations.

        This is deliberately a signalling operation. Even prefetches that have
        not started retain credits until their owner explicitly releases them.
        """
        with self._lock:
            cancelled = []
            for state in self._allocations.values():
                allocation = state['allocation']
                if (allocation.priority is Priority.PLE_PREFETCH
                        and not state['cancelled']):
                    state['cancelled'] = True
                    cancelled.append(allocation)
            return tuple(cancelled)

    def release(self, allocation):
        """Return credits after work is idle; releasing in-flight work is forbidden."""
        with self._lock:
            state = self._state(allocation)
            if state['in_flight']:
                raise RuntimeError('in-flight allocation must complete before release')
            del self._allocations[allocation.allocation_epoch]
            for name in _RESOURCE_NAMES:
                self._used[name] -= getattr(allocation, name)

    def retire_request(self, request_id):
        """Forget a drained request's placement cursor and permit ID reuse."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('request_id must be a nonempty string')
        with self._lock:
            if request_id not in self._latest_placement:
                raise ValueError('unknown or already retired request')
            if any(state['allocation'].identity.request_id == request_id
                   for state in self._allocations.values()):
                raise RuntimeError('request still owns allocations')
            del self._latest_placement[request_id]

    def snapshot(self):
        """Return aggregate accounting without exposing mutable internal state."""
        with self._lock:
            states = tuple(self._allocations.values())
            return {
                'enabled': self.enabled,
                'capacity': dict(self._capacity),
                'reserve': dict(self._reserve),
                'used': dict(self._used),
                'allocations': len(states),
                'active_requests': len(self._latest_placement),
                'in_flight': sum(state['in_flight'] for state in states),
                'cancelled': sum(state['cancelled'] for state in states),
            }

    def _state(self, allocation):
        if not isinstance(allocation, Allocation):
            raise ValueError('invalid allocation identity')
        state = self._allocations.get(allocation.allocation_epoch)
        if state is None or state['allocation'] is not allocation:
            raise ValueError('stale or mismatched allocation identity')
        return state
