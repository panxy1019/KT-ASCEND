"""Unbound component candidate: one <=32-token working set, BF16 source.

Not enabled by the model launcher. A new physical receipt is required before
integration. Existing copier synchronization and publication are preserved.
"""
RESIDENT_CAPACITY = 32
SHADOW_CAPACITY = 320
# Compatibility for the original mapping-only component draft.
CAPACITY = SHADOW_CAPACITY
COPY_SEGMENT_EXPERTS = 64


def plan_chunk(topk_ids):
    # Keep this candidate importable if layerwise later selects it explicitly.
    from .layerwise import group_topk_routes

    if not 1 <= len(topk_ids) <= 32:
        raise ValueError('chunk workset requires 1..32 explicit token rows')
    groups = group_topk_routes(topk_ids, capacity=CAPACITY, experts=512, topk=10)
    if len(groups) != 1:
        raise RuntimeError('complete Top10 chunk did not fit the workset')
    return groups[0]


class SegmentedCopy:
    """Keep the qualified per-call 400MiB projection ceiling, no new streams."""
    def __init__(self, inner):
        self.inner = inner
        self.state = 'READY'

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def copy_experts(self, layer, moves):
        if self.state != 'READY':
            raise RuntimeError(self.state)
        moves = tuple(moves)
        if (type(layer) is not int or not 0 <= layer < 48 or not moves
                or len(moves) > CAPACITY
                or len({slot for slot, _ in moves}) != len(moves)
                or len({expert for _, expert in moves}) != len(moves)
                or any(type(slot) is not int or not 0 <= slot < CAPACITY
                       or type(expert) is not int or not 0 <= expert < 512
                       for slot, expert in moves)):
            raise ValueError('invalid chunk working-set mapping')
        try:
            for start in range(0, len(moves), COPY_SEGMENT_EXPERTS):
                self.inner.copy_experts(layer, moves[start:start + COPY_SEGMENT_EXPERTS])
        except BaseException:
            # Earlier segments may already have overwritten slots. Never
            # publish/compute or retry this partially prepared allocation.
            self.state = 'BROKEN_RELOAD_REQUIRED'
            raise


def _require_resident_template(template):
    try:
        slots = (template.w13_weight.shape[0], template.w2_weight.shape[0])
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError('resident template must expose both expert weights') from exc
    if slots != (RESIDENT_CAPACITY, RESIDENT_CAPACITY):
        raise ValueError(
            f'resident template must remain {RESIDENT_CAPACITY} slots'
        )


def _require_conservative_reservation(reservation):
    if getattr(reservation, 'capacity', None) != SHADOW_CAPACITY:
        raise ValueError('component reservation must cover the 320-slot shadow')
    try:
        w13_bytes = (SHADOW_CAPACITY * reservation.hidden *
                     2 * reservation.intermediate * 2)
        w2_bytes = (SHADOW_CAPACITY * reservation.intermediate *
                    reservation.hidden * 2)
        bank_bytes = w13_bytes + w2_bytes
        projection_bytes = max(w13_bytes, w2_bytes)
        minimum_reserve = (
            bank_bytes + projection_bytes + reservation.observed_workspace_bytes
        )
        minimum_required = reservation.observed_base_peak_bytes + minimum_reserve
        valid = (
            reservation.private_bank_bytes >= bank_bytes and
            reservation.max_projection_bytes >= projection_bytes and
            reservation.reserve_bytes >= minimum_reserve and
            reservation.required_peak_bytes >= minimum_required and
            reservation.required_peak_bytes <= reservation.allocator_limit_bytes
        )
    except (AttributeError, TypeError) as exc:
        raise ValueError('component requires a validated reservation') from exc
    if not valid:
        raise ValueError('component reservation must retain whole-parent projection')


class WorksetComponentBackend:
    """Fail-closed 320-slot decorator around the existing synchronous backend."""
    status = 'LOCAL_COMPONENT_ONLY_DEVICE_GATE_PENDING'
    resident_capacity = RESIDENT_CAPACITY
    shadow_capacity = SHADOW_CAPACITY

    def __init__(self, inner):
        if getattr(inner, 'capacity', None) != SHADOW_CAPACITY:
            raise ValueError('component backend requires a 320-slot private shadow')
        self.inner = inner
        self.segmented_copy = None
        self.state = 'NEW'

    def _poison(self, error):
        self.state = 'BROKEN_RELOAD_REQUIRED'
        try:
            self.inner.fail_stop(error)
        except BaseException:
            # The copy or cleanup exception remains the primary failure.
            pass

    def preallocate(self, reservation):
        if self.state != 'NEW':
            raise RuntimeError(self.state)
        _require_conservative_reservation(reservation)
        try:
            self.inner.preallocate(reservation)
            adapter = self.inner.adapter
            if adapter is None:
                raise RuntimeError('component backend did not create a copy adapter')
            self.segmented_copy = SegmentedCopy(adapter)
            # CopyOnly remains the outermost surface. The existing adapter may
            # itself already be CopyOnly; retaining that inner guard is benign.
            from .layerwise import _CopyOnlyAdapter
            self.inner.adapter = _CopyOnlyAdapter(self.segmented_copy)
        except BaseException as primary:
            self._poison(primary)
            raise
        self.state = 'READY'

    def execute_group(self, layer, group, hidden_states, topk_weights, identity):
        if self.state != 'READY':
            raise RuntimeError(self.state)
        try:
            return self.inner.execute_group(
                layer, group, hidden_states, topk_weights, identity
            )
        except BaseException as primary:
            if (self.segmented_copy is not None and
                    self.segmented_copy.state != 'READY'):
                self._poison(primary)
            raise

    def synchronize(self, identity):
        # A segmented copy can fail after earlier segments have written. Even
        # after poisoning, retain the original synchronous drain operation.
        try:
            return self.inner.synchronize(identity)
        except BaseException as primary:
            self._poison(primary)
            raise

    def release(self, identity):
        try:
            return self.inner.release(identity)
        except BaseException as primary:
            self._poison(primary)
            raise

    def assemble(self, rows, hidden_states):
        if self.state != 'READY':
            raise RuntimeError(self.state)
        return self.inner.assemble(rows, hidden_states)

    def abort(self, request_id):
        if self.state != 'READY':
            raise RuntimeError(self.state)
        try:
            return self.inner.abort(request_id)
        except BaseException as primary:
            self._poison(primary)
            raise

    def fail_stop(self, error):
        self._poison(error)


def make_component_transport(template_layer, original_model_path, reservation, *,
                             layers=48, experts=512, topk=10,
                             host_staging_limit=64 * 1024**2,
                             backend_factory=None):
    """Build the isolated local component without selecting it in a launcher."""
    _require_resident_template(template_layer)
    _require_conservative_reservation(reservation)

    from .layerwise import AscendBF16TransportBackend, LayerwiseTransport

    factory = backend_factory or AscendBF16TransportBackend
    inner = factory(
        template_layer, original_model_path, capacity=SHADOW_CAPACITY,
        layers=layers, experts=experts, host_staging_limit=host_staging_limit,
    )
    backend = WorksetComponentBackend(inner)
    return LayerwiseTransport(
        backend, reservation, experts=experts, topk=topk, layers=layers,
    )
