"""Bounded route-grouped BF16 expert transport groundwork.

The scheduler integration is deliberately outside this module.  The generic
transport is host-testable through an injected backend; the Ascend backend is a
real consumer of the original-BF16 reader, the qualified KTEP batch copier and
SGLang's unquantized fused MoE method.  Its device path remains unqualified
until a separately admitted probe exercises it.
"""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

from .framework_config import BufferIdentity


_ROUND9_ALLOCATOR_LIMIT_BYTES = 43_008 * 1024**2
_ROUND9_MAX_GROUP_ROWS = 32


@dataclass(frozen=True)
class RouteGroup:
    rows: tuple
    logical_experts: tuple
    remapped_ids: tuple


@dataclass(frozen=True)
class TransportReservation:
    capacity: int
    hidden: int
    intermediate: int
    max_group_rows: int
    private_bank_bytes: int
    max_projection_bytes: int
    observed_workspace_bytes: int
    observed_base_peak_bytes: int
    reserve_bytes: int
    required_peak_bytes: int
    allocator_limit_bytes: int


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _host_rows(value):
    """Convert a device tensor to plain rows without importing torch here."""
    detach = getattr(value, 'detach', None)
    cpu = getattr(value, 'cpu', None)
    tolist = getattr(value, 'tolist', None)
    if callable(detach) and callable(cpu) and callable(tolist):
        return value.detach().cpu().tolist()
    return value


def make_reservation(*, capacity, hidden, intermediate, max_group_rows,
                     allocator_limit_bytes, observed_base_peak_bytes,
                     observed_workspace_bytes):
    """Return the bytes a caller must reserve before sizing the KV pool."""
    for name, value in (('capacity', capacity), ('hidden', hidden),
                        ('intermediate', intermediate),
                        ('max_group_rows', max_group_rows),
                        ('allocator_limit_bytes', allocator_limit_bytes)):
        _positive_int(value, name)
    if max_group_rows > _ROUND9_MAX_GROUP_ROWS:
        raise ValueError('max_group_rows exceeds the admitted Round9 bound')
    if allocator_limit_bytes > _ROUND9_ALLOCATOR_LIMIT_BYTES:
        raise ValueError('allocator limit exceeds the admitted Round9 42GiB cap')
    for name, value in (('observed_base_peak_bytes', observed_base_peak_bytes),
                        ('observed_workspace_bytes', observed_workspace_bytes)):
        _positive_int(value, name)
    # Processed Ascend tensors are [E,H,2I] and [E,I,H], both BF16.
    w13_bytes = capacity * hidden * 2 * intermediate * 2
    w2_bytes = capacity * intermediate * hidden * 2
    bank = w13_bytes + w2_bytes
    projection = max(w13_bytes, w2_bytes)
    reserve = bank + projection + observed_workspace_bytes
    required = observed_base_peak_bytes + reserve
    if required > allocator_limit_bytes:
        raise ValueError(
            f'layerwise reservation exceeds allocator: {required}>{allocator_limit_bytes}'
        )
    return TransportReservation(
        capacity, hidden, intermediate, max_group_rows, bank, projection,
        observed_workspace_bytes, observed_base_peak_bytes, reserve, required,
        allocator_limit_bytes,
    )


def group_topk_routes(topk_ids, *, capacity=32, experts=512, topk=10):
    """Group contiguous token rows without splitting any token's complete routes."""
    _positive_int(capacity, 'capacity')
    _positive_int(experts, 'experts')
    _positive_int(topk, 'topk')
    if capacity < topk:
        raise ValueError('capacity cannot hold one complete TopK route set')
    try:
        rows = [tuple(row) for row in _host_rows(topk_ids)]
    except TypeError as exc:
        raise ValueError('topk_ids must be a sequence of rows') from exc
    if not rows:
        raise ValueError('topk_ids cannot be empty')
    for row in rows:
        if len(row) != topk:
            raise ValueError('every token must contain the complete TopK routes')
        if (any(type(expert) is not int or not 0 <= expert < experts for expert in row)
                or len(set(row)) != topk):
            raise ValueError('routes must be distinct in-range logical expert IDs')

    result = []
    group_rows = []
    group_experts = set()

    def emit():
        logical = tuple(sorted(group_experts))
        mapping = {expert: physical for physical, expert in enumerate(logical)}
        remapped = tuple(tuple(mapping[expert] for expert in rows[row])
                         for row in group_rows)
        result.append(RouteGroup(tuple(group_rows), logical, remapped))

    for row_index, row in enumerate(rows):
        added = set(row) - group_experts
        if group_rows and len(group_experts) + len(added) > capacity:
            emit()
            group_rows = []
            group_experts = set()
        group_rows.append(row_index)
        group_experts.update(row)
    emit()
    return tuple(result)


_SHADOW_SCALARS = (
    'top_k', 'moe_ep_size', 'moe_ep_rank', 'moe_tp_size', 'moe_tp_rank',
    'hidden_size', 'intermediate_size_per_partition',
    'num_fused_shared_experts',
)


def _build_private_shadow(template, runner_config, capacity, module_factory):
    """Construct a new module from the scalar surface used by NPU unquant MoE."""
    shadow = module_factory()
    for name in _SHADOW_SCALARS:
        value = getattr(template, name)
        if type(value) is not int or value < 0:
            raise ValueError(f'invalid private shadow scalar {name}')
        setattr(shadow, name, value)
    if shadow.top_k <= 0 or shadow.moe_ep_size <= 0 or shadow.moe_tp_size <= 0:
        raise ValueError('private shadow requires positive TopK/EP/TP sizes')
    shadow.num_experts = capacity
    shadow.num_local_experts = capacity
    shadow.num_gpu_experts = capacity
    shadow.moe_runner_config = runner_config
    return shadow


class _CopyOnlyAdapter:
    """Narrow facade: qualified copy math without placement publication APIs."""
    __slots__ = ('__inner',)

    def __init__(self, inner):
        self.__inner = inner

    @property
    def copied_bytes(self):
        return self.__inner.copied_bytes

    def copy_experts(self, layer, moves):
        return self.__inner.copy_experts(layer, moves)

    def synchronize(self):
        return self.__inner.synchronize()


class LayerwiseTransport:
    """Serial request/placement lease for one complete layer span."""
    def __init__(self, backend, reservation, *, experts=512, topk=10, layers=48):
        if not isinstance(reservation, TransportReservation):
            raise TypeError('a validated TransportReservation is required')
        for name, value in (('experts', experts), ('topk', topk), ('layers', layers)):
            _positive_int(value, name)
        if reservation.capacity < topk or reservation.capacity > experts:
            raise ValueError('reservation capacity is incompatible with routing')
        for name in ('preallocate', 'execute_group', 'synchronize', 'release',
                     'assemble', 'abort', 'fail_stop'):
            if not callable(getattr(backend, name, None)):
                raise TypeError(f'backend is missing {name}()')
        self.backend = backend
        self.reservation = reservation
        self.experts = experts
        self.topk = topk
        self.layers = layers
        self._active = None
        self._next_layer = 0
        self._allocation_epoch = 0
        self.state = 'READY'
        backend.preallocate(reservation)

    def _ready(self):
        if self.state != 'READY':
            raise RuntimeError(self.state)

    def _fail_stop(self, error):
        self.state = 'BROKEN_RELOAD_REQUIRED'
        try:
            self.backend.fail_stop(error)
        except BaseException:
            # Retain the causative execution/synchronization exception. A
            # backend whose fail-stop hook itself failed is still unusable.
            pass

    def begin(self, request_id, placement_epoch):
        self._ready()
        identity = BufferIdentity(request_id, placement_epoch, 0)
        if self._active is not None:
            raise RuntimeError('a layerwise request is already active')
        self._active = identity
        self._next_layer = 0
        return identity

    def _require(self, request_id, placement_epoch):
        self._ready()
        if self._active is None:
            raise RuntimeError('no active layerwise request')
        if (self._active.request_id != request_id or
                self._active.placement_epoch != placement_epoch):
            raise RuntimeError('stale request or placement epoch')

    @staticmethod
    def _rows(value, expected, width=None, name='value'):
        try:
            rows = list(value)
        except TypeError as exc:
            raise ValueError(f'{name} must expose token rows') from exc
        if len(rows) != expected:
            raise ValueError(f'{name} token count mismatch')
        if width is not None:
            for row in rows:
                if len(row) != width:
                    raise ValueError(f'{name} TopK width mismatch')
        return rows

    def execute_layer(self, request_id, placement_epoch, layer, hidden_states,
                      topk_ids, topk_weights):
        self._require(request_id, placement_epoch)
        if type(layer) is not int or layer != self._next_layer:
            raise RuntimeError(f'expected layer {self._next_layer}, got {layer!r}')
        try:
            token_count = len(topk_ids)
        except TypeError as exc:
            raise ValueError('topk_ids must expose token rows') from exc
        if token_count > self.reservation.max_group_rows:
            raise ValueError('token count exceeds reserved max_group_rows')
        hidden = self._rows(hidden_states, token_count, name='hidden_states')
        weights = self._rows(
            _host_rows(topk_weights), token_count, self.topk, 'topk_weights'
        )
        for row in weights:
            for weight in row:
                if (type(weight) not in (int, float) or
                        not math.isfinite(weight) or weight < 0):
                    raise ValueError('topk_weights must be finite and non-negative')
        groups = group_topk_routes(
            topk_ids, capacity=self.reservation.capacity,
            experts=self.experts, topk=self.topk,
        )
        output_rows = [None] * token_count
        try:
            for group in groups:
                self._allocation_epoch += 1
                identity = BufferIdentity(
                    request_id, placement_epoch, self._allocation_epoch
                )
                primary = None
                try:
                    values = self.backend.execute_group(
                        layer, group, hidden_states, topk_weights, identity
                    )
                    values = list(values)
                    if len(values) != len(group.rows):
                        raise RuntimeError('backend returned the wrong number of rows')
                    for row, value in zip(group.rows, values):
                        output_rows[row] = value
                except BaseException as exc:
                    primary = exc
                try:
                    self.backend.synchronize(identity)
                except BaseException as cleanup:
                    self._fail_stop(cleanup)
                    if primary is not None:
                        raise primary from cleanup
                    raise
                try:
                    self.backend.release(identity)
                except BaseException as cleanup:
                    self._fail_stop(cleanup)
                    if primary is not None:
                        raise primary from cleanup
                    raise
                if primary is not None:
                    raise primary
            if any(value is None for value in output_rows):
                raise RuntimeError('route groups did not produce every token row')
            result = self.backend.assemble(output_rows, hidden_states)
        except BaseException as primary:
            if self.state == 'READY' and self._active is not None:
                try:
                    self.backend.abort(request_id)
                except BaseException as cleanup:
                    self._fail_stop(cleanup)
                    raise primary from cleanup
                self._active = None
                self._next_layer = 0
            raise
        self._next_layer += 1
        return result

    def finish(self, request_id, placement_epoch):
        self._require(request_id, placement_epoch)
        if self._next_layer != self.layers:
            raise RuntimeError('layerwise phase is incomplete')
        identity = self._active
        self._active = None
        self._next_layer = 0
        return identity

    def abort(self, request_id, placement_epoch):
        self._require(request_id, placement_epoch)
        try:
            self.backend.abort(request_id)
        except BaseException as exc:
            self._fail_stop(exc)
            raise
        self._active = None
        self._next_layer = 0


class AscendBF16TransportBackend:
    """Private NZ29 shadow-MoE backend; DEVICE_PENDING until a gated probe."""
    status = 'DEVICE_PENDING_LAYERWISE_BF16_TRANSPORT'

    def __init__(self, template_layer, original_model_path, *, capacity=32,
                 layers=48, experts=512, host_staging_limit=64 * 1024**2,
                 enable_same_layer_slot_reuse=False):
        if type(enable_same_layer_slot_reuse) is not bool:
            raise ValueError('same-layer slot reuse flag must be boolean')
        self.template_layer = template_layer
        self.original_model_path = original_model_path
        self.capacity = _positive_int(capacity, 'capacity')
        self.layers = _positive_int(layers, 'layers')
        self.experts = _positive_int(experts, 'experts')
        self.host_staging_limit = _positive_int(
            host_staging_limit, 'host_staging_limit'
        )
        self.shadow = None
        self.method = None
        self.adapter = None
        self._current = None
        self._torch = None
        self._synchronized = None
        self.state = 'NEW'
        self.measured_preallocation_peak_bytes = None
        self._slot_reuse = None
        if enable_same_layer_slot_reuse:
            if self.capacity != 32 or self.layers != 48:
                raise ValueError('same-layer slot reuse requires the fixed shadow32 all48 profile')
            from .slot_reuse import SameLayerSlotReuse
            self._slot_reuse = SameLayerSlotReuse(capacity=self.capacity)

    def slot_reuse_snapshot(self):
        if self._slot_reuse is not None:
            return self._slot_reuse.snapshot()
        return {
            'enabled': False,
            'committed_scope': None,
            'committed_slots': {},
            'pending_identity': None,
            'requested_slots': 0,
            'cache_hits': 0,
            'changed_slots': 0,
        }

    def preallocate(self, reservation):
        if self.state != 'NEW':
            raise RuntimeError(self.state)
        if (reservation.capacity != self.capacity or
                reservation.hidden <= 0 or reservation.intermediate <= 0):
            raise ValueError('reservation does not match the private shadow bank')
        import gc
        import torch
        import torch_npu
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from .dynamic_adapter import TorchKTEPAdapter
        from .expert_source import OriginalExpertSource

        source = shadow = method = adapter = raw_adapter = layer_map = method_map = None
        device = None
        try:
            source = OriginalExpertSource(
                self.original_model_path, layers=self.layers, experts=self.experts,
                hidden=reservation.hidden, intermediate=reservation.intermediate,
            )
            device = next(self.template_layer.parameters()).device
            if device.type != 'npu':
                raise RuntimeError('Ascend layerwise transport requires an NPU layer')
            torch.npu.synchronize(device)
            torch.npu.reset_peak_memory_stats(device)
            baseline = int(torch.npu.memory_allocated(device))
            runner_config = replace(
                self.template_layer.moe_runner_config,
                num_experts=self.capacity, num_local_experts=self.capacity,
                routed_scaling_factor=None,
            )
            shadow = _build_private_shadow(
                self.template_layer, runner_config, self.capacity,
                torch.nn.Module,
            )
            method = UnquantizedFusedMoEMethod(
                use_triton_kernels=False, use_flashinfer_trtllm_moe=False
            )
            # Ascend grouped-MoE accumulates all TopK contributions in FP32;
            # assemble() performs the single final cast back to the input dtype.
            method._kt_return_fp32_contribution = True
            shadow.quant_method = method
            method.create_weights(
                shadow, self.capacity, reservation.hidden,
                reservation.intermediate, torch.bfloat16,
            )
            for parameter in shadow.parameters():
                parameter.data = parameter.data.to(device)
            method.create_moe_runner(shadow, runner_config)
            method.process_weights_after_loading(shadow)
            layer_map = {layer: shadow for layer in range(self.layers)}
            method_map = {}
            for layer in range(self.layers):
                # Shape/dtype sentinels satisfy the unchanged qualified copier.
                # They intentionally do not describe placement: this backend
                # passes physical IDs directly and exposes no publish/controller
                # surface through _CopyOnlyAdapter.
                mask = torch.zeros(self.experts, dtype=torch.bool, device='cpu')
                mapping = torch.full(
                    (self.experts,), -1, dtype=torch.int32, device='cpu'
                )
                reverse = torch.arange(
                    self.capacity, dtype=torch.int32, device='cpu'
                )
                method_map[layer] = SimpleNamespace(
                    gpu_experts_mask=mask,
                    gpu_experts_mask_cuda=mask.to(device),
                    logical_to_gpu_index=mapping,
                    logical_to_gpu_index_cuda=mapping.to(device),
                    gpu_index_to_logical=reverse,
                    wrapper=SimpleNamespace(gpu_experts_mask=mask.clone()),
                )
            raw_adapter = TorchKTEPAdapter(
                layer_map, method_map, source,
                complete_cpu_backing_ids=range(self.experts),
                host_staging_limit=self.host_staging_limit,
                allow_unqualified_npu_probe=True,
                qualified_npu_formats=(29,),
            )
            adapter = _CopyOnlyAdapter(raw_adapter)
            torch.npu.synchronize(device)
            actual = sum(t.numel() * t.element_size()
                         for t in (shadow.w13_weight, shadow.w2_weight))
            if actual != reservation.private_bank_bytes:
                raise RuntimeError('private shadow tensor bytes differ from reservation')
            peak = int(torch.npu.max_memory_allocated(device))
            measured_peak = max(0, peak - baseline)
            if measured_peak > reservation.reserve_bytes:
                raise RuntimeError('private shadow peak allocation exceeded reserved bytes')

            self.shadow = shadow
            self.method = method
            self.adapter = adapter
            self.device = device
            self._torch = torch
            self.measured_preallocation_peak_bytes = measured_peak
            self._StandardDispatchOutput = __import__(
                'sglang.srt.layers.moe.token_dispatcher',
                fromlist=['StandardDispatchOutput'],
            ).StandardDispatchOutput
            self._StandardTopKOutput = __import__(
                'sglang.srt.layers.moe.topk', fromlist=['StandardTopKOutput'],
            ).StandardTopKOutput
            self.state = 'READY'
        except BaseException as primary:
            if self._slot_reuse is not None:
                self._slot_reuse.invalidate()
            self.state = 'BROKEN_RELOAD_REQUIRED'
            self.shadow = self.method = self.adapter = None
            adapter = raw_adapter = method_map = layer_map = method = shadow = source = None
            gc.collect()
            if device is not None:
                try:
                    torch.npu.empty_cache()
                except BaseException:
                    pass
            raise primary

    def execute_group(self, layer, group, hidden_states, topk_weights, identity):
        if self.state != 'READY':
            raise RuntimeError(self.state)
        if self.adapter is None or self._current is not None:
            raise RuntimeError('private shadow bank is unavailable or still leased')
        if identity.placement_epoch < 0 or len(group.logical_experts) > self.capacity:
            raise RuntimeError('invalid shadow-bank group identity')
        self._current = identity
        self._synchronized = None
        try:
            moves = tuple(enumerate(group.logical_experts))
            remapped_ids = group.remapped_ids
            if self._slot_reuse is not None:
                plan = self._slot_reuse.prepare(
                    layer, group.logical_experts, group.remapped_ids, identity
                )
                moves, remapped_ids = plan.moves, plan.remapped_ids
            self.adapter.copy_experts(layer, moves)
            torch = self._torch
            rows = torch.tensor(group.rows, dtype=torch.int64, device=self.device)
            hidden = hidden_states.index_select(0, rows)
            weights = topk_weights.index_select(0, rows)
            remapped = torch.tensor(
                remapped_ids, dtype=torch.int32, device=self.device
            )
            dispatch = self._StandardDispatchOutput(
                hidden, None, self._StandardTopKOutput(weights, remapped, None)
            )
            result = self.method.apply(self.shadow, dispatch).hidden_states
            return tuple(result[index] for index in range(result.shape[0]))
        except BaseException:
            if self._slot_reuse is not None:
                self._slot_reuse.invalidate()
            raise

    def synchronize(self, identity):
        if self._current != identity:
            if self._slot_reuse is not None:
                self._slot_reuse.invalidate()
            raise RuntimeError('stale private shadow allocation')
        try:
            self._torch.npu.synchronize(self.device)
            if self._slot_reuse is not None:
                self._slot_reuse.publish_if_pending(identity)
        except BaseException:
            if self._slot_reuse is not None:
                self._slot_reuse.invalidate()
            self.state = 'BROKEN_RELOAD_REQUIRED'
            raise
        self._synchronized = identity

    def release(self, identity):
        if self._current != identity or self._synchronized != identity:
            if self._slot_reuse is not None:
                self._slot_reuse.invalidate()
            raise RuntimeError('stale private shadow release')
        self._current = None
        self._synchronized = None

    def assemble(self, rows, hidden_states):
        return self._torch.stack(rows, dim=0).to(dtype=hidden_states.dtype)

    def abort(self, request_id):
        if self._slot_reuse is not None:
            self._slot_reuse.invalidate()
        if self.state != 'READY':
            raise RuntimeError(self.state)
        if self._current is not None:
            try:
                self._torch.npu.synchronize(self.device)
            except BaseException:
                self.state = 'BROKEN_RELOAD_REQUIRED'
                raise
            self._current = None
            self._synchronized = None

    def fail_stop(self, error):
        if self._slot_reuse is not None:
            self._slot_reuse.invalidate()
        self.state = 'BROKEN_RELOAD_REQUIRED'
