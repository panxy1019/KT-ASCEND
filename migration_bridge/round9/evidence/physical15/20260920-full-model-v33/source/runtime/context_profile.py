"""CPU-only F8 context plans; no native KV or long-context qualification.

The caller supplies model and backend geometry explicitly.  This module does
not inspect a running allocator, enable a cache implementation, or infer that a
native KV representation exists.
"""
from dataclasses import dataclass


_TARGETS = frozenset((4096, 16384, 65536, 131072))
_MAX_I64 = 2**63 - 1
_COMPONENTS = ('QSA_KV', 'QSA_INDEX', 'GDN', 'PLE')


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _product(*values):
    result = 1
    for value in values:
        if value and result > _MAX_I64 // value:
            raise OverflowError('context byte arithmetic exceeds signed64')
        result *= value
    return result


def _sum(*values):
    result = 0
    for value in values:
        if result > _MAX_I64 - value:
            raise OverflowError('context byte arithmetic exceeds signed64')
        result += value
    return result


@dataclass(frozen=True)
class ContextGeometry:
    model_layers: int
    qsa_layers: int
    qsa_kv_heads: int
    qsa_head_dim: int
    qsa_kv_element_bytes: int
    qsa_index_heads: int
    qsa_index_head_dim: int
    qsa_index_element_bytes: int
    qsa_compress_ratio: int
    page_size: int
    request_slots: int
    reserved_token_slots: int
    page_metadata_bytes: int
    position_element_bytes: int
    gdn_bytes_per_request: int
    ple_bytes_per_request: int

    def __post_init__(self):
        for name in ('model_layers', 'qsa_layers', 'qsa_kv_heads', 'qsa_head_dim',
                     'qsa_kv_element_bytes',
                     'qsa_index_heads', 'qsa_index_head_dim', 'qsa_compress_ratio',
                     'qsa_index_element_bytes',
                     'page_size', 'request_slots', 'position_element_bytes',
                     'gdn_bytes_per_request', 'ple_bytes_per_request'):
            _integer(getattr(self, name), name)
        for name in ('reserved_token_slots', 'page_metadata_bytes'):
            _integer(getattr(self, name), name, minimum=0)
        if self.qsa_layers > self.model_layers:
            raise ValueError('QSA layers exceed model layers')
        # The current reference consumer is explicitly BF16, one index-KV head,
        # and ratio4.  Other formats require their own implementation and gate.
        if (self.qsa_kv_element_bytes != 2 or
                self.qsa_index_element_bytes != 2 or
                self.qsa_index_heads != 1 or self.qsa_compress_ratio != 4):
            raise ValueError('geometry differs from the current BF16 QSA consumer')
        if (self.page_size % self.qsa_compress_ratio or
                self.reserved_token_slots % self.page_size):
            raise ValueError('pages and reserved slots must align to QSA groups')


@dataclass(frozen=True)
class ContextPlan:
    context_tokens: int
    token_capacity: int
    pages: int
    compressed_slots: int
    qsa_kv_bytes: int
    qsa_index_bytes: int
    qsa_pending_bytes: int
    gdn_bytes: int
    ple_bytes: int
    page_metadata_bytes: int
    total_bytes: int
    page_size: int
    compress_ratio: int
    native_kv_qualified: bool = False
    scope: str = 'CPU_CONTEXT_PLAN_ONLY'


def derive_context_plan(context_tokens, geometry, *, budget_limit_bytes=None):
    """Derive exact component bytes for one explicit supported target."""
    if type(context_tokens) is not int or context_tokens not in _TARGETS:
        raise ValueError('context target must be exactly4k/16k/64k/128k')
    if not isinstance(geometry, ContextGeometry):
        raise TypeError('explicit ContextGeometry is required')
    if context_tokens % geometry.page_size or context_tokens % geometry.qsa_compress_ratio:
        raise ValueError('context target crosses a partial page or QSA group')
    token_capacity = _sum(context_tokens, geometry.reserved_token_slots)
    if token_capacity % geometry.page_size:
        raise ValueError('token capacity must end on a page boundary')
    pages = token_capacity // geometry.page_size
    qsa_kv = _product(token_capacity, geometry.qsa_layers, 2,
                      geometry.qsa_kv_heads, geometry.qsa_head_dim,
                      geometry.qsa_kv_element_bytes)
    groups = token_capacity // geometry.qsa_compress_ratio
    qsa_index = _product(groups, geometry.qsa_layers,
                         geometry.qsa_index_heads,
                         geometry.qsa_index_head_dim,
                         geometry.qsa_index_element_bytes)
    pending_per_request = _product(
        geometry.qsa_compress_ratio,
        _sum(_product(geometry.qsa_layers, geometry.qsa_index_heads,
                      geometry.qsa_index_head_dim,
                      geometry.qsa_index_element_bytes),
             geometry.position_element_bytes),
    )
    qsa_pending = _product(geometry.request_slots, pending_per_request)
    gdn = _product(geometry.request_slots, geometry.gdn_bytes_per_request)
    ple = _product(geometry.request_slots, geometry.ple_bytes_per_request)
    metadata = _product(pages, geometry.page_metadata_bytes)
    total = _sum(qsa_kv, qsa_index, qsa_pending, gdn, ple, metadata)
    if budget_limit_bytes is not None:
        _integer(budget_limit_bytes, 'budget_limit_bytes')
        if total > budget_limit_bytes:
            raise ValueError('context plan exceeds caller budget')
    return ContextPlan(context_tokens, token_capacity, pages, groups, qsa_kv,
                       qsa_index, qsa_pending, gdn, ple, metadata, total,
                       geometry.page_size, geometry.qsa_compress_ratio)


def token_page(plan, logical_token, *, include_reserved=False):
    if not isinstance(plan, ContextPlan):
        raise TypeError('ContextPlan is required')
    if type(logical_token) is not int or logical_token < 0:
        raise ValueError('logical token must be nonnegative exact int')
    limit = plan.token_capacity if include_reserved else plan.context_tokens
    if logical_token >= limit:
        raise ValueError('logical token exceeds planned capacity')
    return divmod(logical_token, plan.page_size)


def compressed_page(plan, logical_group_start):
    """Return the compressed-slot index for an aligned logical group start."""
    if not isinstance(plan, ContextPlan):
        raise TypeError('ContextPlan is required')
    if (type(logical_group_start) is not int or logical_group_start < 0 or
            logical_group_start >= plan.context_tokens or
            logical_group_start % plan.compress_ratio):
        raise ValueError('compressed group start is unaligned or outside context')
    return logical_group_start // plan.compress_ratio


@dataclass(frozen=True)
class ContextStateIdentity:
    request_id: str
    prefix_id: str | None
    allocation_epoch: int
    component: str

    def __post_init__(self):
        if (not isinstance(self.request_id, str) or not self.request_id or
                (self.prefix_id is not None and
                 (not isinstance(self.prefix_id, str) or not self.prefix_id)) or
                type(self.allocation_epoch) is not int or self.allocation_epoch < 0 or
                self.component not in _COMPONENTS):
            raise ValueError('invalid context state identity')


def context_identities(request_id, prefix_id, allocation_epoch):
    return {component: ContextStateIdentity(
                request_id, prefix_id, allocation_epoch, component)
            for component in _COMPONENTS}


def validate_context_identities(identities):
    if not isinstance(identities, dict) or set(identities) != set(_COMPONENTS):
        raise ValueError('all four context state identities are required')
    values = tuple(identities[name] for name in _COMPONENTS)
    if (any(not isinstance(value, ContextStateIdentity) or value.component != name
            for name, value in zip(_COMPONENTS, values)) or
            len(set(values)) != len(values) or
            len({(value.request_id, value.prefix_id, value.allocation_epoch)
                 for value in values}) != 1):
        raise ValueError('context state identities are aliased, stale or unrelated')
    return values


@dataclass(frozen=True)
class OwnedExtent:
    identity: ContextStateIdentity
    start: int
    stop: int
    unit: str


def qsa_kv_page_owner(plan, identities, page):
    validate_context_identities(identities)
    if type(page) is not int or not 0 <= page < plan.pages:
        raise ValueError('QSA KV page is outside planned ownership')
    start = page * plan.page_size
    return OwnedExtent(identities['QSA_KV'], start, start + plan.page_size,
                       'token_slot')


def qsa_index_owner(plan, identities, compressed_slot):
    validate_context_identities(identities)
    if (type(compressed_slot) is not int or
            not 0 <= compressed_slot < plan.compressed_slots):
        raise ValueError('QSA index slot is outside planned ownership')
    start = compressed_slot * plan.compress_ratio
    return OwnedExtent(identities['QSA_INDEX'], start,
                       start + plan.compress_ratio, 'logical_token')


@dataclass(frozen=True)
class Qwen4TP1Geometry:
    """Checkpoint-derived geometry for the admitted single-rank Round9 profile."""
    context_geometry: ContextGeometry
    qsa_layers: int
    gdn_layers: int
    gdn_conv_shape_per_layer: tuple[int, int]
    gdn_temporal_shape_per_layer: tuple[int, int, int]
    gdn_bytes_per_live_request: int
    ple_conv_shape_per_live_request: tuple[int, int, int]
    ple_history_shape_per_live_request: tuple[int]
    ple_bytes_per_live_request: int
    raw_request_slots: int
    recurrent_request_slots: int
    recurrent_storage_slots: int
    index_token_budget: int


def _require_fields(config, expected):
    missing = set(expected).difference(config)
    if missing:
        raise ValueError(f'Qwen4 config is missing fields: {sorted(missing)}')
    changed = {name: (config[name], value) for name, value in expected.items()
               if type(config[name]) is not type(value) or config[name] != value}
    if changed:
        raise ValueError(f'Qwen4 checkpoint geometry differs: {changed}')


def derive_qwen4_tp1_geometry(config, *, config_scope):
    """Derive the exact admitted TP1 geometry from an explicit config scope.

    ``config_scope`` prevents a nested full config from being silently confused
    with its text config.  The input has already crossed the caller's trust and
    identity boundary; this function performs no file I/O.
    """
    if not isinstance(config, dict):
        raise TypeError('Qwen4 config must be a dictionary')
    if config_scope == 'full_config':
        if set(config).issuperset({'text_config'}) and isinstance(config['text_config'], dict):
            text = config['text_config']
        else:
            raise ValueError('full_config scope requires a text_config dictionary')
    elif config_scope == 'text_config':
        if 'text_config' in config:
            raise ValueError('text_config scope rejects a nested full config')
        text = config
    else:
        raise ValueError('config_scope must be full_config or text_config')

    expected = {
        'model_type': 'qwen4_exp_text',
        'dtype': 'bfloat16',
        'num_hidden_layers': 48,
        'full_attention_interval': 4,
        'num_key_value_heads': 2,
        'head_dim': 256,
        'indexer_kv_heads': 1,
        'indexer_head_dim': 128,
        'indexer_compress_ratio': 4,
        'indexer_budget': 2048,
        'linear_num_value_heads': 48,
        'linear_value_head_dim': 128,
        'linear_num_key_heads': 16,
        'linear_key_head_dim': 128,
        'linear_conv_kernel_dim': 4,
        'mamba_ssm_dtype': 'float32',
        'hidden_size': 2560,
        'hc_count': 4,
        'ple_embed_dim': 2560,
        'ple_layer_ids': [2],
        'ple_conv_kernel_size': 4,
        'ngram_size': 3,
        'heads_per_ngram': 8,
    }
    _require_fields(text, expected)
    layer_types = text.get('layer_types')
    if (not isinstance(layer_types, list) or len(layer_types) != text['num_hidden_layers'] or
            any(layer_type != ('full_attention' if (layer + 1) %
                               text['full_attention_interval'] == 0 else 'linear_attention')
                for layer, layer_type in enumerate(layer_types))):
        raise ValueError('Qwen4 layer_types differ from the admitted 3GDN:1QSA pattern')
    qsa_layers = layer_types.count('full_attention')
    gdn_layers = layer_types.count('linear_attention')

    # Mamba2StateShape.create at TP1.
    value_width = _product(text['linear_num_value_heads'],
                           text['linear_value_head_dim'])
    conv_width = _sum(value_width, _product(2, text['linear_num_key_heads'],
                                            text['linear_key_head_dim']))
    conv_shape = (conv_width, text['linear_conv_kernel_dim'] - 1)
    temporal_shape = (text['linear_num_value_heads'],
                      text['linear_value_head_dim'], text['linear_key_head_dim'])
    gdn_bytes = _product(gdn_layers, conv_shape[0], conv_shape[1], 2)
    gdn_bytes = _sum(gdn_bytes, _product(gdn_layers, *temporal_shape, 4))

    ple_channels = _product(text['hidden_size'], text['hc_count'])
    ple_state_len = _product(text['ple_conv_kernel_size'] - 1, text['ngram_size'])
    ple_layers = len(text['ple_layer_ids'])
    ple_conv_shape = (ple_layers, ple_channels, ple_state_len)
    ple_history_shape = (text['ngram_size'] - 1,)
    ple_bytes = _sum(_product(*ple_conv_shape, 2),
                     _product(*ple_history_shape, 8))

    raw_request_slots = 1
    recurrent_request_slots = 1
    recurrent_storage_slots = recurrent_request_slots + 1  # padded slot zero
    context = ContextGeometry(
        model_layers=text['num_hidden_layers'], qsa_layers=qsa_layers,
        qsa_kv_heads=text['num_key_value_heads'], qsa_head_dim=text['head_dim'],
        qsa_kv_element_bytes=2, qsa_index_heads=text['indexer_kv_heads'],
        qsa_index_head_dim=text['indexer_head_dim'], qsa_index_element_bytes=2,
        qsa_compress_ratio=text['indexer_compress_ratio'], page_size=4,
        request_slots=raw_request_slots, reserved_token_slots=4,
        page_metadata_bytes=16, position_element_bytes=8,
        gdn_bytes_per_request=gdn_bytes, ple_bytes_per_request=ple_bytes)
    return Qwen4TP1Geometry(
        context, qsa_layers, gdn_layers, conv_shape, temporal_shape, gdn_bytes,
        ple_conv_shape, ple_history_shape, ple_bytes, raw_request_slots,
        recurrent_request_slots, recurrent_storage_slots, text['indexer_budget'])


_DTYPE_BYTES = {
    'torch.bfloat16': 2,
    'torch.float32': 4,
    'torch.int64': 8,
}


@dataclass(frozen=True)
class TensorExpectation:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or
                not isinstance(self.shape, tuple) or not self.shape or
                any(type(value) is not int or value < 1 for value in self.shape) or
                self.dtype not in _DTYPE_BYTES):
            raise ValueError('invalid expected tensor')

    @property
    def nbytes(self):
        return _product(*self.shape, _DTYPE_BYTES[self.dtype])


@dataclass(frozen=True)
class Qwen4TP1Footprint:
    plan: ContextPlan
    logical_token_capacity: int
    logical_pages: int
    physical_padding_pages: int
    physical_pages: int
    physical_token_slots: int
    raw_request_slots: int
    recurrent_request_slots: int
    recurrent_storage_slots: int
    expected_tensors: tuple[TensorExpectation, ...]
    physical_floor_bytes: int
    scope: str = 'CPU_DERIVED_TP1_PHYSICAL_FLOOR_NOT_DEVICE_MEASURED'


def derive_qwen4_tp1_footprint(context_tokens, geometry):
    if not isinstance(geometry, Qwen4TP1Geometry):
        raise TypeError('Qwen4TP1Geometry is required')
    plan = derive_context_plan(context_tokens, geometry.context_geometry)
    page = plan.page_size
    padding_pages = 1
    physical_pages = _sum(plan.pages, padding_pages)
    physical_tokens = _product(physical_pages, page)
    physical_groups = (plan.token_capacity + page) // plan.compress_ratio
    raw_width = _product(geometry.raw_request_slots, plan.compress_ratio)
    tensors = (
        TensorExpectation('ordinary_kv',
            (2, geometry.qsa_layers, physical_pages, page,
             geometry.context_geometry.qsa_kv_heads,
             geometry.context_geometry.qsa_head_dim), 'torch.bfloat16'),
        TensorExpectation('qsa_compressed',
            (geometry.qsa_layers, physical_groups,
             geometry.context_geometry.qsa_index_heads,
             geometry.context_geometry.qsa_index_head_dim), 'torch.bfloat16'),
        TensorExpectation('qsa_raw',
            (geometry.qsa_layers, raw_width,
             geometry.context_geometry.qsa_index_heads,
             geometry.context_geometry.qsa_index_head_dim), 'torch.bfloat16'),
        TensorExpectation('qsa_positions', (raw_width,), 'torch.int64'),
        TensorExpectation('gdn_conv',
            (geometry.gdn_layers, geometry.recurrent_storage_slots,
             *geometry.gdn_conv_shape_per_layer), 'torch.bfloat16'),
        TensorExpectation('gdn_temporal',
            (geometry.gdn_layers, geometry.recurrent_storage_slots,
             *geometry.gdn_temporal_shape_per_layer), 'torch.float32'),
        TensorExpectation('ple_conv',
            (geometry.ple_conv_shape_per_live_request[0],
             geometry.recurrent_storage_slots,
             *geometry.ple_conv_shape_per_live_request[1:]), 'torch.bfloat16'),
        TensorExpectation('ple_history',
            (geometry.recurrent_storage_slots,
             *geometry.ple_history_shape_per_live_request), 'torch.int64'),
    )
    floor = _sum(*(tensor.nbytes for tensor in tensors))
    return Qwen4TP1Footprint(
        plan, plan.token_capacity, plan.pages, padding_pages, physical_pages,
        physical_tokens, geometry.raw_request_slots,
        geometry.recurrent_request_slots, geometry.recurrent_storage_slots,
        tensors, floor)


@dataclass(frozen=True)
class TensorMeasurement:
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def __post_init__(self):
        if (not isinstance(self.shape, tuple) or not self.shape or
                any(type(value) is not int or value < 1 for value in self.shape) or
                not isinstance(self.dtype, str) or not self.dtype or
                type(self.nbytes) is not int or self.nbytes < 1):
            raise ValueError('invalid tensor measurement')


@dataclass(frozen=True)
class MeasurementSummary:
    total_bytes: int
    tensor_count: int
    scope: str = 'EXACT_SHAPES_VALIDATED_CALLER_SUPPLIED_MEASUREMENTS'


def validate_qwen4_tp1_measurements(footprint, measurements):
    """Validate caller-recorded tensor facts without importing a device runtime."""
    if not isinstance(footprint, Qwen4TP1Footprint):
        raise TypeError('Qwen4TP1Footprint is required')
    if not isinstance(measurements, dict):
        raise TypeError('measurements must be a dictionary')
    expected = {tensor.name: tensor for tensor in footprint.expected_tensors}
    if set(measurements) != set(expected):
        raise ValueError('measured tensor names differ from expected physical footprint')
    total = 0
    for name, tensor in expected.items():
        measured = measurements[name]
        if not isinstance(measured, TensorMeasurement):
            raise TypeError(f'{name} must be a TensorMeasurement')
        if ((measured.shape, measured.dtype, measured.nbytes) !=
                (tensor.shape, tensor.dtype, tensor.nbytes)):
            raise ValueError(f'{name} measurement differs from expected physical tensor')
        total = _sum(total, measured.nbytes)
    if total != footprint.physical_floor_bytes:
        raise ValueError('measured tensor total differs from physical floor')
    return MeasurementSummary(total, len(expected))
