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
