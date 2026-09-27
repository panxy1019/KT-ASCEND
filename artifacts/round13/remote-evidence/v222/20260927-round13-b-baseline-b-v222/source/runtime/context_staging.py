"""CPU-only compact selected-row staging for the QSA reference consumer.

The helper is deliberately synchronous and side-effect free. It reads immutable
host records or a caller-owned resident page, validates every requested page,
and returns one complete compact artifact. It does not install a runtime hook,
move tensors to an accelerator, alter a page table, or release HBM.
"""
from dataclasses import dataclass
import importlib
import inspect

from .context_profile import ContextStateIdentity
from . import context_page_store as _bf16
from . import kv_quant_reference as _fp8


SCOPE = 'CPU_SELECTED_ROW_STAGING_REFERENCE_ONLY'
RUNTIME_INTEGRATION_STATUS = 'NOT_INSTALLED'
HBM_RELEASE_STATUS = 'NO_HBM_RELEASE'
MAX_SELECTED_TOKENS = 2051
BF16_LAYER_PAGE_BYTES = (
    _bf16.PAGE_SIZE * 2 * _bf16.KV_HEADS * _bf16.HEAD_DIM * _bf16.BF16_BYTES)
MAX_FINAL_STAGING_BYTES = (
    MAX_SELECTED_TOKENS * 2 * _bf16.KV_HEADS
    * _bf16.HEAD_DIM * _bf16.BF16_BYTES)


@dataclass(frozen=True, slots=True)
class ResidentBF16Page:
    """Identity-stamped result returned synchronously by a resident reader."""

    key: _bf16.PageKey
    layer_id: int
    keys: object
    values: object


@dataclass(frozen=True, slots=True)
class CompactSelectedRows:
    """Complete compact tensors and indices for one sparse_gqa query row."""

    identity: ContextStateIdentity
    layer_id: int
    current_logical_position: int
    logical_tokens: tuple
    source_pages: tuple
    keys: object
    values: object
    remapped: object
    scope: str = SCOPE

    @property
    def staging_bytes(self):
        return (self.keys.numel() * self.keys.element_size()
                + self.values.numel() * self.values.element_size())


def _torch():
    try:
        return importlib.import_module('torch')
    except ModuleNotFoundError as error:
        raise RuntimeError('CPU selected-row staging requires PyTorch') from error


def _identity(identity):
    if (not isinstance(identity, ContextStateIdentity)
            or identity.component != 'QSA_KV'):
        raise ValueError('staging requires a QSA_KV ContextStateIdentity')
    if identity.prefix_id is not None:
        raise ValueError('prefix-shared staging is not supported')
    return identity


def _layer(layer_id):
    if type(layer_id) is not int or layer_id not in _bf16.QSA_LAYER_IDS:
        raise ValueError('layer is not an actual QSA layer')
    return layer_id


def _position(position):
    maximum = _bf16.MAX_LOGICAL_PAGES * _bf16.PAGE_SIZE
    if type(position) is not int or not 0 <= position < maximum:
        raise ValueError('current logical position is outside the bounded plan')
    return position


def _selection(selected, current_position):
    torch = _torch()
    if (not isinstance(selected, torch.Tensor)
            or selected.device.type != 'cpu'
            or selected.dtype not in (torch.int32, torch.int64)
            or selected.ndim != 2 or selected.shape[0] != 1
            or not 1 <= selected.shape[1] <= MAX_SELECTED_TOKENS):
        raise ValueError('selected must be one bounded CPU int32/int64 query row')
    logical = selected.detach().contiguous().tolist()[0]
    padding = False
    for token in logical:
        if token == -1:
            padding = True
        elif token < -1:
            raise ValueError('selected contains an invalid negative token')
        elif padding:
            raise ValueError('selected tokens must precede -1 padding')
        elif token > current_position:
            raise ValueError('selected token violates the causal bound')
    return logical


def _owned_bf16(payload, shape):
    torch = _torch()
    raw = torch.empty(len(payload), dtype=torch.uint8, device='cpu')
    memoryview(raw.numpy()).cast('B')[:] = payload
    return raw.view(torch.bfloat16).reshape(shape)


def _validate_page_tensors(keys, values, source, *, copy):
    torch = _torch()
    if (not isinstance(keys, torch.Tensor)
            or not isinstance(values, torch.Tensor)):
        raise TypeError(f'{source} K/V must be torch tensors')
    if (keys.device.type != 'cpu' or values.device.type != 'cpu'
            or keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16):
        raise ValueError(f'{source} K/V must be CPU BF16')
    shape = tuple(keys.shape)
    if (shape != tuple(values.shape) or len(shape) != 3
            or not 1 <= shape[0] <= _bf16.PAGE_SIZE
            or shape[1:] != (_bf16.KV_HEADS, _bf16.HEAD_DIM)):
        raise ValueError(f'{source} K/V has invalid fixed geometry')
    keys = keys.detach().contiguous()
    values = values.detach().contiguous()
    if copy:
        keys = keys.clone()
        values = values.clone()
    if (not bool(torch.isfinite(keys).all().item())
            or not bool(torch.isfinite(values).all().item())):
        raise ValueError(f'{source} K/V must be finite')
    return keys, values


def _validate_resident(result, expected_key, layer_id):
    if inspect.isawaitable(result):
        close = getattr(result, 'close', None)
        if callable(close):
            close()
        raise TypeError('resident_reader must complete synchronously')
    if not isinstance(result, ResidentBF16Page):
        raise TypeError('resident_reader must return ResidentBF16Page synchronously')
    if result.key != expected_key or result.layer_id != layer_id:
        raise ValueError('resident page identity is stale or mismatched')
    return _validate_page_tensors(
        result.keys, result.values, 'resident page', copy=True)


def _read_page(identity, layer_id, page, record_reader, resident_reader):
    key = _bf16.PageKey(identity.request_id, identity.allocation_epoch, page)
    record = record_reader(key, layer_id) if record_reader is not None else None
    if inspect.isawaitable(record):
        close = getattr(record, 'close', None)
        if callable(close):
            close()
        raise TypeError('record_reader must complete synchronously')
    if isinstance(record, _bf16.HostPageRecord):
        _bf16._validate_record(record)
        if record.key != key:
            raise ValueError('BF16 host page identity is stale or mismatched')
        payload = record.layer(layer_id)
        keys = _owned_bf16(payload.k_bytes, payload.shape)
        values = _owned_bf16(payload.v_bytes, payload.shape)
        return _validate_page_tensors(
            keys, values, 'BF16 host page', copy=False)
    if isinstance(record, _fp8.KVPageRecord):
        keys, values = _fp8.decode_kv_page(
            record, identity, logical_page=page, layer_id=layer_id)
        return _validate_page_tensors(
            keys, values, 'FP8 host page', copy=False)
    if record is not None:
        raise TypeError('record_reader returned an unsupported page record')
    if resident_reader is None:
        raise ValueError('selected logical page is unavailable')
    return _validate_resident(resident_reader(key, layer_id), key, layer_id)


def stage_selected_rows(
        identity, layer_id, selected, *, current_logical_position,
        record_reader=None, resident_reader=None):
    """Build one complete compact CPU BF16 input for ``sparse_gqa``.

    ``record_reader(PageKey, layer_id)`` and
    ``resident_reader(PageKey, layer_id)`` are synchronous callbacks. A record
    reader may return a validated BF16
    ``HostPageRecord``, a layer-specific FP8 ``KVPageRecord``, or ``None`` to
    request the resident fallback. A resident reader must return an
    identity-stamped ``ResidentBF16Page`` before the callback returns. Readers
    are read-only: they must not publish mappings, release pages, or start
    asynchronous work. The helper immediately snapshots resident tensors.

    The final K/V tensors are allocated once. Each source page is read,
    validated, copied to its precomputed destination rows, and released before
    the next page is read. No result object is created until every page has
    succeeded. Unique logical tokens become compact rows in their
    first-occurrence order. Duplicate selections reuse that row and ``-1``
    padding remains in its original columns.

    At the TP1 bound, the returned K/V tensors contain at most 4,200,448 bytes.
    This helper additionally owns at most one 8,192-byte decoded BF16 layer
    page at a time. Caller-owned records and resident tensors, Python objects,
    and FP8 codec conversion temporaries are separate. These formulas are
    allocation bounds, not a measured process-peak claim.
    """
    identity = _identity(identity)
    layer_id = _layer(layer_id)
    current_position = _position(current_logical_position)
    if record_reader is not None and not callable(record_reader):
        raise TypeError('record_reader must be callable')
    if resident_reader is not None and not callable(resident_reader):
        raise TypeError('resident_reader must be callable')
    if record_reader is None and resident_reader is None:
        raise ValueError('at least one synchronous page reader is required')
    logical = _selection(selected, current_position)
    torch = _torch()

    token_to_row = {}
    page_plan = {}
    page_order = []
    for token in logical:
        if token < 0 or token in token_to_row:
            continue
        page, offset = divmod(token, _bf16.PAGE_SIZE)
        destination = len(token_to_row)
        token_to_row[token] = destination
        if page not in page_plan:
            page_plan[page] = []
            page_order.append(page)
        page_plan[page].append((offset, destination))

    compact_shape = (len(token_to_row), _bf16.KV_HEADS, _bf16.HEAD_DIM)
    compact_keys = torch.empty(
        compact_shape, dtype=torch.bfloat16, device='cpu')
    compact_values = torch.empty_like(compact_keys)
    for page in page_order:
        keys, values = _read_page(
            identity, layer_id, page, record_reader, resident_reader)
        copies = page_plan[page]
        if any(offset >= keys.shape[0] for offset, unused in copies):
            raise ValueError('selected token offset exceeds source page')
        for offset, destination in copies:
            compact_keys[destination].copy_(keys[offset])
            compact_values[destination].copy_(values[offset])
        del keys, values

    remapped_values = [token_to_row[token] if token >= 0 else -1
                       for token in logical]
    remapped = torch.tensor(
        [remapped_values], dtype=selected.dtype, device='cpu')
    logical_tokens = tuple(token_to_row)
    return CompactSelectedRows(
        identity=identity,
        layer_id=layer_id,
        current_logical_position=current_position,
        logical_tokens=logical_tokens,
        source_pages=tuple(_bf16.PageKey(
            identity.request_id, identity.allocation_epoch, page)
            for page in page_order),
        keys=compact_keys,
        values=compact_values,
        remapped=remapped,
    )


__all__ = (
    'ResidentBF16Page', 'CompactSelectedRows', 'stage_selected_rows',
    'SCOPE', 'RUNTIME_INTEGRATION_STATUS', 'HBM_RELEASE_STATUS',
    'MAX_SELECTED_TOKENS', 'MAX_FINAL_STAGING_BYTES',
    'BF16_LAYER_PAGE_BYTES',
)
