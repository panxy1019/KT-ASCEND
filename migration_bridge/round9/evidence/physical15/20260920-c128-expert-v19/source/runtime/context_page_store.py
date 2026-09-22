"""CPU-only request-local ownership for canonical QSA K/V page artifacts.

This module does not call an allocator, free HBM, allocate staging tensors, or
install a model/backend hook. Leases only protect immutable host artifacts.
"""
from dataclasses import dataclass
import hashlib
import hmac
import json
import threading


SCOPE = 'CPU_REQUEST_LOCAL_OWNERSHIP_ONLY'
STATE = 'HOST_PUBLISHED_NO_HBM_ACTION'
FORMAT_VERSION = 1
PAGE_SIZE = 4
KV_HEADS = 2
HEAD_DIM = 256
BF16_BYTES = 2
MAX_LOGICAL_PAGES = 32769
QSA_LAYER_IDS = tuple(range(3, 48, 4))
_QSA_LAYER_SET = frozenset(QSA_LAYER_IDS)


def _exact_nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative exact integer')
    return value


def _request(request_id):
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('request_id must be a nonempty string')
    return request_id


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode('ascii')


@dataclass(frozen=True, slots=True)
class PageKey:
    request_id: str
    allocation_epoch: int
    logical_page: int

    def __post_init__(self):
        _request(self.request_id)
        _exact_nonnegative(self.allocation_epoch, 'allocation_epoch')
        if (type(self.logical_page) is not int
                or not 0 <= self.logical_page < MAX_LOGICAL_PAGES):
            raise ValueError('logical_page is outside the bounded 128K plan')


@dataclass(frozen=True, slots=True)
class BF16LayerPayload:
    layer_id: int
    valid_tokens: int
    shape: tuple
    k_bytes: bytes
    v_bytes: bytes
    metadata_bytes: bytes
    digest_sha256: str
    mode: str = 'BF16_EXACT'

    @property
    def artifact_bytes(self):
        return (len(self.metadata_bytes) + len(self.k_bytes) + len(self.v_bytes)
                + 32)


@dataclass(frozen=True, slots=True)
class HostPageRecord:
    key: PageKey
    page_generation: int
    layers: tuple
    metadata_bytes: bytes
    digest_sha256: str
    host_bytes: int
    scope: str = SCOPE
    state: str = STATE
    format_version: int = FORMAT_VERSION

    def layer(self, layer_id):
        if type(layer_id) is not int or layer_id not in _QSA_LAYER_SET:
            raise ValueError('invalid QSA layer')
        return self.layers[QSA_LAYER_IDS.index(layer_id)]


@dataclass(frozen=True, slots=True)
class PageLease:
    key: PageKey
    page_generation: int
    lease_id: int
    kind: str


@dataclass(frozen=True, slots=True)
class ReloadTicket:
    key: PageKey
    page_generation: int
    ticket_id: int


@dataclass(slots=True)
class _Entry:
    record: HostPageRecord
    invalidating: bool = False


def _layer_metadata(layer_id, valid_tokens):
    return {
        'format_version': FORMAT_VERSION,
        'head_dim': HEAD_DIM,
        'kv_heads': KV_HEADS,
        'layer_id': layer_id,
        'mode': 'BF16_EXACT',
        'page_size': PAGE_SIZE,
        'shape': [valid_tokens, KV_HEADS, HEAD_DIM],
        'source_dtype': 'bfloat16',
        'valid_tokens': valid_tokens,
    }


def _layer_digest(metadata, k_bytes, v_bytes):
    digest = hashlib.sha256()
    for payload in (metadata, k_bytes, v_bytes):
        digest.update(len(payload).to_bytes(8, 'little'))
        digest.update(payload)
    return digest.hexdigest()


def make_bf16_layer_payload(layer_id, valid_tokens, k_bytes, v_bytes):
    """Build an exact-byte BF16 layer artifact; no numeric conversion occurs."""
    if type(layer_id) is not int or layer_id not in _QSA_LAYER_SET:
        raise ValueError('layer is not an actual QSA layer')
    if type(valid_tokens) is not int or not 1 <= valid_tokens <= PAGE_SIZE:
        raise ValueError('valid_tokens must be in the bounded page')
    expected = valid_tokens * KV_HEADS * HEAD_DIM * BF16_BYTES
    if (type(k_bytes) is not bytes or type(v_bytes) is not bytes
            or len(k_bytes) != expected or len(v_bytes) != expected):
        raise ValueError('BF16 K/V payload length differs from fixed geometry')
    metadata = _json_bytes(_layer_metadata(layer_id, valid_tokens))
    return BF16LayerPayload(
        layer_id=layer_id, valid_tokens=valid_tokens,
        shape=(valid_tokens, KV_HEADS, HEAD_DIM),
        k_bytes=k_bytes, v_bytes=v_bytes, metadata_bytes=metadata,
        digest_sha256=_layer_digest(metadata, k_bytes, v_bytes),
    )


def _validate_layer(layer):
    if not isinstance(layer, BF16LayerPayload):
        raise TypeError('BF16LayerPayload is required')
    if type(layer.layer_id) is not int or layer.layer_id not in _QSA_LAYER_SET:
        raise ValueError('invalid QSA layer payload identity')
    if (type(layer.valid_tokens) is not int
            or not 1 <= layer.valid_tokens <= PAGE_SIZE
            or layer.shape != (layer.valid_tokens, KV_HEADS, HEAD_DIM)):
        raise ValueError('invalid BF16 layer shape')
    if layer.mode != 'BF16_EXACT':
        raise ValueError('unsupported layer artifact mode')
    expected = layer.valid_tokens * KV_HEADS * HEAD_DIM * BF16_BYTES
    if (type(layer.k_bytes) is not bytes or type(layer.v_bytes) is not bytes
            or len(layer.k_bytes) != expected or len(layer.v_bytes) != expected):
        raise ValueError('invalid BF16 layer payload length')
    metadata = _json_bytes(_layer_metadata(layer.layer_id, layer.valid_tokens))
    if type(layer.metadata_bytes) is not bytes or layer.metadata_bytes != metadata:
        raise ValueError('noncanonical BF16 layer metadata')
    digest = _layer_digest(metadata, layer.k_bytes, layer.v_bytes)
    if (not isinstance(layer.digest_sha256, str)
            or not hmac.compare_digest(layer.digest_sha256, digest)):
        raise ValueError('BF16 layer digest mismatch')
    return layer


def _page_metadata(key, generation, layer_digests):
    return {
        'allocation_epoch': key.allocation_epoch,
        'format_version': FORMAT_VERSION,
        'layer_digests': list(layer_digests),
        'logical_page': key.logical_page,
        'page_generation': generation,
        'request_id': key.request_id,
        'scope': SCOPE,
        'state': STATE,
    }


def _page_digest(metadata, layers):
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, 'little'))
    digest.update(metadata)
    for layer in layers:
        digest.update(bytes.fromhex(layer.digest_sha256))
    return digest.hexdigest()


def _validate_record(record):
    if not isinstance(record, HostPageRecord):
        raise TypeError('HostPageRecord is required')
    if (record.scope != SCOPE or record.state != STATE
            or record.format_version != FORMAT_VERSION):
        raise ValueError('unsupported page-record scope, state or version')
    if not isinstance(record.key, PageKey):
        raise ValueError('invalid page key')
    _exact_nonnegative(record.page_generation, 'page_generation')
    if (not isinstance(record.layers, tuple) or len(record.layers) != 12
            or tuple(layer.layer_id for layer in record.layers) != QSA_LAYER_IDS):
        raise ValueError('page record requires the ordered 12-layer payload set')
    for layer in record.layers:
        _validate_layer(layer)
        if layer.valid_tokens != PAGE_SIZE:
            raise ValueError('only completed full pages may publish for eviction')
    metadata = _json_bytes(_page_metadata(
        record.key, record.page_generation,
        tuple(layer.digest_sha256 for layer in record.layers)))
    if type(record.metadata_bytes) is not bytes or record.metadata_bytes != metadata:
        raise ValueError('noncanonical page metadata')
    digest = _page_digest(metadata, record.layers)
    if (not isinstance(record.digest_sha256, str)
            or not hmac.compare_digest(record.digest_sha256, digest)):
        raise ValueError('page digest mismatch')
    host_bytes = (len(metadata) + 32
                  + sum(layer.artifact_bytes for layer in record.layers))
    if type(record.host_bytes) is not int or record.host_bytes != host_bytes:
        raise ValueError('page host-byte accounting mismatch')
    return record


class RequestLocalPageStore:
    """Bounded immutable host records with request-local lease ownership."""

    def __init__(self, *, max_pages, max_host_bytes):
        if type(max_pages) is not int or max_pages <= 0:
            raise ValueError('max_pages must be a positive exact integer')
        if type(max_host_bytes) is not int or max_host_bytes <= 0:
            raise ValueError('max_host_bytes must be a positive exact integer')
        self.max_pages = max_pages
        self.max_host_bytes = max_host_bytes
        self._entries = {}
        self._leases = {}
        self._reloads = {}
        self._next_generation = {}
        self._next_token = 0
        self._used_bytes = 0
        self._lock = threading.RLock()

    def publish_export(self, key, layer_payloads):
        """Atomically publish one full page after every layer and digest validates."""
        if not isinstance(key, PageKey):
            raise TypeError('PageKey is required')
        if not isinstance(layer_payloads, dict) or set(layer_payloads) != _QSA_LAYER_SET:
            raise ValueError('export requires exactly all 12 QSA layer payloads')
        layers = tuple(layer_payloads[layer_id] for layer_id in QSA_LAYER_IDS)
        for expected_layer, layer in zip(QSA_LAYER_IDS, layers):
            _validate_layer(layer)
            if layer.layer_id != expected_layer:
                raise ValueError('layer payload is stored under the wrong identity')
            if layer.valid_tokens != PAGE_SIZE:
                raise ValueError('only a completed full page may publish for eviction')
        with self._lock:
            if key in self._entries:
                raise ValueError('page is already published or draining')
            generation = self._next_generation.get(key, 0)
            metadata = _json_bytes(_page_metadata(
                key, generation, tuple(layer.digest_sha256 for layer in layers)))
            host_bytes = (len(metadata) + 32
                          + sum(layer.artifact_bytes for layer in layers))
            record = HostPageRecord(
                key=key, page_generation=generation, layers=layers,
                metadata_bytes=metadata,
                digest_sha256=_page_digest(metadata, layers),
                host_bytes=host_bytes,
            )
            _validate_record(record)
            if (len(self._entries) + 1 > self.max_pages
                    or self._used_bytes + host_bytes > self.max_host_bytes):
                raise RuntimeError('request-local host page capacity exhausted')
            self._entries[key] = _Entry(record)
            self._next_generation[key] = generation + 1
            self._used_bytes += host_bytes
            return record

    def pin(self, key):
        with self._lock:
            entry = self._entry(key)
            if entry.invalidating:
                raise RuntimeError('page is invalidating')
            return self._new_lease(entry, 'HOST_PIN')

    def read(self, lease):
        with self._lock:
            stored = self._leases.get(getattr(lease, 'lease_id', None))
            if stored is None or stored is not lease:
                raise ValueError('stale or copied page lease')
            entry = self._entries.get(lease.key)
            if entry is None or entry.record.page_generation != lease.page_generation:
                raise ValueError('stale page lease generation')
            return _validate_record(entry.record)

    def release(self, lease):
        with self._lock:
            stored = self._leases.get(getattr(lease, 'lease_id', None))
            if stored is None or stored is not lease:
                raise ValueError('stale or copied page lease')
            del self._leases[lease.lease_id]
            self._maybe_remove(lease.key)

    def begin_reload(self, key):
        with self._lock:
            entry = self._entry(key)
            if entry.invalidating:
                raise RuntimeError('page is invalidating')
            ticket = ReloadTicket(
                key, entry.record.page_generation, self._take_token())
            self._reloads[ticket.ticket_id] = ticket
            return ticket

    def complete_reload(self, ticket):
        """Turn an exact callback into a staging-reference lease, without NPU IO."""
        with self._lock:
            stored = self._reloads.get(getattr(ticket, 'ticket_id', None))
            if stored is None or stored is not ticket:
                raise ValueError('stale or copied reload ticket')
            del self._reloads[ticket.ticket_id]
            entry = self._entries.get(ticket.key)
            if (entry is None
                    or entry.record.page_generation != ticket.page_generation):
                self._maybe_remove(ticket.key)
                raise ValueError('stale reload callback generation')
            if entry.invalidating:
                self._maybe_remove(ticket.key)
                raise RuntimeError('reload callback belongs to an invalidated page')
            return self._new_lease(entry, 'CPU_STAGING_REFERENCE_ONLY')

    def abort_reload(self, ticket):
        with self._lock:
            stored = self._reloads.get(getattr(ticket, 'ticket_id', None))
            if stored is None or stored is not ticket:
                raise ValueError('stale or copied reload ticket')
            del self._reloads[ticket.ticket_id]
            self._maybe_remove(ticket.key)

    def invalidate(self, request_id, allocation_epoch):
        """Invalidate one exact request allocation; defer removal until drain."""
        _request(request_id)
        _exact_nonnegative(allocation_epoch, 'allocation_epoch')
        with self._lock:
            keys = tuple(key for key in self._entries
                         if key.request_id == request_id
                         and key.allocation_epoch == allocation_epoch)
            for key in keys:
                self._entries[key].invalidating = True
            for key in keys:
                self._maybe_remove(key)
            return not any(key in self._entries for key in keys)

    def snapshot(self):
        with self._lock:
            return {
                'scope': SCOPE,
                'state': STATE,
                'pages': len(self._entries),
                'host_bytes': self._used_bytes,
                'active_leases': len(self._leases),
                'inflight_reloads': len(self._reloads),
                'invalidating': sum(entry.invalidating
                                    for entry in self._entries.values()),
                'allocator_free_calls': 0,
                'hbm_bytes_freed_claimed': 0,
            }

    def _entry(self, key):
        if not isinstance(key, PageKey):
            raise TypeError('PageKey is required')
        entry = self._entries.get(key)
        if entry is None:
            raise ValueError('unknown request-local page')
        _validate_record(entry.record)
        return entry

    def _new_lease(self, entry, kind):
        lease = PageLease(
            entry.record.key, entry.record.page_generation,
            self._take_token(), kind)
        self._leases[lease.lease_id] = lease
        return lease

    def _take_token(self):
        token = self._next_token
        self._next_token += 1
        return token

    def _maybe_remove(self, key):
        entry = self._entries.get(key)
        if entry is None or not entry.invalidating:
            return
        generation = entry.record.page_generation
        active = any(lease.key == key and lease.page_generation == generation
                     for lease in self._leases.values())
        inflight = any(ticket.key == key and ticket.page_generation == generation
                       for ticket in self._reloads.values())
        if not active and not inflight:
            self._used_bytes -= entry.record.host_bytes
            del self._entries[key]


__all__ = (
    'PageKey', 'BF16LayerPayload', 'HostPageRecord', 'PageLease',
    'ReloadTicket', 'RequestLocalPageStore', 'make_bf16_layer_payload',
    'SCOPE', 'STATE', 'PAGE_SIZE', 'KV_HEADS', 'HEAD_DIM',
    'QSA_LAYER_IDS',
)
