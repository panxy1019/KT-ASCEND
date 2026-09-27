"""Canonical CPU FP8 E4M3FN QSA K/V page records for F9 groundwork.

This module defines a storage-format oracle only. It does not install a cache,
publish a page mapping, import an NPU backend, or qualify native FP8 execution.
"""
from dataclasses import dataclass
import hashlib
import hmac
import importlib
import json
import math
import struct

from .context_profile import ContextStateIdentity


SCOPE = 'CPU_REFERENCE_ONLY'
NATIVE_BACKEND_STATUS = 'NOT_IMPLEMENTED_UNQUALIFIED'
FORMAT_VERSION = 1
PAGE_SIZE = 4
KV_HEADS = 2
HEAD_DIM = 256
MAX_LOGICAL_PAGES = 32769
QSA_LAYER_IDS = frozenset(range(3, 48, 4))
FP8_FORMAT = 'float8_e4m3fn'

_MAGIC = b'R9KVFP8\0'
_HEADER = struct.Struct('<8sIII')
_DIGEST_BYTES = 32
_MAX_METADATA_BYTES = 4096
_MAX_PLANE_BYTES = PAGE_SIZE * KV_HEADS * HEAD_DIM


@dataclass(frozen=True, slots=True)
class KVPageRecord:
    identity: ContextStateIdentity
    logical_page: int
    layer_id: int
    valid_tokens: int
    shape: tuple
    k_scale_f32: float
    v_scale_f32: float
    k_bytes: bytes
    v_bytes: bytes
    metadata_bytes: bytes
    digest_sha256: str
    scope: str = SCOPE
    format_version: int = FORMAT_VERSION

    def to_bytes(self):
        _validate_record(self)
        header = _HEADER.pack(
            _MAGIC, len(self.metadata_bytes), len(self.k_bytes), len(self.v_bytes))
        try:
            digest = bytes.fromhex(self.digest_sha256)
        except (TypeError, ValueError) as error:
            raise ValueError('invalid KV page digest encoding') from error
        if len(digest) != _DIGEST_BYTES:
            raise ValueError('invalid KV page digest length')
        return header + self.metadata_bytes + self.k_bytes + self.v_bytes + digest


def _torch():
    try:
        torch = importlib.import_module('torch')
    except ModuleNotFoundError as error:
        raise RuntimeError('CPU FP8 reference requires PyTorch') from error
    if not hasattr(torch, 'float8_e4m3fn'):
        raise RuntimeError('PyTorch build lacks float8_e4m3fn')
    return torch


def _identity(identity):
    if (not isinstance(identity, ContextStateIdentity)
            or identity.component != 'QSA_KV'):
        raise ValueError('KV page requires a QSA_KV ContextStateIdentity')
    if identity.prefix_id is not None:
        raise ValueError('prefix-shared KV pages are not supported')
    return identity


def _coordinates(logical_page, layer_id):
    if (type(logical_page) is not int
            or not 0 <= logical_page < MAX_LOGICAL_PAGES):
        raise ValueError('logical KV page is outside the bounded 128K plan')
    if type(layer_id) is not int or layer_id not in QSA_LAYER_IDS:
        raise ValueError('layer is not an actual QSA layer')


def _canonical_f32(value, name):
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive FP32 scale')
    try:
        packed = struct.pack('<f', value)
    except (OverflowError, struct.error) as error:
        raise ValueError(f'{name} must be a finite positive FP32 scale') from error
    rounded = struct.unpack('<f', packed)[0]
    if rounded != value:
        raise ValueError(f'{name} is not canonical FP32')
    return rounded


def _scale_hex(value):
    return struct.pack('<f', value).hex()


def _scale_from_hex(value, name):
    if not isinstance(value, str) or len(value) != 8:
        raise ValueError(f'invalid {name} metadata')
    try:
        scale = struct.unpack('<f', bytes.fromhex(value))[0]
    except (ValueError, struct.error) as error:
        raise ValueError(f'invalid {name} metadata') from error
    return _canonical_f32(float(scale), name)


def _metadata(identity, logical_page, layer_id, shape, k_scale, v_scale):
    return {
        'allocation_epoch': identity.allocation_epoch,
        'component': identity.component,
        'format_version': FORMAT_VERSION,
        'head_dim': HEAD_DIM,
        'k_format': FP8_FORMAT,
        'k_scale_f32_le': _scale_hex(k_scale),
        'kv_heads': KV_HEADS,
        'layer_id': layer_id,
        'logical_page': logical_page,
        'page_size': PAGE_SIZE,
        'prefix_id': identity.prefix_id,
        'request_id': identity.request_id,
        'scope': SCOPE,
        'shape': list(shape),
        'source_dtype': 'bfloat16',
        'v_format': FP8_FORMAT,
        'v_scale_f32_le': _scale_hex(v_scale),
        'valid_tokens': shape[0],
    }


def _metadata_bytes(identity, logical_page, layer_id, shape, k_scale, v_scale):
    return json.dumps(
        _metadata(identity, logical_page, layer_id, shape, k_scale, v_scale),
        sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False,
    ).encode('ascii')


def _body(metadata_bytes, k_bytes, v_bytes):
    return (_HEADER.pack(_MAGIC, len(metadata_bytes), len(k_bytes), len(v_bytes))
            + metadata_bytes + k_bytes + v_bytes)


def _digest(metadata_bytes, k_bytes, v_bytes):
    return hashlib.sha256(_body(metadata_bytes, k_bytes, v_bytes)).hexdigest()


def _tensor(tensor, name):
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor')
    if tensor.device.type != 'cpu':
        raise ValueError(f'{name} must be on CPU')
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f'{name} must be BF16')
    shape = tuple(tensor.shape)
    if (len(shape) != 3 or not 1 <= shape[0] <= PAGE_SIZE
            or shape[1:] != (KV_HEADS, HEAD_DIM)):
        raise ValueError(
            f'{name} shape must be [1..{PAGE_SIZE},{KV_HEADS},{HEAD_DIM}]')
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f'{name} must contain only finite values')
    return tensor.detach().contiguous(), shape


def _quantize(tensor):
    torch = _torch()
    fp32 = tensor.to(torch.float32)
    amax = fp32.abs().max()
    if float(amax.item()) == 0.0:
        scale = 1.0
    else:
        limit = float(torch.finfo(torch.float8_e4m3fn).max)
        scale = struct.unpack('<f', struct.pack('<f', float(amax.item()) / limit))[0]
    scale = _canonical_f32(float(scale), 'scale')
    quantized = (fp32 / scale).clamp(
        min=-float(torch.finfo(torch.float8_e4m3fn).max),
        max=float(torch.finfo(torch.float8_e4m3fn).max),
    ).to(torch.float8_e4m3fn)
    payload = quantized.contiguous().view(torch.uint8).numpy().tobytes(order='C')
    return scale, payload


def encode_kv_page(identity, logical_page, layer_id, key, value):
    """Encode one request-owned QSA layer page into canonical host bytes."""
    identity = _identity(identity)
    _coordinates(logical_page, layer_id)
    key, key_shape = _tensor(key, 'key')
    value, value_shape = _tensor(value, 'value')
    if key_shape != value_shape:
        raise ValueError('K/V page shapes differ')
    k_scale, k_bytes = _quantize(key)
    v_scale, v_bytes = _quantize(value)
    metadata = _metadata_bytes(
        identity, logical_page, layer_id, key_shape, k_scale, v_scale)
    if not 0 < len(metadata) <= _MAX_METADATA_BYTES:
        raise ValueError('KV page metadata exceeds the canonical bound')
    digest = _digest(metadata, k_bytes, v_bytes)
    return KVPageRecord(
        identity=identity, logical_page=logical_page, layer_id=layer_id,
        valid_tokens=key_shape[0], shape=key_shape,
        k_scale_f32=k_scale, v_scale_f32=v_scale,
        k_bytes=k_bytes, v_bytes=v_bytes,
        metadata_bytes=metadata, digest_sha256=digest,
    )


def _validate_record(record):
    if not isinstance(record, KVPageRecord):
        raise TypeError('KVPageRecord is required')
    identity = _identity(record.identity)
    _coordinates(record.logical_page, record.layer_id)
    if record.scope != SCOPE:
        raise ValueError('unsupported KV page scope')
    if (type(record.format_version) is not int
            or record.format_version != FORMAT_VERSION):
        raise ValueError('unsupported KV page format version')
    if (not isinstance(record.shape, tuple) or len(record.shape) != 3
            or any(type(dimension) is not int for dimension in record.shape)
            or type(record.valid_tokens) is not int
            or record.shape != (record.valid_tokens, KV_HEADS, HEAD_DIM)
            or not 1 <= record.valid_tokens <= PAGE_SIZE):
        raise ValueError('invalid KV page shape metadata')
    k_scale = _canonical_f32(record.k_scale_f32, 'k_scale_f32')
    v_scale = _canonical_f32(record.v_scale_f32, 'v_scale_f32')
    expected_bytes = record.valid_tokens * KV_HEADS * HEAD_DIM
    if (type(record.k_bytes) is not bytes or type(record.v_bytes) is not bytes
            or len(record.k_bytes) != expected_bytes
            or len(record.v_bytes) != expected_bytes):
        raise ValueError('invalid KV page payload length')
    if (type(record.metadata_bytes) is not bytes
            or not 0 < len(record.metadata_bytes) <= _MAX_METADATA_BYTES):
        raise ValueError('KV page metadata exceeds the canonical bound')
    expected_metadata = _metadata_bytes(
        identity, record.logical_page, record.layer_id, record.shape,
        k_scale, v_scale)
    if record.metadata_bytes != expected_metadata:
        raise ValueError('KV page metadata is noncanonical or inconsistent')
    expected_digest = _digest(record.metadata_bytes, record.k_bytes, record.v_bytes)
    if (not isinstance(record.digest_sha256, str)
            or not hmac.compare_digest(record.digest_sha256, expected_digest)):
        raise ValueError('KV page digest mismatch')
    return record


def _decode_plane(payload, shape, scale):
    torch = _torch()
    raw = torch.empty(len(payload), dtype=torch.uint8, device='cpu')
    memoryview(raw.numpy()).cast('B')[:] = payload
    quantized = raw.view(torch.float8_e4m3fn).reshape(shape)
    fp32 = quantized.to(torch.float32)
    if not bool(torch.isfinite(fp32).all().item()):
        raise ValueError('KV page FP8 payload contains a nonfinite value')
    bf16_limit = float(torch.finfo(torch.bfloat16).max)
    return (fp32 * scale).clamp(
        min=-bf16_limit, max=bf16_limit).to(torch.bfloat16)


def decode_kv_page(record, expected_identity, *, logical_page, layer_id):
    """Validate identity and bytes before returning CPU BF16 K/V tensors."""
    expected_identity = _identity(expected_identity)
    _coordinates(logical_page, layer_id)
    record = _validate_record(record)
    if (record.identity != expected_identity
            or record.logical_page != logical_page
            or record.layer_id != layer_id):
        raise ValueError('stale or mismatched KV page identity')
    return (
        _decode_plane(record.k_bytes, record.shape, record.k_scale_f32),
        _decode_plane(record.v_bytes, record.shape, record.v_scale_f32),
    )


def parse_kv_page_record(blob):
    """Parse one exact canonical record and reject truncation or trailing bytes."""
    if type(blob) is not bytes or len(blob) < _HEADER.size + _DIGEST_BYTES:
        raise ValueError('invalid KV page record bytes')
    try:
        magic, metadata_len, k_len, v_len = _HEADER.unpack_from(blob)
    except struct.error as error:
        raise ValueError('invalid KV page record header') from error
    if (magic != _MAGIC or not 0 < metadata_len <= _MAX_METADATA_BYTES
            or not 0 < k_len <= _MAX_PLANE_BYTES
            or not 0 < v_len <= _MAX_PLANE_BYTES):
        raise ValueError('invalid KV page record bounds')
    body_len = _HEADER.size + metadata_len + k_len + v_len
    if len(blob) != body_len + _DIGEST_BYTES:
        raise ValueError('truncated or trailing KV page bytes')
    metadata_bytes = blob[_HEADER.size:_HEADER.size + metadata_len]
    k_start = _HEADER.size + metadata_len
    v_start = k_start + k_len
    k_bytes = blob[k_start:v_start]
    v_bytes = blob[v_start:body_len]
    digest = blob[body_len:].hex()
    if not hmac.compare_digest(hashlib.sha256(blob[:body_len]).hexdigest(), digest):
        raise ValueError('KV page digest mismatch')
    try:
        metadata = json.loads(metadata_bytes.decode('ascii'))
        identity = ContextStateIdentity(
            metadata['request_id'], metadata['prefix_id'],
            metadata['allocation_epoch'], metadata['component'])
        shape = tuple(metadata['shape'])
        record = KVPageRecord(
            identity=identity,
            logical_page=metadata['logical_page'],
            layer_id=metadata['layer_id'],
            valid_tokens=metadata['valid_tokens'],
            shape=shape,
            k_scale_f32=_scale_from_hex(metadata['k_scale_f32_le'], 'k_scale_f32'),
            v_scale_f32=_scale_from_hex(metadata['v_scale_f32_le'], 'v_scale_f32'),
            k_bytes=k_bytes, v_bytes=v_bytes,
            metadata_bytes=metadata_bytes, digest_sha256=digest,
            scope=metadata['scope'], format_version=metadata['format_version'],
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid KV page metadata') from error
    return _validate_record(record)


__all__ = (
    'KVPageRecord', 'encode_kv_page', 'decode_kv_page',
    'parse_kv_page_record', 'SCOPE', 'NATIVE_BACKEND_STATUS',
    'FORMAT_VERSION', 'PAGE_SIZE', 'KV_HEADS', 'HEAD_DIM',
)
