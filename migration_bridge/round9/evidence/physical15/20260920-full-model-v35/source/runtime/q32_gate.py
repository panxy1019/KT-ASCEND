"""Strict Q32 receipt checks, separate from every historical Q64 gate.

``max_device_staging_bytes`` covers only the explicit projection tensor. It is
not an operator-workspace or total-HBM claim. Measured migration allocation is
recorded separately and remains a bounded-probe result, not F7 qualification.

The exact per-layer physical record consumed by ``validate_q32_physical_layers``
is::

    {
      "layer": 0,
      "resident_capacity": 32,
      "tensors": {
        "w13_weight": {"shape": [32, 2560, 1280],
                       "dtype": "torch.bfloat16", "bytes": 209715200},
        "w2_weight": {"shape": [32, 640, 2560],
                      "dtype": "torch.bfloat16", "bytes": 104857600}
      },
      "total_bytes": 314572800
    }
"""
import hashlib
import json
import math
from pathlib import Path
import re


_MIB = 1024**2
_GIB = 1024**3
_FORMATS = {'w13_weight': 29, 'w2_weight': 29}
_BATCH_COPY = {
    'experts': 32,
    'reverse_and_restore_exact': True,
    'max_device_staging_bytes': 200 * _MIB,
    'staging_scope': 'EXPLICIT_PROJECTION_TENSOR_ONLY',
}
_PHYSICAL_CAPACITY = {
    'resident_capacity': 32,
    'tensors': {
        'w13_weight': {
            'shape': [32, 2560, 1280],
            'dtype': 'torch.bfloat16',
            'bytes': 209715200,
        },
        'w2_weight': {
            'shape': [32, 640, 2560],
            'dtype': 'torch.bfloat16',
            'bytes': 104857600,
        },
    },
    'total_bytes': 314572800,
}
_MATRIX = {(tokens, kind) for tokens in (1, 10, 32)
           for kind in ('mixed', 'resident_only', 'repeated_cpu')}
_ERRORS = {'cpu_partial', 'npu_owned_partial', 'routed_merge'}


def _sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _validate_measured_migration(value):
    if not isinstance(value, dict) or set(value) != {
            'peak_extra_allocated_bytes', 'allocator_limit_bytes', 'samples'}:
        raise RuntimeError('Q32 measured migration fields are incomplete')
    peak = value['peak_extra_allocated_bytes']
    limit = value['allocator_limit_bytes']
    samples = value['samples']
    if (type(peak) is not int or not 0 <= peak <= _GIB or
            type(limit) is not int or not 0 < limit <= 3 * _GIB or
            peak > limit or type(samples) is not int or samples < 2):
        raise RuntimeError('Q32 measured migration allocation is outside the bounded probe')


def _validate_rows(rows):
    if (not isinstance(rows, list) or len(rows) != 9 or
            {(row.get('tokens'), row.get('kind')) for row in rows} != _MATRIX):
        raise RuntimeError('Q32 KTEP numeric matrix is incomplete')
    for row in rows:
        errors = row.get('errors')
        hashes = row.get('hashes')
        if (not isinstance(errors, dict) or set(errors) != _ERRORS or
                any(type(error) not in (int, float) or not math.isfinite(error)
                    or not 0 <= error <= .01 for error in errors.values()) or
                not isinstance(hashes, list) or len(hashes) != 3 or
                any(not _sha256(digest) for digest in hashes) or len(set(hashes)) != 1):
            raise RuntimeError('Q32 KTEP repeat or numeric gate failed')


def _validate_migrations(migrations):
    if not isinstance(migrations, list) or len(migrations) != 3:
        raise RuntimeError('Q32 migration evidence requires three transactions')
    if ([item.get('generation') for item in migrations] != [1, 2, 3] or
            [item.get('moves') for item in migrations] !=
            [[[0, 511]], [[0, 0]], [[0, 511]]]):
        raise RuntimeError('Q32 promotion/demotion sequence is incomplete')
    for item in migrations:
        if (item.get('all32_processed_experts_exact') is not True or
                item.get('pinned_mask_pointer_stable') is not True or
                item.get('formats_after') != _FORMATS):
            raise RuntimeError('Q32 migration identity or NZ29 format gate failed')


def require_q32_device_gate(path, expected_sha, *, storage, workers,
                            source_identity, layer0_sha256, native_sha256):
    """Load and validate one source-bound Q32 module receipt.

    Passing this check establishes only the bounded receipt contract. It does
    not establish all-layer integration, total HBM, smoke, or F7 completion.
    """
    if not path or not expected_sha:
        raise RuntimeError('Q32 device gate receipt and SHA are required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError('Q32 device gate SHA mismatch')
    gate = json.loads(raw)
    if (gate.get('schema') != 1 or gate.get('resident_capacity') != 32 or
            gate.get('status') != 'PASS_Q32_REAL_KTEP_LOCAL_MERGE'):
        raise RuntimeError('Q32 receipt schema, capacity or status mismatch')
    if (gate.get('storage') != storage or gate.get('cpu_workers') != workers or
            gate.get('source_identity') != source_identity or
            gate.get('bank_sha256') != layer0_sha256 or
            not _sha256(native_sha256) or gate.get('native_sha256') != native_sha256):
        raise RuntimeError('Q32 receipt source, storage, workers, bank or native identity mismatch')
    if gate.get('npu_weight_formats') != _FORMATS:
        raise RuntimeError('Q32 receipt requires actual NZ29 resident tensors')
    if gate.get('physical_capacity') != _PHYSICAL_CAPACITY:
        raise RuntimeError('Q32 physical resident tensor geometry or bytes mismatch')
    if gate.get('batch_copy_probe') != _BATCH_COPY:
        raise RuntimeError('Q32 explicit projection staging evidence mismatch')
    _validate_measured_migration(gate.get('measured_migration'))
    _validate_migrations(gate.get('dynamic_migrations'))
    _validate_rows(gate.get('rows'))
    return gate


def validate_q32_physical_layers(records):
    """Validate exact runtime tensor records for all 48 layers and sum bytes."""
    if not isinstance(records, (list, tuple)) or len(records) != 48:
        raise ValueError('Q32 physical record requires exactly 48 layers')
    layers = set()
    expected_fields = {'layer', 'resident_capacity', 'tensors', 'total_bytes'}
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError('invalid Q32 physical layer record fields')
        layer = record['layer']
        if type(layer) is not int or not 0 <= layer < 48 or layer in layers:
            raise ValueError('missing, duplicate or invalid Q32 layer')
        layers.add(layer)
        physical = {key: record[key] for key in expected_fields - {'layer'}}
        if physical != _PHYSICAL_CAPACITY:
            raise ValueError('Q32 physical layer geometry, dtype or bytes mismatch')
    if layers != set(range(48)):
        raise ValueError('Q32 physical record does not cover all layers')
    return {
        'schema': 1,
        'layers': 48,
        'resident_capacity': 32,
        'per_layer_bytes': _PHYSICAL_CAPACITY['total_bytes'],
        'total_bytes': 48 * _PHYSICAL_CAPACITY['total_bytes'],
    }
