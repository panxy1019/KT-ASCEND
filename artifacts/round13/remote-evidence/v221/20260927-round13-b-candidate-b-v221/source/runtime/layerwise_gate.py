"""Strict admission for the separately measured layerwise BF16 transport.

This validates one SHA-bound component-probe receipt.  The receipt explicitly
scopes numerical execution to layer 0; the 48-layer sweep is source-address
identity evidence and is not a claim of full-model scheduler qualification.
"""
import hashlib
import json
import math
from pathlib import Path
import re


_MIB = 1024**2
_MAX_ALLOCATOR_BYTES = 43_008 * _MIB
_SOURCE_FILES = {'layerwise.py', 'dynamic_adapter.py', 'expert_source.py'}
_GEOMETRY = {
    'hidden': 2560,
    'intermediate': 640,
    'experts': 512,
    'layers': 48,
    'topk': 10,
    'dtype': 'torch.bfloat16',
    'w13_shape': [32, 2560, 1280],
    'w2_shape': [32, 640, 2560],
    'private_bank_bytes': 314572800,
}
_NUMERIC_MATRIX = {
    (tokens, kind)
    for tokens in (1, 10, 32)
    for kind in ('one_group', 'four_block_multigroup')
}
_NUMERIC_FIELDS = {
    'tokens', 'kind', 'groups', 'relative_l2',
    'relative_l2_tolerance', 'finite', 'execution_peak_extra_bytes',
    'output_sha256', 'all_top10_routes_executed',
    'last_group_original_weights_exact', 'private_adapter_copy_only',
    'template_residents_unchanged',
}
_SWEEP_FIELDS = {
    'layer', 'logical_expert', 'gate_sha256', 'up_sha256', 'down_sha256',
    'combined_sha256',
}


def _sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _positive_int(value):
    return type(value) is int and value > 0


def _validate_source_hashes(actual, expected):
    if (not isinstance(expected, dict) or set(expected) != _SOURCE_FILES or
            any(not _sha256(value) for value in expected.values()) or
            actual != expected):
        raise RuntimeError('layerwise runtime source SHA binding mismatch')


def _validate_source_sweep(rows):
    if not isinstance(rows, list) or len(rows) != 48:
        raise RuntimeError('layerwise source identity sweep must cover all 48 layers')
    combined = set()
    for layer, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _SWEEP_FIELDS:
            raise RuntimeError('layerwise source identity sweep fields mismatch')
        if (row['layer'] != layer or
                row['logical_expert'] != (layer * 11 + 7) % 512):
            raise RuntimeError('layerwise source identity sweep order mismatch')
        projections = [row[f'{name}_sha256'] for name in ('gate', 'up', 'down')]
        if any(not _sha256(value) for value in projections):
            raise RuntimeError('layerwise source projection hash is invalid')
        expected_combined = hashlib.sha256(
            ''.join(projections).encode('ascii')
        ).hexdigest()
        if row['combined_sha256'] != expected_combined:
            raise RuntimeError('layerwise combined source identity hash mismatch')
        combined.add(expected_combined)
    if len(combined) != 48:
        raise RuntimeError('layerwise source identity sweep is not layer-distinct')


def _validate_numeric_cases(rows, execution_peak):
    if (not isinstance(rows, list) or len(rows) != 6 or
            any(not isinstance(row, dict) for row in rows) or
            {(row.get('tokens'), row.get('kind')) for row in rows} != _NUMERIC_MATRIX):
        raise RuntimeError('layerwise numeric matrix is incomplete')
    for row in rows:
        if set(row) != _NUMERIC_FIELDS:
            raise RuntimeError('layerwise numeric evidence fields mismatch')
        relative_l2 = row['relative_l2']
        tolerance = row['relative_l2_tolerance']
        peak = row['execution_peak_extra_bytes']
        groups = row['groups']
        expected_min_groups = 2 if (
            row['kind'] == 'four_block_multigroup' and row['tokens'] > 1
        ) else 1
        if (type(relative_l2) not in (int, float) or
                not math.isfinite(relative_l2) or tolerance != .01 or
                not 0 <= relative_l2 <= tolerance or row['finite'] is not True or
                not _positive_int(peak) or peak > execution_peak or
                not _positive_int(groups) or groups < expected_min_groups or
                not _sha256(row['output_sha256'])):
            raise RuntimeError('layerwise numeric or resource evidence failed')
        for field in (
                'all_top10_routes_executed', 'last_group_original_weights_exact',
                'private_adapter_copy_only', 'template_residents_unchanged'):
            if row[field] is not True:
                raise RuntimeError('layerwise numerical ownership evidence failed')


def _validate_lifecycle(gate):
    if gate.get('abort_probe') != {
            'inflight_abort_synchronized': True,
            'backend_reusable_after_inflight_abort': True,
            'public_abort_reuse': True,
    }:
        raise RuntimeError('layerwise abort and reuse evidence mismatch')
    if gate.get('unused_slot_probe') != {
            'loaded_unused_slots': 22,
            'unchanged_after_execution': True,
            'numerically_ignored': True,
    }:
        raise RuntimeError('layerwise unused-slot evidence mismatch')
    if (gate.get('original_resident_weights_and_maps_exact') is not True or
            gate.get('private_shadow_has_no_resident_weight_alias') is not True or
            not _positive_int(gate.get('copied_original_bf16_bytes'))):
        raise RuntimeError('layerwise private ownership evidence mismatch')


def require_layerwise_device_gate(path, expected_sha, *, source_identity,
                                  native_sha256, source_sha256):
    """Load a real layerwise component receipt bound to exact runtime sources.

    Passing this gate does not establish a full-model scheduler, throughput,
    long-context, or F2/F3 completion claim.
    """
    if not path or not _sha256(expected_sha):
        raise RuntimeError('layerwise device receipt and SHA are required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError('layerwise device receipt SHA mismatch')
    gate = json.loads(raw)
    if (not isinstance(gate, dict) or gate.get('schema') != 1 or
            gate.get('status') != 'PASS_LAYERWISE_BF16_TRANSPORT' or
            gate.get('scope') != 'LAYER0_SYNTHETIC_REAL_NPU_NOT_FULL_MODEL_OR_SCHEDULER'):
        raise RuntimeError('layerwise receipt schema, status or scope mismatch')
    if (gate.get('capacity') != 32 or gate.get('resident_capacity') != 32 or
            gate.get('max_group_rows') != 32 or gate.get('storage') != 'BF16' or
            gate.get('fixture_layers') != [0] or gate.get('geometry') != _GEOMETRY):
        raise RuntimeError('layerwise capacity, fixture or geometry mismatch')

    if (not isinstance(source_identity, dict) or not source_identity or
            gate.get('source_identity') != source_identity or
            not _sha256(native_sha256) or gate.get('native_sha256') != native_sha256):
        raise RuntimeError('layerwise original model or native identity mismatch')
    _validate_source_hashes(gate.get('source_sha256'), source_sha256)

    allocator = gate.get('allocator_limit_bytes')
    constructor_peak = gate.get('constructor_peak_extra_bytes')
    execution_peak = gate.get('execution_peak_extra_bytes')
    if (not _positive_int(allocator) or allocator > _MAX_ALLOCATOR_BYTES or
            not _positive_int(constructor_peak) or constructor_peak > allocator or
            not _positive_int(execution_peak) or execution_peak > allocator):
        raise RuntimeError('layerwise measured resource evidence is invalid')

    _validate_source_sweep(gate.get('source_layer_identity_sweep'))
    _validate_numeric_cases(gate.get('numeric_cases'), execution_peak)
    _validate_lifecycle(gate)
    return gate
