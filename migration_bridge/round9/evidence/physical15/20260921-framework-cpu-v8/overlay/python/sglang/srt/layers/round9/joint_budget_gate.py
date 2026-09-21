"""Source-bound admission for synthetic PLE/expert transport contention.

This receipt qualifies shared credit accounting and transport progress only.  It
does not qualify expert values, model execution, placement quality, or latency.
"""
import hashlib
import json
from pathlib import Path
import re


SOURCE_FILES = frozenset((
    'host_budget.py', 'budgeted_migration.py', 'ple_async.py',
))
CHECKS = frozenset((
    'physical_mapping',
    'ple_current_overlap',
    'migration_h2d_exact',
    'required_progress_with_prefetch',
    'prefetch_cancel_reclaim',
    'no_capacity_exceeded',
    'request_retirement',
    'zero_credits_after_close',
    'full_q32_budget_math',
))
SNAPSHOT_STAGES = frozenset((
    'ple_current_live', 'prefetch_live', 'migration_live',
    'after_migration', 'after_cancel', 'final',
))
EXPERT_BYTES = 2560 * 640 * 3 * 2
RESIDENT_CAPACITY = 32
MIGRATION_PROBE_EXPERTS = 2
PLE_SLOT_BYTES = 512 * 160 * 2
EXPECTED_MIGRATION_SHA256 = (
    '8eca02bdec774408ad743e89af338ac0f5a65e38532d9b25e5d1ce7ef8b28896'
)
MIN_PEAK_EXTRA_BYTES = MIGRATION_PROBE_EXPERTS * EXPERT_BYTES + 2 * PLE_SLOT_BYTES
CONFIGURED_BUDGET = {
    'capacity': {
        'host_bytes': 4 * EXPERT_BYTES + 6 * PLE_SLOT_BYTES,
        'h2d_bytes': RESIDENT_CAPACITY * EXPERT_BYTES + 2 * PLE_SLOT_BYTES,
        'pinned_bytes': 2 * PLE_SLOT_BYTES,
    },
    'reserve': {
        'host_bytes': 4 * EXPERT_BYTES,
        'h2d_bytes': RESIDENT_CAPACITY * EXPERT_BYTES,
        'pinned_bytes': 0,
    },
}
_CURRENT_HOST = 2 * PLE_SLOT_BYTES
_CURRENT_H2D = PLE_SLOT_BYTES
_PREFETCH_PAYLOAD = 16 * 160 * 2
EXPECTED_SNAPSHOTS = {
    'ple_current_live': {
        'used': {'host_bytes': _CURRENT_HOST, 'h2d_bytes': _CURRENT_H2D,
                 'pinned_bytes': 2 * PLE_SLOT_BYTES},
        'allocations': 3, 'active_requests': 3, 'in_flight': 1,
    },
    'prefetch_live': {
        'used': {'host_bytes': _CURRENT_HOST + 2 * _PREFETCH_PAYLOAD,
                 'h2d_bytes': _CURRENT_H2D + _PREFETCH_PAYLOAD,
                 'pinned_bytes': 2 * PLE_SLOT_BYTES},
        'allocations': 4, 'active_requests': 3, 'in_flight': 2,
    },
    'migration_live': {
        'used': {'host_bytes': _CURRENT_HOST + 2 * _PREFETCH_PAYLOAD +
                               4 * EXPERT_BYTES,
                 'h2d_bytes': _CURRENT_H2D + _PREFETCH_PAYLOAD +
                              MIGRATION_PROBE_EXPERTS * EXPERT_BYTES,
                 'pinned_bytes': 2 * PLE_SLOT_BYTES},
        'allocations': 5, 'active_requests': 3, 'in_flight': 3,
    },
    'after_migration': {
        'used': {'host_bytes': _CURRENT_HOST + 2 * _PREFETCH_PAYLOAD,
                 'h2d_bytes': _CURRENT_H2D + _PREFETCH_PAYLOAD,
                 'pinned_bytes': 2 * PLE_SLOT_BYTES},
        'allocations': 4, 'active_requests': 3, 'in_flight': 2,
    },
    'after_cancel': {
        'used': {'host_bytes': 0, 'h2d_bytes': 0,
                 'pinned_bytes': 2 * PLE_SLOT_BYTES},
        'allocations': 2, 'active_requests': 3, 'in_flight': 0,
    },
    'final': {
        'used': {'host_bytes': 0, 'h2d_bytes': 0, 'pinned_bytes': 0},
        'allocations': 0, 'active_requests': 0, 'in_flight': 0,
    },
}


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _validate_snapshot(item):
    if (not isinstance(item, dict) or
            set(item) != {'stage', 'used', 'allocations', 'active_requests', 'in_flight'} or
            item['stage'] not in SNAPSHOT_STAGES or
            not isinstance(item['used'], dict) or
            set(item['used']) != set(CONFIGURED_BUDGET['capacity']) or
            any(type(item[key]) is not int or item[key] < 0
                for key in ('allocations', 'active_requests', 'in_flight')) or
            any(type(value) is not int or value < 0 or
                value > CONFIGURED_BUDGET['capacity'][name]
                for name, value in item['used'].items())):
        raise RuntimeError('joint budget snapshot is malformed or exceeds capacity')


def require_joint_budget_device_gate(
        path, expected_sha, *, source_identity, source_sha256,
        ple_gate_sha256, torch_version, torch_npu_version):
    if not path or not _sha(expected_sha) or not _sha(ple_gate_sha256):
        raise RuntimeError('joint budget requires frozen gate and standalone PLE SHA256')
    if (not isinstance(source_identity, dict) or not source_identity or
            not isinstance(source_sha256, dict) or set(source_sha256) != SOURCE_FILES or
            any(not _sha(value) for value in source_sha256.values()) or
            not isinstance(torch_version, str) or not torch_version or
            not isinstance(torch_npu_version, str) or not torch_npu_version):
        raise RuntimeError('joint budget caller identity is incomplete')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError('joint budget receipt SHA changed')
    gate = json.loads(raw)
    exact = {
        'schema': 1,
        'status': 'PASS_JOINT_BUDGET_SYNTHETIC_CONTENTION',
        'scope': 'SYNTHETIC_Q32_TRANSPORT_CREDITS_NOT_EXPERT_MATH_OR_MODEL',
        'physical_device': 15,
        'source_identity': source_identity,
        'source_sha256': source_sha256,
        'ple_gate_sha256': ple_gate_sha256,
        'torch_version': torch_version,
        'torch_npu_version': torch_npu_version,
        'expert_bytes': EXPERT_BYTES,
        'resident_capacity': RESIDENT_CAPACITY,
        'migration_probe_experts': MIGRATION_PROBE_EXPERTS,
        'configured_budget': CONFIGURED_BUDGET,
    }
    required = set(exact) | {'checks', 'snapshots', 'case', 'peak_extra_bytes'}
    if (not isinstance(gate, dict) or set(gate) != required or
            any(gate.get(name) != value for name, value in exact.items())):
        raise RuntimeError('joint budget receipt identity or geometry mismatch')
    checks = gate['checks']
    if (not isinstance(checks, dict) or set(checks) != CHECKS or
            any(value is not True for value in checks.values())):
        raise RuntimeError('joint budget contention checks are incomplete')
    snapshots = gate['snapshots']
    if (not isinstance(snapshots, list) or len(snapshots) != len(SNAPSHOT_STAGES)):
        raise RuntimeError('joint budget snapshots are incomplete')
    for item in snapshots:
        _validate_snapshot(item)
    if {item['stage'] for item in snapshots} != SNAPSHOT_STAGES:
        raise RuntimeError('joint budget snapshot stages differ')
    for item in snapshots:
        expected_snapshot = EXPECTED_SNAPSHOTS[item['stage']]
        if any(item[name] != expected_snapshot[name] for name in
               ('used', 'allocations', 'active_requests', 'in_flight')):
            raise RuntimeError(f"joint budget {item['stage']} snapshot differs")
    case = gate['case']
    expected_case = {
        'ple_rows': 512,
        'ple_row_width': 160,
        'ple_payload_bytes': PLE_SLOT_BYTES,
        'migration_experts': MIGRATION_PROBE_EXPERTS,
        'actual_migration_h2d_bytes': MIGRATION_PROBE_EXPERTS * EXPERT_BYTES,
        'full_q32_h2d_bytes': RESIDENT_CAPACITY * EXPERT_BYTES,
    }
    if (not isinstance(case, dict) or set(case) != set(expected_case) |
            {'ple_sha256', 'migration_sha256'} or
            any(case.get(name) != value for name, value in expected_case.items()) or
            not _sha(case.get('ple_sha256')) or not _sha(case.get('migration_sha256'))):
        raise RuntimeError('joint budget transport case is incomplete')
    if case['migration_sha256'] != EXPECTED_MIGRATION_SHA256:
        raise RuntimeError('joint budget synthetic migration content differs')
    peak = gate['peak_extra_bytes']
    if type(peak) is not int or not MIN_PEAK_EXTRA_BYTES <= peak <= 1024**3:
        raise RuntimeError('joint budget peak device allocation is invalid')
    return gate


__all__ = (
    'SOURCE_FILES', 'CHECKS', 'SNAPSHOT_STAGES', 'EXPERT_BYTES',
    'RESIDENT_CAPACITY', 'MIGRATION_PROBE_EXPERTS', 'PLE_SLOT_BYTES',
    'EXPECTED_MIGRATION_SHA256', 'MIN_PEAK_EXTRA_BYTES', 'CONFIGURED_BUDGET',
    'EXPECTED_SNAPSHOTS', 'require_joint_budget_device_gate',
)
