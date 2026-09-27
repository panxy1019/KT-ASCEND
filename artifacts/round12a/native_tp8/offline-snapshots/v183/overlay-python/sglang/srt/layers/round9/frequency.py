"""Replayable Qwen3.8 per-layer placement; never falls back to synthetic counts.

Digests establish identity, not authenticity or quality qualification. The launch
receipt must bind the producer identity to the actual model/corpus/runtime.
"""
import copy
import hashlib
import json
import re

from .route_stats import frequency_mask


IDENTITY_KEYS = {'model_manifest_sha256', 'tokenizer_sha256', 'runtime_sha256',
                 'corpus_sha256', 'placement_sha256'}


def payload_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _identity(value):
    if (not isinstance(value, dict) or set(value) != IDENTITY_KEYS or
            any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None
                for v in value.values())):
        raise ValueError('complete producer identity with SHA256 values required')


def build_frequency(statistics, identity, *, capacity):
    _identity(identity)
    if capacity not in (32, 64) or type(capacity) is not int:
        raise ValueError('only physical Q32/Q64 contracts are defined')
    if tuple(statistics.get(k) for k in ('layers', 'experts', 'topk')) != (48, 512, 10):
        raise ValueError('Qwen3.8 independent 48x512 Top10 statistics required')
    residents = frequency_mask(statistics, capacity)
    payload = dict(schema=1, strategy='frequency_per_layer', capacity=capacity,
                   qualification='GENERATED_NOT_DEVICE_QUALIFIED',
                   identity=copy.deepcopy(identity), statistics=copy.deepcopy(statistics),
                   residents=residents)
    payload['payload_sha256'] = payload_digest(payload)
    return payload


def validate_frequency(document, expected_identity, *, capacity):
    _identity(expected_identity)
    if not isinstance(document, dict) or document.get('identity') != expected_identity:
        raise ValueError('frequency producer identity mismatch')
    expected = build_frequency(document.get('statistics', {}), expected_identity, capacity=capacity)
    if document != expected:
        raise ValueError('frequency schema/digest/capacity/deterministic placement mismatch')
    return copy.deepcopy(expected['residents'])
