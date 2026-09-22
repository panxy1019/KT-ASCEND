"""Versioned opt-in contract, separate from frozen candidate/bank identity.

Accepting a Q32 schema is not device admission. Each runtime consumer must enforce
its currently qualified physical capacity. No implicit environment enablement.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class BufferIdentity:
    request_id: str
    placement_epoch: int
    allocation_epoch: int

    def __post_init__(self):
        if (not isinstance(self.request_id, str) or not self.request_id or
                any(type(n) is not int or n < 0 for n in (self.placement_epoch, self.allocation_epoch))):
            raise ValueError('invalid request/placement/allocation identity')


def parse_framework(value):
    if value is None: return None
    fields = {'schema', 'resident_capacity', 'route_collection', 'frequency'}
    if isinstance(value, dict) and value.get('schema') in (2, 3, 4):
        fields.add('prefill')
        if value['schema'] >= 3: fields.add('ple')
        if value['schema'] == 4: fields.add('joint_host_budget')
    if (not isinstance(value, dict) or
            set(value) != fields or
            type(value['schema']) is not int or value['schema'] not in (1, 2, 3, 4) or
            type(value['resident_capacity']) is not int or value['resident_capacity'] not in (32, 64) or
            type(value['route_collection']) is not bool):
        raise ValueError('invalid framework version/fields/capacity')
    frequency = value['frequency']
    if frequency is not None and (not isinstance(frequency, dict) or
            set(frequency) != {'path', 'sha256'} or not isinstance(frequency['path'], str) or
            not Path(frequency['path']).is_absolute() or
            not isinstance(frequency['sha256'], str) or
            re.fullmatch('[0-9a-f]{64}', frequency['sha256']) is None):
        raise ValueError('frequency requires absolute path and frozen SHA256')
    prefill = value.get('prefill')
    if prefill is not None:
        if (not isinstance(prefill, dict) or
                set(prefill) not in ({'mode', 'threshold_tokens', 'device_gate'},
                                    {'mode', 'threshold_tokens', 'device_gate', 'update_residents'}) or
                prefill['mode'] != 'LAYERWISE' or
                type(prefill.get('update_residents', False)) is not bool or
                type(prefill['threshold_tokens']) is not int or prefill['threshold_tokens'] < 1):
            raise ValueError('invalid bounded layerwise prefill contract')
        gate = prefill['device_gate']
        if (not isinstance(gate, dict) or set(gate) != {'path', 'sha256'} or
                not isinstance(gate['path'], str) or not Path(gate['path']).is_absolute() or
                not isinstance(gate['sha256'], str) or
                re.fullmatch('[0-9a-f]{64}', gate['sha256']) is None):
            raise ValueError('layerwise requires frozen absolute device receipt')
        if frequency is not None:
            raise ValueError('layerwise with Frequency requires a separate interaction gate')
    ple = value.get('ple')
    if ple is not None:
        if not isinstance(ple, dict) or set(ple) != {'mode','device_gate'} or ple['mode'] != 'ASYNC':
            raise ValueError('invalid exact-input PLE async contract')
        gate = ple['device_gate']
        if (not isinstance(gate,dict) or set(gate) != {'path','sha256'} or
                not isinstance(gate['path'],str) or not Path(gate['path']).is_absolute() or
                not isinstance(gate['sha256'],str) or re.fullmatch('[0-9a-f]{64}',gate['sha256']) is None):
            raise ValueError('async PLE requires frozen absolute device receipt')
    joint=value.get('joint_host_budget',False)
    if type(joint) is not bool or (joint and ple is None):
        raise ValueError('joint host budget requires explicit boolean and async PLE')
    return copy.deepcopy(value)


def load_framework():
    path = os.environ.get('ROUND9_FRAMEWORK_PROFILE')
    expected = os.environ.get('ROUND9_FRAMEWORK_PROFILE_SHA256')
    if not path and not expected: return None
    if not path or not expected: raise RuntimeError('framework profile and SHA required together')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected: raise RuntimeError('framework profile changed')
    return parse_framework(json.loads(raw))
