"""Opt-in context pool measurement, after real pool construction.

This flag sizes and measures pools. It grants no long-context execution,
quantized KV consumer, or cache-tier qualification.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re

from .context_profile import derive_qwen4_tp1_geometry, derive_qwen4_tp1_footprint


def parse_context_probe(document):
    if (not isinstance(document, dict) or
            set(document) != {'schema', 'context_tokens', 'model_config_sha256'} or
            type(document['schema']) is not int or document['schema'] != 1 or
            type(document['context_tokens']) is not int or
            document['context_tokens'] not in (4096, 16384, 65536, 131072) or
            not isinstance(document['model_config_sha256'], str) or
            re.fullmatch('[0-9a-f]{64}', document['model_config_sha256']) is None):
        raise ValueError('invalid context pool probe profile')
    return dict(document)


def load_context_probe():
    path = os.environ.get('ROUND9_CONTEXT_PROBE_PROFILE')
    expected = os.environ.get('ROUND9_CONTEXT_PROBE_SHA256')
    if not path and not expected:
        return None
    if not path or not expected or not Path(path).is_absolute():
        raise ValueError('context probe requires absolute profile and SHA together')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('context probe profile changed')
    return parse_context_probe(json.loads(raw))


def context_pools_ready(runner):
    probe = load_context_probe()
    if probe is None:
        return
    args = runner.server_args
    target = probe['context_tokens']
    if (args.tp_size != 1 or args.device != 'npu' or
            args.max_running_requests != 1 or args.max_mamba_cache_size != 1 or
            args.max_total_tokens != target + 4 or args.context_length != target + 2 or
            args.page_size != 4 or not args.disable_radix_cache or
            not args.disable_overlap_schedule or not args.disable_cuda_graph):
        raise RuntimeError('context probe requires the exact serial TP1 pool profile')
    raw = (Path(args.model_path) / 'config.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != probe['model_config_sha256']:
        raise RuntimeError('context pool model configuration changed')
    geometry = derive_qwen4_tp1_geometry(json.loads(raw), config_scope='full_config')
    footprint = derive_qwen4_tp1_footprint(target, geometry)
    from .context_measurement import measure_and_validate_qwen4_tp1_pools
    measured = measure_and_validate_qwen4_tp1_pools(runner, footprint, require_device='npu')
    output = Path('/out/context-pools.json')
    document = dict(schema=1, status='PASS_ACTUAL_CONTEXT_POOL_GEOMETRY',
                    scope='POOL_ALLOCATION_ONLY_NOT_LONG_CONTEXT_EXECUTION',
                    probe=probe, footprint=asdict(footprint), measurement=asdict(measured))
    # Exclusive creation prevents a reused process from overwriting evidence.
    with output.open('x') as stream:
        stream.write(json.dumps(document, sort_keys=True, allow_nan=False) + '\n')
    print('ROUND9_CONTEXT_POOLS_MEASURED', target, measured.total_bytes, flush=True)
