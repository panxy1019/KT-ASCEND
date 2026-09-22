"""Admission for the synthetic CPU-record / BF16-NPU QSA component only."""
import hashlib
import json
import math
from pathlib import Path

SOURCE_FILES = ('context_profile.py', 'context_page_store.py',
                'kv_quant_reference.py', 'context_staging.py', 'qsa.py')


def require_context_reference_gate(path, sha256, *, source_identity,
                                   source_sha256, torch_version, torch_npu_version):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('context component receipt hash mismatch')
    doc = json.loads(raw)
    if (doc.get('schema') != 1 or type(doc.get('schema')) is not int or
            doc.get('status') != 'PASS_CONTEXT_REFERENCE_QSA_COMPONENT' or
            doc.get('scope') != 'SYNTHETIC_KV_CPU_RECORD_BF16_NPU_QSA_NOT_MODEL_OR_HICACHE' or
            doc.get('physical_device') != 15 or doc.get('source_identity') != source_identity or
            set(source_sha256) != set(SOURCE_FILES) or doc.get('source_sha256') != source_sha256 or
            doc.get('torch_version') != torch_version or doc.get('torch_npu_version') != torch_npu_version or
            doc.get('native_fp8_qualified') is not False or
            doc.get('allocator_free_calls') != 0 or doc.get('stale_lease_rejected') is not True or
            doc.get('qsa_layers') != 12):
        raise ValueError('context component identity or scope mismatch')
    cases = doc.get('cases')
    if not isinstance(cases, list) or [c.get('layer') for c in cases] != list(range(3,48,4)):
        raise ValueError('context component requires all12 ordered QSA cases')
    for case in cases:
        error = case.get('fp8_vs_original_relative_l2')
        if (case.get('bf16_output_exact') is not True or
                case.get('fp8_decoded_output_exact') is not True or
                type(error) not in (float,int) or not math.isfinite(error) or not 0 <= error <= .10 or
                case.get('compact_bytes') != 8192):
            raise ValueError('context component numerical fixture did not pass')
    final = doc.get('host_store_final', {})
    if any(final.get(key) != 0 for key in ('pages','host_bytes','active_leases',
            'inflight_reloads','invalidating','allocator_free_calls','hbm_bytes_freed_claimed')):
        raise ValueError('context host ownership did not drain')
    if type(doc.get('peak_extra_bytes')) is not int or not 0 < doc['peak_extra_bytes'] < 2**30:
        raise ValueError('context component peak is unbounded')
    return doc
