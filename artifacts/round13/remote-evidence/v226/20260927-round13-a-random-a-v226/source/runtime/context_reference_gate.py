"""Admission for the synthetic CPU-record / BF16-NPU QSA component only."""
import hashlib
import json
import math
from pathlib import Path

SOURCE_FILES = ('context_profile.py', 'context_page_store.py',
                'kv_quant_reference.py', 'context_staging.py', 'qsa.py')

APPROVED_QSA_BINDING = {
    'schema': 1,
    'status': 'PASS_ROUND12_QSA_SOURCE_BINDING',
    'candidate': 'QUERY_AWARE_VECTORIZED_SPARSE_GQA',
    'qualified_qsa_sha256': (
        '76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c'),
    'candidate_qsa_sha256': (
        '0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35'),
    'candidate_qwen4_exp_sha256': (
        '83464deecd99e20b94da135337caf4a80f6696a8ead9684492ba6cbcad1263c0'),
    'component_analysis_sha256': (
        '83ea9975210eed86ac487d8f0010afa2b663768b2fa2bef4930931a3e9dbd787'),
    'component_result_sha256': (
        '651fd396942b61bab45b73d34c6dcb66399a7690753e3617eac44fe7f92f0465'),
    'candidate_context_reference_runtime_sha256': (
        '0b6e6a7fad90e3856ffdcbea233a13baaf1bb342837c68ff344579370773c028'),
    'candidate_context_reference_binding_sha256': (
        '97eb1b60d0ee24305ad1a052884b79a3045fa3574fcfc85e8b2c4ccfdbfc0c11'),
    'integration_component_result_sha256': (
        'b6532cd7d9cf6fe7d03d205f6f13e16ee7ebee07a70fe68f1c03d82d165c12d8'),
    'integration_component_snapshot_sha256': (
        '8aae5151490bcafd133e533e5955c74858788022c107baa36ebf95e58223c4c2'),
    'integration_evidence_archive_sha256': (
        '89e7a7d4ba98191efe00ae05385707535adfa6970167acc0a2eb3c91115b87f3'),
}


def _bind_qsa_source(receipt_sources, actual_sources, *, binding_path,
                     binding_sha256, consumer_sha256, runtime_source_sha256,
                     binding_source_sha256):
    """Return receipt-compatible sources after validating one approved QSA tuple."""
    if set(actual_sources) != set(SOURCE_FILES):
        raise ValueError('non-QSA source mismatch')
    for name in SOURCE_FILES:
        if name != 'qsa.py' and actual_sources[name] != receipt_sources.get(name):
            raise ValueError('non-QSA source mismatch')
    actual_qsa = actual_sources['qsa.py']
    qualified_qsa = receipt_sources.get('qsa.py')
    binding_supplied = any(value is not None for value in (
        binding_path, binding_sha256, consumer_sha256,
        runtime_source_sha256, binding_source_sha256))
    if actual_qsa == qualified_qsa:
        if binding_supplied:
            raise ValueError('QSA source binding supplied for qualified source')
        return dict(actual_sources), None
    if actual_qsa != APPROVED_QSA_BINDING['candidate_qsa_sha256']:
        raise ValueError('unapproved QSA source')
    if (binding_path is None or binding_sha256 is None or consumer_sha256 is None
            or runtime_source_sha256 is None or binding_source_sha256 is None):
        raise ValueError('unapproved QSA source')
    raw = Path(binding_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding_sha256:
        raise ValueError('QSA binding receipt hash mismatch')
    proof = json.loads(raw)
    if proof != APPROVED_QSA_BINDING:
        raise ValueError('QSA binding tuple mismatch')
    if proof['qualified_qsa_sha256'] != qualified_qsa:
        raise ValueError('QSA binding tuple mismatch')
    if consumer_sha256 != proof['candidate_qwen4_exp_sha256']:
        raise ValueError('QSA consumer source mismatch')
    if runtime_source_sha256 != proof['candidate_context_reference_runtime_sha256']:
        raise ValueError('QSA runtime source mismatch')
    if binding_source_sha256 != proof['candidate_context_reference_binding_sha256']:
        raise ValueError('QSA binding source mismatch')
    validated_sources = dict(actual_sources)
    validated_sources['qsa.py'] = qualified_qsa
    return validated_sources, {
        'mode': 'APPROVED_CANDIDATE',
        'candidate': proof['candidate'],
        'qualified_qsa_sha256': qualified_qsa,
        'candidate_qsa_sha256': actual_qsa,
        'candidate_qwen4_exp_sha256': consumer_sha256,
        'candidate_context_reference_runtime_sha256': runtime_source_sha256,
        'candidate_context_reference_binding_sha256': binding_source_sha256,
    }


def require_context_reference_gate(path, sha256, *, source_identity,
                                   source_sha256, torch_version, torch_npu_version,
                                   qsa_source_binding_path=None,
                                   qsa_source_binding_sha256=None,
                                   qsa_consumer_sha256=None,
                                   qsa_runtime_source_sha256=None,
                                   qsa_binding_source_sha256=None):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('context component receipt hash mismatch')
    doc = json.loads(raw)
    validated_sources, qsa_binding = _bind_qsa_source(
        doc.get('source_sha256', {}), source_sha256,
        binding_path=qsa_source_binding_path,
        binding_sha256=qsa_source_binding_sha256,
        consumer_sha256=qsa_consumer_sha256,
        runtime_source_sha256=qsa_runtime_source_sha256,
        binding_source_sha256=qsa_binding_source_sha256)
    if (doc.get('schema') != 1 or type(doc.get('schema')) is not int or
            doc.get('status') != 'PASS_CONTEXT_REFERENCE_QSA_COMPONENT' or
            doc.get('scope') != 'SYNTHETIC_KV_CPU_RECORD_BF16_NPU_QSA_NOT_MODEL_OR_HICACHE' or
            doc.get('physical_device') != 15 or doc.get('source_identity') != source_identity or
            set(validated_sources) != set(SOURCE_FILES) or
            doc.get('source_sha256') != validated_sources or
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
    if qsa_binding is not None:
        doc['qsa_source_binding'] = qsa_binding
    return doc
