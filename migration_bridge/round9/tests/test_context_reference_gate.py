import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from migration_bridge.round9.runtime.context_reference_gate import require_context_reference_gate


class ContextReferenceGateTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1]/'evidence/physical15/20260921-ple-v4/result.json'
        self.receipt = json.loads(path.read_text())

    def validate(self, receipt):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            path.write_text(json.dumps(receipt))
            return require_context_reference_gate(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                source_identity=self.receipt['source_identity'], source_sha256=self.receipt['source_sha256'],
                torch_version=self.receipt['torch_version'],torch_npu_version=self.receipt['torch_npu_version'])

    def test_actual_narrow_receipt_passes(self):
        self.assertEqual(self.validate(self.receipt)['status'], 'PASS_CONTEXT_REFERENCE_QSA_COMPONENT')

    def test_scope_identity_numerics_and_drain_mutations_fail(self):
        mutations = [lambda d:d.update(native_fp8_qualified=True),
                     lambda d:d.update(physical_device=14),
                     lambda d:d['cases'].pop(),
                     lambda d:d['cases'][0].update(bf16_output_exact=False),
                     lambda d:d['cases'][0].update(fp8_vs_original_relative_l2=float('nan')),
                     lambda d:d['host_store_final'].update(active_leases=1),
                     lambda d:d['source_sha256'].update({'qsa.py':'0'*64})]
        for mutate in mutations:
            doc=copy.deepcopy(self.receipt);mutate(doc)
            with self.assertRaises(ValueError):self.validate(doc)

    def candidate_proof(self):
        return {
            'schema': 1,
            'status': 'PASS_ROUND12_QSA_SOURCE_BINDING',
            'candidate': 'QUERY_AWARE_VECTORIZED_SPARSE_GQA',
            'qualified_qsa_sha256': self.receipt['source_sha256']['qsa.py'],
            'candidate_qsa_sha256': (
                '0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35'),
            'candidate_qwen4_exp_sha256': (
                'd4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f'),
            'component_analysis_sha256': (
                '83ea9975210eed86ac487d8f0010afa2b663768b2fa2bef4930931a3e9dbd787'),
            'component_result_sha256': (
                '651fd396942b61bab45b73d34c6dcb66399a7690753e3617eac44fe7f92f0465'),
            'candidate_context_reference_runtime_sha256': (
                'eb51e7b0073d397673887b0e0fe6e41d745b8ee72a385faf7f1c28cf78b23fd2'),
            'candidate_context_reference_binding_sha256': (
                '97eb1b60d0ee24305ad1a052884b79a3045fa3574fcfc85e8b2c4ccfdbfc0c11'),
            'integration_component_result_sha256': (
                'b6532cd7d9cf6fe7d03d205f6f13e16ee7ebee07a70fe68f1c03d82d165c12d8'),
            'integration_component_snapshot_sha256': (
                '8aae5151490bcafd133e533e5955c74858788022c107baa36ebf95e58223c4c2'),
            'integration_evidence_archive_sha256': (
                '89e7a7d4ba98191efe00ae05385707535adfa6970167acc0a2eb3c91115b87f3'),
        }

    def validate_candidate(self, *, source_sha256=None, proof=None,
                           proof_sha256=None, consumer_sha256=None):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / 'receipt.json'
            receipt_path.write_text(json.dumps(self.receipt))
            proof_path = Path(directory) / 'binding.json'
            proof_path.write_text(json.dumps(proof or self.candidate_proof()))
            candidate_sources = dict(self.receipt['source_sha256'])
            candidate_sources['qsa.py'] = self.candidate_proof()['candidate_qsa_sha256']
            return require_context_reference_gate(
                receipt_path, hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                source_identity=self.receipt['source_identity'],
                source_sha256=source_sha256 or candidate_sources,
                torch_version=self.receipt['torch_version'],
                torch_npu_version=self.receipt['torch_npu_version'],
                qsa_source_binding_path=proof_path,
                qsa_source_binding_sha256=(
                    proof_sha256 or hashlib.sha256(proof_path.read_bytes()).hexdigest()),
                qsa_consumer_sha256=(
                    consumer_sha256 or self.candidate_proof()['candidate_qwen4_exp_sha256']),
                qsa_runtime_source_sha256=(
                    self.candidate_proof()['candidate_context_reference_runtime_sha256']),
                qsa_binding_source_sha256=(
                    self.candidate_proof()['candidate_context_reference_binding_sha256']),
            )

    def test_approved_qsa_tuple_preserves_original_oracle_and_passes(self):
        gate = self.validate_candidate()
        self.assertEqual(gate['status'], 'PASS_CONTEXT_REFERENCE_QSA_COMPONENT')
        self.assertEqual(gate['qsa_source_binding'], {
            'mode': 'APPROVED_CANDIDATE',
            'candidate': 'QUERY_AWARE_VECTORIZED_SPARSE_GQA',
            'qualified_qsa_sha256': self.receipt['source_sha256']['qsa.py'],
            'candidate_qsa_sha256': self.candidate_proof()['candidate_qsa_sha256'],
            'candidate_qwen4_exp_sha256': self.candidate_proof()['candidate_qwen4_exp_sha256'],
            'candidate_context_reference_runtime_sha256': (
                self.candidate_proof()['candidate_context_reference_runtime_sha256']),
            'candidate_context_reference_binding_sha256': (
                self.candidate_proof()['candidate_context_reference_binding_sha256']),
        })

    def test_unknown_qsa_hash_is_rejected_even_with_a_signed_proof(self):
        sources = dict(self.receipt['source_sha256'])
        sources['qsa.py'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'unapproved QSA source'):
            self.validate_candidate(source_sha256=sources)

    def test_candidate_binding_rejects_tuple_or_proof_hash_drift(self):
        mutations = [
            lambda d: d.update(component_analysis_sha256='0' * 64),
            lambda d: d.update(component_result_sha256='0' * 64),
            lambda d: d.update(candidate_qwen4_exp_sha256='0' * 64),
            lambda d: d.update(candidate='RENAMED'),
            lambda d: d.update(candidate_context_reference_runtime_sha256='0' * 64),
            lambda d: d.update(candidate_context_reference_binding_sha256='0' * 64),
            lambda d: d.update(integration_component_result_sha256='0' * 64),
        ]
        for mutate in mutations:
            proof = self.candidate_proof(); mutate(proof)
            with self.subTest(proof=proof):
                with self.assertRaisesRegex(ValueError, 'binding tuple mismatch'):
                    self.validate_candidate(proof=proof)
        with self.assertRaisesRegex(ValueError, 'binding receipt hash mismatch'):
            self.validate_candidate(proof_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'consumer source mismatch'):
            self.validate_candidate(consumer_sha256='0' * 64)

    def test_candidate_binding_cannot_hide_non_qsa_source_drift(self):
        sources = dict(self.receipt['source_sha256'])
        sources['qsa.py'] = self.candidate_proof()['candidate_qsa_sha256']
        sources['context_staging.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'non-QSA source mismatch'):
            self.validate_candidate(source_sha256=sources)
