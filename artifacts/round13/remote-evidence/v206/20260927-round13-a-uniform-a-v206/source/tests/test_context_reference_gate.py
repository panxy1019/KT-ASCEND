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
