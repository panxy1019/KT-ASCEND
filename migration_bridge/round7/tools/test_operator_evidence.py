import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from operator_evidence import bind_operator_evidence


class OperatorEvidenceTest(unittest.TestCase):
    def fixture(self,root):
        cpu=root/'cpu';cpu.mkdir();native=cpu/'native.so';native.write_bytes(b'fixed')
        digest=hashlib.sha256(native.read_bytes()).hexdigest()
        prov=dict(cpu_native_sha256=digest,cpu_runtime_root=str(cpu),model_revision='same',
            gguf={'sha256':'same'},torch='same',torch_npu='same',
            files={str(native):dict(sha256=digest,size=5)})
        source=root/'old';target=root/'new';source.mkdir();target.mkdir()
        for p in (source,target):(p/'provenance.json').write_text(json.dumps(prov))
        (source/'operator-representatives').mkdir()
        (source/'operator-representatives/summary.json').write_text(json.dumps(
            dict(status='PASS',gate=.01,runs=[dict(max_rel_l2=.002)])))
        return source,target,native

    def test_explicit_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,_=self.fixture(Path(tmp))
            self.assertEqual(bind_operator_evidence(target,source)['scope'],'REUSED_UNCHANGED_CPU_ARITHMETIC')

    def test_changed_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,native=self.fixture(Path(tmp));native.write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError,'CPU_ARTIFACT_DRIFT'):bind_operator_evidence(target,source)

    def test_model_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,_=self.fixture(Path(tmp));p=target/'provenance.json'
            d=json.loads(p.read_text());d['model_revision']='changed';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'PROVENANCE_DRIFT'):bind_operator_evidence(target,source)
