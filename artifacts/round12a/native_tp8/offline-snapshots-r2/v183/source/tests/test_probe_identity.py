import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

P=pathlib.Path(__file__).parents[1]/'tools/probe_identity.py'
class IdentityTests(unittest.TestCase):
    def test_receipt_binding_and_wrong_inputs(self):
        spec=importlib.util.spec_from_file_location('probe_identity',P)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)/'model';root.mkdir()
            (pathlib.Path(td)/'other').mkdir()
            raw=b'{"weight_map":{}}'
            (root/'model.safetensors.index.json').write_bytes(raw)
            (root/'config.json').write_bytes(b'{}')
            records=[{'path':n,'size':(root/n).stat().st_size,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest(),'status':'PASS'} for n in ('config.json','model.safetensors.index.json')]
            receipt=pathlib.Path(td)/'receipt.json'
            receipt.write_text(json.dumps({'status':'PASS','revision':'fixed','manifest_sha256':'manifest','files':records}))
            sha=hashlib.sha256(receipt.read_bytes()).hexdigest()
            def call(**kw):
                args=dict(model=root,receipt=receipt,expected_model=root,expected_receipt_sha256=sha,revision='fixed',manifest_sha256='manifest');args.update(kw)
                return mod.verify_probe_identity(**args)
            self.assertEqual(call()['receipt_sha256'],sha)
            for kw in ({'expected_receipt_sha256':'bad'},{'revision':'wrong'},{'manifest_sha256':'wrong'},{'expected_model':root.parent/'other'}):
                with self.subTest(kw=kw),self.assertRaises(ValueError):call(**kw)
            (root/'config.json').write_bytes(b'[]')
            with self.assertRaises(ValueError):call()
            (root/'config.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):call()
