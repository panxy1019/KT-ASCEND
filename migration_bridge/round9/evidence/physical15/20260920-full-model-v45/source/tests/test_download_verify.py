import hashlib
import importlib.util
import pathlib
import json
import struct
import tempfile
import unittest

TOOL = pathlib.Path(__file__).parents[1] / 'tools' / 'verify_download.py'


class VerifyTests(unittest.TestCase):
    def test_verifier_exists(self):
        self.assertTrue(TOOL.exists(), 'download identity verifier not implemented')

    def load(self):
        if not TOOL.exists():
            self.skipTest('implementation missing; existence test is red')
        spec = importlib.util.spec_from_file_location('verify_download', TOOL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_content_not_just_size(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            p = root / 'weight'
            p.write_bytes(b'abc')
            row = {'path': 'weight', 'size': 3, 'sha256':
                   'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}
            self.assertEqual(m.verify_file(root, row)['status'], 'PASS')
            p.write_bytes(b'abd')
            with self.assertRaises(ValueError):
                m.verify_file(root, row)

    def test_git_blob_and_missing(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'small').write_bytes(b'test\n')
            row = {'path': 'small', 'size': 5, 'sha256': None,
                   'git_blob': '9daeafb9864cf43055ae93beb0afd6c7d144bfa4'}
            self.assertEqual(m.verify_file(root, row)['status'], 'PASS')
            row['path'] = 'absent'
            with self.assertRaises((ValueError, FileNotFoundError)):
                m.verify_file(root, row)

    def test_path_escape_rejected(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            for name in ('../outside', '/etc/passwd'):
                with self.assertRaises(ValueError):
                    m.verify_file(pathlib.Path(d), {'path': name, 'size': 1})

    def test_tensor_census_checks_index_and_offsets(self):
        m = self.load()
        self.assertTrue(hasattr(m, 'tensor_census'), 'header/index audit missing')
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            header = json.dumps({'x': {'dtype': 'BF16', 'shape': [2],
                                      'data_offsets': [0, 4]}}).encode()
            (root / 'a.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
            index = {'weight_map': {'x': 'a.safetensors'}, 'metadata': {'total_size': 4}}
            result = m.tensor_census(root, index)
            self.assertEqual(result['tensor_count'], 1)
            self.assertEqual(result['payload_bytes'], 4)
            index['weight_map'] = {'wrong': 'a.safetensors'}
            with self.assertRaises(ValueError):
                m.tensor_census(root, index)


if __name__ == '__main__':
    unittest.main()
