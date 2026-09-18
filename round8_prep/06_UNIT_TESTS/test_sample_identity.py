import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('sample', Path(__file__).resolve().parents[1] / 'tools/inspect_sample_shards.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SampleIdentityTests(unittest.TestCase):
    def reject(self, payload, error, change=None):
        with tempfile.TemporaryDirectory() as directory:
            name = 'model.safetensors-00006-of-00014.safetensors'
            (Path(directory) / name).write_bytes(payload)
            entry = dict(name=name, size=len(payload), hub_lfs_sha256=hashlib.sha256(payload).hexdigest())
            if change:
                entry.update(change)
            with self.assertRaisesRegex(ValueError, error):
                MODULE.inspect(Path(directory), dict(shards=[entry]))

    def test_reject_size(self):
        self.reject(b'abcd', 'size mismatch', dict(size=10))

    def test_reject_hash(self):
        self.reject(b'abcd', 'SHA mismatch', dict(hub_lfs_sha256='0'*64))

    def test_reject_header_length(self):
        self.reject(struct.pack('<Q', 100000000), 'invalid header')

    def test_reject_non_bf16(self):
        header = json.dumps({'model.language_model.layers.0.mlp.experts.gate_up_proj':
                             dict(dtype='F32', shape=[256, 1024, 2048], data_offsets=[0, 8])}).encode()
        self.reject(struct.pack('<Q', len(header)) + header + b'12345678', 'unexpected packed shape/dtype')
