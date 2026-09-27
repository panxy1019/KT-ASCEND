"""Explicit ARM CPU-only test, requires NumPy; never silently skipped."""
import unittest
import numpy as np
from migration_bridge.round9.runtime.lowbit import encode_block,decode_block
from migration_bridge.round9.runtime.lowbit_numpy import encode_blocks,decode_blocks
class VectorCodecTests(unittest.TestCase):
    def test_byte_identical_to_scalar_across_chunks(self):
        rng=np.random.default_rng(20260920)
        x=(rng.normal(size=(1025,32))*rng.choice([.001,.1,1.,32.],size=(1025,1))).astype(np.float32)
        x[:4]=np.asarray([np.zeros(32),np.arange(-16,16),np.ones(32),-np.ones(32)])
        for fmt in ('Q8_0','MXFP4'):
            raw=encode_blocks(x,fmt)
            self.assertEqual(raw.tobytes(),b''.join(encode_block(v,fmt) for v in x))
            expected=np.asarray([decode_block(v.tobytes(),fmt) for v in raw],dtype=np.float32)
            np.testing.assert_array_equal(decode_blocks(raw,fmt),expected)
    def test_full_gguf_is_byte_identical(self):
        import tempfile
        from pathlib import Path
        from migration_bridge.round9.tests.test_candidate_bank import source_fixture
        from migration_bridge.round9.tools.convert_candidate_bank import convert_layer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=source_fixture(root)
            for fmt in ('Q8_0','MXFP4'):
                scalar=convert_layer(source,0,[0,1],fmt,root/(fmt+'s'),source_identity={})
                vector=convert_layer(source,0,[0,1],fmt,root/(fmt+'v'),source_identity={},encoder='numpy')
                self.assertEqual(scalar['sha256'],vector['sha256'])
                self.assertEqual(scalar['projection_identities'],vector['projection_identities'])

    def test_reject_bad_values_and_scales(self):
        for fmt in ('Q8_0','MXFP4'):
            for x in (np.ones(31),np.full(32,np.nan),np.full(32,np.inf)):
                with self.assertRaises(ValueError):encode_blocks(x,fmt)
        for x in (1e20,1e-20):
            with self.assertRaises(ValueError):encode_blocks(np.full(32,x),'Q8_0')
if __name__=='__main__':unittest.main()
