"""Independent GGUF library reader/dequantizer check; small CPU fixtures only."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import gguf
from gguf.quants import dequantize
from migration_bridge.round9.tests.test_candidate_bank import source_fixture
from migration_bridge.round9.tools.convert_candidate_bank import convert_layer

class CandidateGGUFTests(unittest.TestCase):
    def test_library_reopens_both_formats_and_all_projections(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=source_fixture(root)
            for fmt in ('Q8_0','MXFP4'):
                out=root/fmt;report=convert_layer(source,0,[0,1],fmt,out,source_identity={'scope':'TEST_FIXTURE'})
                reader=gguf.GGUFReader(str(out/report['file']))
                self.assertEqual(len(reader.tensors),3)
                for tensor in reader.tensors:
                    decoded=dequantize(tensor.data,tensor.tensor_type)
                    self.assertEqual(tuple(reversed(tensor.shape.tolist())),(2,32,32))
                    expected=[-1,-2] if 'down' in tensor.name else [2,4] if 'up' in tensor.name else [1,3]
                    for expert,value in enumerate(expected):
                        np.testing.assert_allclose(decoded.reshape(2,32,32)[expert],value,rtol=0,atol=.001)

    def test_nonconstant_block_decode_matches_independent_library(self):
        from migration_bridge.round9.runtime.lowbit import encode_block,decode_block,TYPES
        rng=np.random.default_rng(20260920)
        for fmt in ('MXFP4','Q8_0'):
            for scale in (.01,1,100):
                values=(rng.normal(size=320)*scale).astype(np.float32)
                raw=b''.join(encode_block(values[i:i+32],fmt) for i in range(0,len(values),32))
                expected=np.concatenate([decode_block(raw[i:i+(17 if fmt=='MXFP4' else 34)],fmt) for i in range(0,len(raw),17 if fmt=='MXFP4' else 34)])
                actual=dequantize(np.frombuffer(raw,dtype=np.uint8),gguf.GGMLQuantizationType(TYPES[fmt]))
                np.testing.assert_array_equal(actual,expected.astype(np.float32))

if __name__=='__main__':unittest.main()
