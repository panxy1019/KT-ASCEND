import math
import struct
import unittest
from migration_bridge.round9.runtime.lowbit import encode_block, decode_block, bf16_values

class LowbitTests(unittest.TestCase):
    def test_q8_known_bytes_roundtrip(self):
        x=list(range(-16,16)); x[0]=-127
        raw=encode_block(x,'Q8_0')
        self.assertEqual(raw,struct.pack('<e32b',1,*x))
        self.assertEqual(decode_block(raw,'Q8_0'),x)

    def test_mxfp4_half_block_order_and_scale(self):
        values=[0,.5,1,1.5,2,3,4,6]*2+[-0.,-.5,-1,-1.5,-2,-3,-4,-6]*2
        raw=encode_block(values,'MXFP4')
        self.assertEqual(raw,bytes([127])+bytes([0,0x91,0xa2,0xb3,0xc4,0xd5,0xe6,0xf7])*2)
        self.assertEqual(decode_block(raw,'MXFP4'),values)

    def test_zero_tie_and_invalid(self):
        self.assertEqual(encode_block([0]*32,'MXFP4'),bytes(17))
        self.assertEqual(encode_block([0]*32,'Q8_0'),bytes(34))
        for fmt in ('Q8_0','MXFP4'):
            for x in ([0]*31,[float('nan')]*32,[float('inf')]*32):
                with self.assertRaises(ValueError):encode_block(x,fmt)
            with self.assertRaises(ValueError):decode_block(b'',fmt)
        for x in (1e20,1e-20):
            with self.assertRaises(ValueError):encode_block([x]*32,'Q8_0')
        with self.assertRaises(ValueError):decode_block(bytes([255])+bytes(16),'MXFP4')

    def test_original_bf16_bytes(self):
        self.assertEqual(bf16_values(bytes.fromhex('803f00c00000')), [1,-2,0])
        with self.assertRaises(ValueError):bf16_values(b'x')

if __name__=='__main__':unittest.main()
