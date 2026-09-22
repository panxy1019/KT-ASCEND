"""Negative control: rounding TP partials does not commute with reduction."""
import struct
import unittest

def bf16(value):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    return struct.unpack('<f',struct.pack('<I',(bits+0x7fff+((bits>>16)&1))&0xffff0000))[0]

class MergeContract(unittest.TestCase):
    def test_historical_counterexample(self):
        p0,p1=1.00390625,.00390625
        correct=bf16(p0+p1)
        wrong=bf16(bf16(p0)+bf16(p1))
        self.assertEqual(correct,1.0078125)
        self.assertEqual(wrong,1.0)
        self.assertNotEqual(wrong,correct)

if __name__=='__main__':unittest.main()
