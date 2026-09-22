import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen_bringup_hooks import make_shared_part, _shared_parts

class Tensor:
    def detach(self):return self
    def clone(self):return Tensor()

class SharedCapture(unittest.TestCase):
    def test_duplicate_before_parent_and_ignore_after_parent(self):
        _shared_parts.clear()
        hook=make_shared_part(dict(layer=0,kind='shared'))
        hook(None,None,Tensor())
        self.assertIn('shared',_shared_parts[0])
        with self.assertRaises(AssertionError):hook(None,None,Tensor())
        _shared_parts.pop(0)
        hook(None,None,Tensor())
        self.assertEqual(_shared_parts,{})
