import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen_cmmlu_reference import score

class ChoiceScore(unittest.TestCase):
 def test_id_not_response_order(self):
  self.assertEqual(score([[-4,13,None],[-1,12,None],[-5,11,None],[-9,10,None]],[10,11,12,13]),'C')
 def test_tie(self):
  self.assertEqual(score([[-1,i,None] for i in [13,12,11,10]],[10,11,12,13]),'A')
 def test_invalid(self):
  with self.assertRaises(AssertionError):score([[float('nan'),i,None] for i in range(4)],list(range(4)))
  with self.assertRaises(AssertionError):score([[-1,0,None]]*4,list(range(4)))
