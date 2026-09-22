import pathlib
import sys
import tempfile
import unittest
import torch
import gguf
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/'tools'))
from bf16_expert_bank import write_bank

class BankTests(unittest.TestCase):
 def test_exact_original_bits_axes_and_first_last_expert(self):
  torch.manual_seed(9)
  data={'gate':torch.randn(3,32,64).bfloat16(),'up':torch.randn(3,32,64).bfloat16(),
        'down':torch.randn(3,64,32).bfloat16()}
  data['gate'][0,0,0]=-0.0
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp)/'bank'
   row=write_bank(root,47,lambda n,e:data[n][e],experts=3,hidden=64,intermediate=32)
   self.assertEqual(row['exact_projection_instances'],9)
   self.assertEqual(list(root.glob('*.staging')),[])
   reader=gguf.GGUFReader(str(root/row['file']))
   for tensor in reader.tensors:
    name=tensor.name.split('ffn_')[1].split('_exps')[0]
    self.assertEqual(int(tensor.tensor_type),30)
    self.assertTrue(np.array_equal(tensor.data.view(np.uint16).reshape(data[name].shape),data[name].view(torch.uint16).numpy()))
   with self.assertRaises(FileExistsError):write_bank(root,47,lambda n,e:data[n][e],experts=3,hidden=64,intermediate=32)
 def test_invalid_source_does_not_produce_manifest(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp)/'bank'
   with self.assertRaises(ValueError):write_bank(root,0,lambda n,e:torch.zeros(32,64),experts=1,hidden=64,intermediate=32)
   self.assertFalse((root/'manifest.json').exists())

if __name__=='__main__':unittest.main()
