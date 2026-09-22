"""Real wrapper configuration gate; no model bank or worker allocation."""
import tempfile
import unittest
from unittest.mock import patch
import torch
from kt_kernel.utils import llamafile as module

class WrapperConfigurationTests(unittest.TestCase):
 def construct(self,types,pools=2,intermediate=640):
  class Loader:
   def __init__(self,path):pass
   def get_undequanted_tensor_and_ggml_type(self,key):
    return None,types[['gate','up','down'].index(key.split('ffn_')[1].split('_exps')[0])]
  with tempfile.TemporaryDirectory() as path,patch.object(module,'GGUFLoader',Loader),patch.object(module.BaseMoEWrapper,'__init__',return_value=None):
   return module.LlamafileMoEWrapper(layer_idx=0,num_experts=512,num_experts_per_tok=10,
    hidden_size=2560,moe_intermediate_size=intermediate,gpu_experts_mask=torch.zeros(512,dtype=torch.bool),
    cpuinfer_threads=64,threadpool_count=pools,weight_path=path,chunked_prefill_size=32)
 def test_bf16_c64_320_partitions_admitted(self):
  wrapper=self.construct([30]*3)
  self.assertEqual(wrapper.output_dtype,torch.float32)
 def test_f32_existing_c64_stays_admitted(self):
  self.construct([0]*3)
 def test_bf16_requires_compiled_native_capability(self):
  with patch.object(module,'_BF16_F32_TAIL_SUPPORTED',False,create=True):
   with self.assertRaises(ValueError):self.construct([30]*3)
   self.construct([0]*3)
 def test_unaligned_or_mixed_storage_rejected(self):
  for types,pools,width in (([30]*3,3,640),([30]*3,2,672),([30,0,30],2,640),([8]*3,2,640)):
   with self.subTest(types=types,pools=pools,width=width),self.assertRaises(ValueError):
    self.construct(types,pools,width)

if __name__=='__main__':unittest.main()
