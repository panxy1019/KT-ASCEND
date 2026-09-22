import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bf16_resident_layout import resident_to_raw
from runtime_selection import select_sglang_runtime


class Tensor:
    def __init__(self, shape, fmt=2, device='npu', dtype='bf16'):
        self.shape=shape;self.ndim=len(shape);self.fmt=fmt
        self.device=SimpleNamespace(type=device);self.dtype=dtype
        self.calls=[]
    def detach(self):self.calls.append('detach');return self
    def transpose(self, a, b):
        self.calls.append(('transpose',a,b))
        shape=list(self.shape);shape[a],shape[b]=shape[b],shape[a];self.shape=tuple(shape)
        return self
    def contiguous(self):self.calls.append('contiguous');return self


class ResidentLayoutTests(unittest.TestCase):
    def test_private_resident_helper_explicit_hash_binding(self):
        root='/workspace/c64-dev/round7/new/sglang-python'
        wrapper=root+'/sglang/srt/layers/moe/kt_ep_wrapper.py'
        helper=root+'/sglang/srt/layers/moe/bf16_resident_layout.py'
        provenance=dict(sglang_runtime_root=root,files={wrapper:{'sha256':'wrapper'},helper:{'sha256':'helper'}},
                        bf16_context_port={'ported_sha256':'wrapper','resident_layout_helper_sha256':'helper'})
        self.assertEqual(select_sglang_runtime({'sglang_runtime_root':root},provenance),root)
        provenance['files'][helper]['sha256']='other'
        with self.assertRaisesRegex(ValueError,'RESIDENT_LAYOUT_HELPER_HASH_BINDING'):
            select_sglang_runtime({'sglang_runtime_root':root},provenance)
    def call(self, source, dst):
        def cast(tensor, fmt):
            tensor.calls.append(('full_tensor_format_cast',fmt));tensor.fmt=fmt;return tensor
        with patch.dict(sys.modules,torch=SimpleNamespace(bfloat16='bf16'),
                        torch_npu=SimpleNamespace(npu_format_cast=cast,get_npu_format=lambda t:t.fmt)):
            return resident_to_raw(source,dst)
    def test_full_nz_inference_normalized_before_slice(self):
        source=Tensor((8,4,12),29);dst=Tensor((64,12,4))
        out=self.call(source,dst)
        self.assertEqual(out.shape,(8,12,4))
        self.assertEqual(source.calls,['detach',('full_tensor_format_cast',2),('transpose',1,2),'contiguous'])
    def test_already_raw_still_converts_full_tensor(self):
        src=Tensor((8,12,4),29)
        self.call(src,Tensor((64,12,4)))
        self.assertNotIn(('transpose',1,2),src.calls)
    def test_invalid_boundaries_rejected_before_copy(self):
        for src,dst,error in [
            (Tensor((8,4,12),device='cuda'),Tensor((64,12,4)),'NPU_ONLY'),
            (Tensor((8,4,12),dtype='fp32'),Tensor((64,12,4)),'DTYPE'),
            (Tensor((4,12)),Tensor((64,12,4)),'RANK'),
            (Tensor((8,4,12)),Tensor((64,12,4),29),'NOT_BASE_FORMAT'),
            (Tensor((8,7,12)),Tensor((64,12,4)),'LAYOUT_SHAPE')]:
            with self.subTest(error=error),self.assertRaisesRegex(RuntimeError,error):self.call(src,dst)
            self.assertEqual(src.calls,[])
