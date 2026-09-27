"""Real CPU torch tensor seam. No accelerator devices are needed or permitted."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from migration_bridge.round9.tests.test_candidate_bank import source_fixture
from migration_bridge.round9.runtime.dynamic import PlacementController
from migration_bridge.round9.runtime.dynamic_adapter import TorchKTEPAdapter, bind_controller

class DynamicTensorTests(unittest.TestCase):
    def test_original_promotion_and_inplace_native_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            source=source_fixture(Path(directory));mask=torch.tensor([True,False]);pointer=mask.data_ptr()
            wrapper=SimpleNamespace(gpu_experts_mask=mask)
            method=SimpleNamespace(wrapper=wrapper,gpu_experts_mask=mask.clone(),gpu_experts_mask_cuda=mask.clone(),
                logical_to_gpu_index=torch.tensor([0,-1],dtype=torch.int32),logical_to_gpu_index_cuda=torch.tensor([0,-1],dtype=torch.int32),
                gpu_index_to_logical=torch.tensor([0],dtype=torch.int32))
            layer=SimpleNamespace(w13_weight=torch.zeros(1,32,64,dtype=torch.bfloat16),w2_weight=torch.zeros(1,32,32,dtype=torch.bfloat16))
            adapter=TorchKTEPAdapter({0:layer},{0:method},source,complete_cpu_backing_ids=(0,1),host_staging_limit=32768)
            controller=bind_controller(adapter)
            generation=controller.begin('request')
            controller.observe('request',generation,0,[1,1])
            controller.finish('request',consumers_drained=True)
            controller.commit(controller.plan())
            self.assertTrue(torch.equal(layer.w13_weight[:,:,:32],torch.full_like(layer.w13_weight[:,:,:32],3)))
            self.assertTrue(torch.equal(layer.w13_weight[:,:,32:],torch.full_like(layer.w13_weight[:,:,32:],4)))
            self.assertTrue(torch.equal(layer.w2_weight,torch.full_like(layer.w2_weight,-2)))
            self.assertEqual(wrapper.gpu_experts_mask.data_ptr(),pointer)
            self.assertEqual(wrapper.gpu_experts_mask.tolist(),[False,True])
            self.assertEqual(method.logical_to_gpu_index.tolist(),[-1,0])
            self.assertEqual(method.gpu_index_to_logical.tolist(),[1])
            self.assertEqual(method._round9_generation,1)
            method._round9_generation=0
            method.logical_to_gpu_index[0]=0
            with self.assertRaises(ValueError):bind_controller(adapter)

    def test_incomplete_backing_and_unbounded_staging_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=source_fixture(Path(directory))
            with self.assertRaises(ValueError):TorchKTEPAdapter({}, {}, source,complete_cpu_backing_ids=(0,),host_staging_limit=32768)
            with self.assertRaises(ValueError):TorchKTEPAdapter({}, {}, source,complete_cpu_backing_ids=(0,1),host_staging_limit=1)

if __name__=='__main__':unittest.main()
