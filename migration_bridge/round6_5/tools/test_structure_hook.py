import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from r65_structure_hook import make_hook


@unittest.skipUnless(importlib.util.find_spec('torch'),'target torch unit test')
class StructureHook(unittest.TestCase):
    def run_case(self,bad=False):
        import torch
        with tempfile.TemporaryDirectory(prefix='r65-hook-test-',dir='/workspace/results') as temp:
            root=Path(temp);profile=root/'profile.json';out=root/'structure.jsonl'
            profile.write_text(json.dumps(dict(rows=[dict(layer=17,npu_ids=list(range(64)),cpu_ids=[])])))
            def weight():return SimpleNamespace(shape=(63 if bad else 64,8,8),dtype=torch.bfloat16,
                device=SimpleNamespace(type='npu'),numel=lambda:64*8*8,element_size=lambda:2)
            module=SimpleNamespace(layer_id=17,experts=SimpleNamespace(quant_method=object(),w13_weight=weight(),w2_weight=weight()))
            npu=SimpleNamespace(memory_allocated=lambda:1,memory_reserved=lambda:2,max_memory_allocated=lambda:3)
            with patch.object(torch,'npu',npu,create=True):
                hook=make_hook(dict(profile=str(profile),output=str(out)))
                hook(module,None,None);hook(module,None,None)
            self.assertEqual(len(out.read_text().splitlines()),1)
    def test_one_shot(self):self.run_case()
    def test_reject_bad_physical_count(self):
        with self.assertRaisesRegex(ValueError,'PHYSICAL_WEIGHT'):self.run_case(True)


if __name__=='__main__':unittest.main()
