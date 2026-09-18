"""Static contracts only; these tests do not instantiate KTEP or execute NPU."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_static import profile
from round8_resource_guard import reject_reasons

class PrepTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'01_MODEL/config.json').read_text())['text_config']
    def test_architecture(self):
        self.assertEqual((self.c['num_experts'],self.c['num_experts_per_tok']), (256,8))
        self.assertEqual(self.c['layer_types'].count('linear_attention'),30)
        self.assertEqual(self.c['layer_types'].count('full_attention'),10)
    def test_payload(self):
        n=3*self.c['hidden_size']*self.c['moe_intermediate_size']
        self.assertEqual(n*256*40*4,120*2**30)
        self.assertEqual(n*64*40*2,15*2**30)
    def test_masks_and_mapping(self):
        for r in (16,32,64):
            for row in profile(self.c,r)['layers']:
                gpu,cpu=set(row['npu_ids']),set(row['cpu_ids'])
                self.assertFalse(gpu&cpu); self.assertEqual(gpu|cpu,set(range(256)))
                self.assertEqual(len(gpu),r)
                for slot,e in enumerate(row['npu_ids']):
                    self.assertEqual(row['logical_to_physical'][e],slot)
                self.assertTrue(all(row['logical_to_physical'][e]==-1 for e in cpu))
    def test_nested(self):
        a=[set(profile(self.c,r)['layers'][0]['npu_ids']) for r in (16,32,64)]
        self.assertTrue(a[0]<a[1]<a[2])
    def test_invalid_budget(self):
        with self.assertRaises(ValueError): profile(self.c,257)
    def test_guard_fail_closed(self):
        self.assertTrue(reject_reasons({}))
    def test_guard_collision(self):
        x=dict(inventory_age_seconds=0,round7_perf_active=False,round7_npus=[2],round7_cpus=list(range(64)),
            requested_npu=2,requested_cpus=list(range(64)),host_available_bytes=500*2**30,
            host_total_bytes=1000*2**30,predicted_private_bytes=150*2**30,swap_used_bytes=0,
            npu_free_bytes=60*2**30,predicted_npu_peak_bytes=40*2**30,disk_free_bytes=500*2**30,
            download_conversion_budget_bytes=300*2**30,other_npu_processes=[],same_host_inventory=True,
            physical_core_count=64)
        self.assertIn('NPU_BUSY',reject_reasons(x)); self.assertIn('CPU_COLLISION',reject_reasons(x))
        x.update(requested_npu=3,requested_cpus=list(range(64,128)))
        self.assertEqual(reject_reasons(x),[])

if __name__=='__main__': unittest.main()
