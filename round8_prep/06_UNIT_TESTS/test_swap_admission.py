import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from round8_resource_guard import reject_reasons


class SwapAdmission(unittest.TestCase):
    def inventory(self):
        return dict(inventory_age_seconds=0,round7_perf_active=False,round7_npus=[],
            round7_cpus=[],requested_npu=2,requested_cpus=list(range(128,192)),
            host_available_bytes=980*2**30,host_total_bytes=1024*2**30,
            predicted_private_bytes=384*2**30,swap_used_bytes=48*2**20,
            npu_free_bytes=60*2**30,predicted_npu_peak_bytes=40*2**30,
            disk_free_bytes=2000*2**30,download_conversion_budget_bytes=150*2**30,
            other_npu_processes=[],same_host_inventory=True,physical_core_count=64)
    def test_residual_requires_measurement(self):
        self.assertIn('SWAP_ACTIVE_OR_UNVERIFIED',reject_reasons(self.inventory()))
    def test_idle_residual(self):
        x=self.inventory();x.update(swap_observation_seconds=30,host_swapin_delta=0,
                                   host_swapout_delta=0,cgroup_swap_current=0)
        self.assertEqual(reject_reasons(x),[])
        for key,value in [('swap_observation_seconds',29),('host_swapin_delta',1),
                          ('host_swapout_delta',1),('cgroup_swap_current',1)]:
            y=dict(x);y[key]=value
            self.assertIn('SWAP_ACTIVE_OR_UNVERIFIED',reject_reasons(y))
    def test_bounded_historical_readback_not_new_pressure(self):
        x=self.inventory();x.update(swap_observation_seconds=30,host_swapin_delta=16,
            host_page_size_bytes=65536,host_swapout_delta=0,cgroup_swap_current=0)
        self.assertEqual(reject_reasons(x),[])
        for key,value in [('host_swapin_delta',17),('host_swapin_delta',-1),
                          ('host_page_size_bytes',0),('host_swapout_delta',1),('cgroup_swap_current',1)]:
            y=dict(x);y[key]=value
            self.assertIn('SWAP_ACTIVE_OR_UNVERIFIED',reject_reasons(y))
