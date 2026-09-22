import unittest
from summarize_performance_resources import describe


class ResourceDescription(unittest.TestCase):
    def test_exact_owned_scope_and_errors(self):
        row = dict(time_unix=2, status='SAMPLED', owned_guard_paths={'1':'/r/p/guard'},
                   summed_owned_rss_bytes=20, owned_hbm_mb_by_pid={'2':30},
                   other_process_hbm_mb=7, host_load_average=[3,4,5])
        other = dict(row, time_unix=3, owned_guard_paths={'1':'/other/guard'})
        error = dict(time_unix=4, status='MONITOR_ERROR', exception='read failure')
        result = describe([row,other,error,dict(row,time_unix=9)],1,5,'/r/p/guard')
        self.assertEqual(result['samples'],3)
        self.assertEqual(result['owned_samples'],1)
        self.assertEqual(result['other_npu0_hbm_mb_range'],[7,7])
        self.assertEqual(result['monitor_errors'],[error])

    def test_empty_is_not_zero_memory(self):
        result = describe([],1,5,'/r/p/guard')
        self.assertEqual(result['owned_samples'],0)
        self.assertIsNone(result['owned_hbm_mb_range'])


if __name__ == '__main__':
    unittest.main()
