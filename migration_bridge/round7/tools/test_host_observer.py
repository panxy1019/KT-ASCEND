import unittest
from host_resource_observer import assess_window,process_ids,memory_blockers


class HostQuiet(unittest.TestCase):
    def test_actual_npu_pid_format(self):
        self.assertEqual(process_ids('Process id:274359 Process memory(MB):7880\nProcess id:273820'),[274359,273820])

    def test_full_clean_window(self):
        samples=[dict(time_unix=t,status='EXCLUSIVE') for t in range(0,101,5)]
        self.assertTrue(assess_window(samples,12,94)['formal_eligible'])

    def test_mid_request_contention_rejected(self):
        samples=[dict(time_unix=t,status='WAIT' if t==50 else 'EXCLUSIVE') for t in range(0,101,5)]
        self.assertFalse(assess_window(samples,12,94)['formal_eligible'])

    def test_missing_or_stale_coverage_rejected(self):
        self.assertFalse(assess_window([],12,94)['formal_eligible'])
        self.assertFalse(assess_window([dict(time_unix=t,status='EXCLUSIVE') for t in (0,100)],12,94)['formal_eligible'])

    def test_retained_v2_admission_allows_inactive_host_swap(self):
        mem=dict(MemTotal=2**40,MemAvailable=800*2**30,SwapTotal=8*2**30,SwapFree=8*2**30-262144)
        counters=dict(pswpin=0,pswpout=1)
        self.assertEqual(memory_blockers(mem,0,counters,counters),[])
        self.assertTrue(memory_blockers(mem,1,counters,counters))
        self.assertTrue(memory_blockers(mem,0,dict(pswpin=1,pswpout=1),counters))
