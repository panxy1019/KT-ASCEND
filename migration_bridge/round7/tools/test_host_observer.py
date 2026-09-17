import unittest
from host_resource_observer import assess_window,process_ids,memory_blockers,unified_group,in_owned_group,outside_usage


class HostQuiet(unittest.TestCase):
    def test_exact_cgroup_scope_not_prefix_collision(self):
        group='/system.slice/docker-owned.scope'
        self.assertTrue(in_owned_group(group,group))
        self.assertTrue(in_owned_group(group+'/child',group))
        self.assertFalse(in_owned_group(group+'-other',group))
        with self.assertRaises(ValueError):unified_group('0::/\n')
        self.assertEqual(unified_group('0::/\n',allow_root=True),'/')
        self.assertEqual(unified_group('0::'+group+'\n'),group)

    def test_new_owned_worker_not_misclassified_by_stale_top(self):
        group='/docker/owned'
        previous={1:dict(ticks=100,start_ticks=10,cgroup=group,command='python')}
        current={1:dict(ticks=5100,start_ticks=10,cgroup=group,command='python'),
                 2:dict(ticks=200,start_ticks=20,cgroup=group,command='sglang')}
        self.assertEqual(outside_usage(current,previous,group,999,100,5),[])

    def test_real_outside_usage_and_reused_pid(self):
        old=dict(ticks=100,start_ticks=10,cgroup='/business',command='/bin/worker')
        new=dict(old,ticks=1100)
        usage=outside_usage({1:new},{1:old},'/docker/owned',999,100,5)
        self.assertEqual(usage[0]['cores'],2)
        self.assertEqual(usage[0]['executable'],'worker')
        self.assertEqual(outside_usage({1:dict(new,start_ticks=11)},{1:old},'/docker/owned',999,100,5),[])

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
