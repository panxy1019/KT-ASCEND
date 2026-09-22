import unittest
import tempfile
from pathlib import Path
from host_resource_observer import ownership_group,group_swap
from host_resource_observer import assess_window,process_ids,memory_blockers,unified_group,in_owned_group,outside_usage,fresh_exclusive


class HostQuiet(unittest.TestCase):
    def test_v1_hybrid_ownership_not_unified_root(self):
        self.assertEqual(ownership_group('7:memory:/docker/owned\n0::/\n'),('/docker/owned','v1'))
        self.assertEqual(ownership_group('0::/docker/owned\n'),('/docker/owned','v2'))
        with self.assertRaises(ValueError):ownership_group('7:memory:/\n0::/\n')

    def test_host_v1_swap_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'memory/docker/owned';path.mkdir(parents=True)
            (path/'memory.stat').write_text('total_swap 17\n')
            self.assertEqual(group_swap('/docker/owned','v1',root),17)
            (path/'memory.stat').write_text('rss 17\n')
            with self.assertRaises(KeyError):group_swap('/docker/owned','v1',root)

    def test_entrance_requires_fresh_clean_state(self):
        state=dict(status='EXCLUSIVE',time_unix=100,blockers=[])
        self.assertTrue(fresh_exclusive(state,110))
        self.assertFalse(fresh_exclusive(state,121))
        self.assertFalse(fresh_exclusive(state,99))
        self.assertFalse(fresh_exclusive(dict(state,status='WAIT'),110))
        self.assertFalse(fresh_exclusive(dict(state,blockers=['swap']),110))

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

    def test_activity_ceases_after_interval_not_lifetime(self):
        mem=dict(MemTotal=2**40,MemAvailable=800*2**30)
        before=dict(pswpin=124,pswpout=905)
        active=dict(pswpin=161,pswpout=905)
        self.assertTrue(memory_blockers(mem,0,active,before))
        self.assertEqual(memory_blockers(mem,0,active,active),[])
