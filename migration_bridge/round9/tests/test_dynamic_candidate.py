import unittest
from migration_bridge.round9.runtime.dynamic import PlacementController

class RecordingAdapter:
    def __init__(self,fail_at=None):self.events=[];self.fail_at=fail_at
    def synchronize(self):self.events.append(('sync',))
    def copy_expert(self,layer,slot,expert):
        self.events.append(('copy',layer,slot,expert))
        if self.fail_at=='copy':raise OSError('partial device copy')
    def publish(self,layer,slots,generation):
        self.events.append(('publish',layer,slots,generation))
        if self.fail_at=='publish':raise OSError('partial mapping publish')

class DynamicCandidateTests(unittest.TestCase):
    def test_layer_batch_is_used_and_failure_prevents_publication(self):
        class BatchAdapter(RecordingAdapter):
            def copy_experts(self,layer,moves):
                self.events.append(('batch',layer,moves))
                if self.fail_at=='batch':raise OSError('partial layer copy')
            def copy_expert(self,*args):raise AssertionError('must use layer batch')
        for failure in (None,'batch'):
            adapter=BatchAdapter(failure);c=self.controller(adapter)
            self.observe(c,0,[3,2,3,2]);plans=c.plan()
            if failure:
                with self.assertRaises(OSError):c.commit(plans)
                self.assertEqual(c.state,'BROKEN_RELOAD_REQUIRED')
                self.assertFalse(any(e[0]=='publish' for e in adapter.events))
            else:
                c.commit(plans);self.assertEqual(c.generation,1)
                self.assertIn(('batch',0,((0,2),(1,3))),adapter.events)

    def observe(self,c,layer,ids):
        generation=c.begin("sample")
        try:c.observe("sample",generation,layer,ids)
        finally:c.finish("sample",consumers_drained=True)

    def controller(self,adapter=None):
        return PlacementController({0:(0,1),1:(2,3)},adapter or RecordingAdapter(),experts=4,resident=2)

    def test_preserve_slots_and_deterministic_frequency(self):
        c=self.controller();self.observe(c,0,[3,3,3,1,1,0])
        plan=c.plan();self.assertEqual(plan[0].slots,(3,1))
        self.assertEqual(plan[0].moves,((0,3),))
        c.commit(plan);self.assertEqual(c.generation,1)
        self.assertEqual(c.adapter.events,[('sync',),('copy',0,0,3),('sync',),('publish',0,(3,1),1),('sync',)])
        self.assertEqual(c.begin('request2'),1);c.finish('request2',consumers_drained=True)

    def test_active_request_and_undrained_state_block(self):
        c=self.controller();self.observe(c,0,[3]);plan=c.plan();c.begin('a')
        with self.assertRaises(RuntimeError):c.commit(plan)
        with self.assertRaises(RuntimeError):c.plan()
        with self.assertRaises(RuntimeError):c.begin('b')
        with self.assertRaises(RuntimeError):c.finish('a',consumers_drained=False)
        c.finish('a',consumers_drained=True);c.commit(plan)
        with self.assertRaises(ValueError):c.commit(plan)

    def test_partial_mutation_failure_poison_no_metadata_rollback(self):
        for failure in ('copy','publish'):
            c=self.controller(RecordingAdapter(failure));self.observe(c,0,[3]);plan=c.plan()
            with self.assertRaises(OSError):c.commit(plan)
            self.assertEqual(c.state,'BROKEN_RELOAD_REQUIRED')
            self.assertEqual(c.generation,0)
            with self.assertRaises(RuntimeError):c.begin('new')
            with self.assertRaises(RuntimeError):c.commit(plan)

    def test_duplicate_invalid_and_noop(self):
        with self.assertRaises(ValueError):PlacementController({0:(1,1)},RecordingAdapter(),experts=4,resident=2)
        c=self.controller()
        for routes in ([True],[-1],[4]):
            with self.assertRaises(ValueError):self.observe(c,0,routes)
        self.assertEqual(c.plan(),());self.assertEqual(c.commit(()),0)
        self.assertFalse(c.adapter.events)

    def test_unleased_and_stale_route_observations_rejected(self):
        c=self.controller()
        with self.assertRaises(RuntimeError):c.observe('late',0,0,[3])
        generation=c.begin('active')
        with self.assertRaises(RuntimeError):c.observe('old',generation,0,[3])
        with self.assertRaises(RuntimeError):c.observe('active',generation-1,0,[3])
        c.observe('active',generation,0,[3])
        c.finish('active',consumers_drained=True)
        with self.assertRaises(RuntimeError):c.observe('active',generation,0,[3])

    def test_new_observation_invalidates_pending_plan(self):
        c=self.controller();self.observe(c,0,[3]);plan=c.plan()
        self.observe(c,0,[2,2,2])
        with self.assertRaises(ValueError):c.commit(plan)

    def test_partial_plan_rejected_and_noop_closes_statistics_window(self):
        c=self.controller();self.observe(c,0,[3]);self.observe(c,1,[0])
        plans=c.plan();self.assertEqual(len(plans),2)
        with self.assertRaises(ValueError):c.commit(plans[:1])
        self.assertFalse(c.adapter.events)
        c=self.controller();self.observe(c,0,[0,1]*100)
        self.assertEqual(c.plan(),());c.commit(())
        self.observe(c,0,[3]);self.assertEqual(len(c.plan()),1)

if __name__=='__main__':unittest.main()
