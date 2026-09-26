import unittest
from types import SimpleNamespace as NS

from migration_bridge.round9.runtime.dynamic import PlacementController
from migration_bridge.round9.runtime.prefill_end_boundary import PrefillEndBoundary
from migration_bridge.round9.runtime.route_stats import RouteStats


class Adapter:
    def __init__(self): self.copies = []; self.fail = False
    def synchronize(self): pass
    def copy_experts(self, layer, moves):
        if self.fail: raise RuntimeError('copy failed')
        self.copies.append((layer, moves))
    def publish(self, *args): pass


class PrefillEndBoundaryTests(unittest.TestCase):
    def setup_case(self):
        adapter = Adapter(); controller = PlacementController({0:(0,1)}, adapter, experts=4, resident=2)
        events = []; state = {'kv': 123, 'gdn': 'state', 'ple': 'history'}
        boundary = PrefillEndBoundary(controller, lambda req: dict(state), events.append)
        req = NS(rid='r', req_pool_idx=1, mamba_pool_idx=1, output_ids=[7],
                 origin_input_ids=[11],
                 done=False, finished=lambda: req.done)
        boundary.before([req]); stats = RouteStats(layers=1, experts=4, topk=2)
        stats.observe('r', 0, 'prefill', 0, 0, [[2,3]]); stats.finish('r','prefill',1)
        boundary.observe(0,[2,3])
        return boundary, req, adapter, state, dict(stats.snapshot(),request_id='r',generation=0), events

    def test_same_request_decode_epoch_and_end_does_not_migrate_twice(self):
        b,req,adapter,state,stats,events = self.setup_case()
        b.prefill_complete(stats)
        self.assertEqual(b.generation,1); self.assertEqual(b.state,'DECODE_ACTIVE')
        self.assertEqual(req.req_pool_idx,1); self.assertEqual(req.mamba_pool_idx,1)
        b.observe(0,[0,1])
        req.done=True; req.req_pool_idx=req.mamba_pool_idx=None; b.after()
        self.assertEqual(len(adapter.copies),1)
        self.assertEqual(b.controller._counts,{})
        req.rid='next';req.done=False;req.req_pool_idx=req.mamba_pool_idx=1
        b.before([req]);self.assertEqual(b.state,'PREFILL_ACTIVE')
        self.assertEqual(events[0]['first_output_token_generation'],0)

    def test_finished_or_aborted_first_token_skips_migration(self):
        b,req,adapter,state,stats,events=self.setup_case()
        req.done=True; req.req_pool_idx=req.mamba_pool_idx=None
        self.assertFalse(b.prefill_complete(stats));b.after()
        self.assertEqual(adapter.copies,[]);self.assertEqual(b.controller._counts,{})

    def test_explicit_cancel_discards_routes_and_returns_idle(self):
        b,req,adapter,state,stats,events=self.setup_case()
        req.done=True;req.req_pool_idx=req.mamba_pool_idx=None
        b.cancel(req)
        self.assertEqual(b.state,'IDLE')
        self.assertIsNone(b.request)
        self.assertEqual(adapter.copies,[])
        self.assertEqual(b.controller._counts,{})

    def test_live_state_mutation_or_copy_failure_poison(self):
        for failure in ('state','copy'):
            b,req,adapter,state,stats,events=self.setup_case()
            if failure=='copy': adapter.fail=True
            else:
                original=adapter.publish
                def mutate(*args):state['ple']='reset';original(*args)
                adapter.publish=mutate
            with self.assertRaises(RuntimeError):b.prefill_complete(stats)
            self.assertEqual(b.state,'BROKEN_RELOAD_REQUIRED')
            self.assertEqual(b.controller.state,'BROKEN_RELOAD_REQUIRED')

    def test_duplicate_or_unsampled_barrier_rejected(self):
        b,req,adapter,state,stats,events=self.setup_case()
        req.output_ids=[]
        with self.assertRaises(RuntimeError):b.prefill_complete(stats)
        self.assertEqual(adapter.copies,[])
        req.output_ids=[7];b.prefill_complete(stats)
        with self.assertRaises(RuntimeError):b.prefill_complete(stats)

    def test_hybrid_statistics_use_leased_original_route_counts(self):
        b,req,adapter,state,stats,events=self.setup_case()
        req.output_ids=[]
        self.assertIsNone(b.hybrid_statistics(topk=2))
        req.output_ids=[7]
        actual=b.hybrid_statistics(topk=2)
        self.assertEqual(actual['request_id'],'r')
        self.assertEqual(actual['generation'],0)
        self.assertEqual(actual['tokens'],1)
        self.assertEqual(actual['counts'][0],[0,0,1,1])
        b.prefill_complete(actual)
        self.assertEqual(b.state,'DECODE_ACTIVE')

    def test_hybrid_statistics_reject_incomplete_layer_or_route_conservation(self):
        b,req,adapter,state,stats,events=self.setup_case()
        b.controller._counts.clear()
        with self.assertRaises(ValueError):b.hybrid_statistics(topk=2)
        b.controller._counts[0]=[0,0,1,0]
        with self.assertRaises(ValueError):b.hybrid_statistics(topk=2)
        self.assertEqual(adapter.copies,[])

    def test_round11_timing_is_exclusive_and_emitted_only_at_request_drain(self):
        ticks = iter(range(1_000, 200_000, 1_000))
        clock = ticks.__next__
        adapter = Adapter()
        controller = PlacementController({0:(0,1)}, adapter, experts=4, resident=2)
        events = []
        state = {'kv': 123, 'gdn': 'state', 'ple': 'history'}
        boundary = PrefillEndBoundary(
            controller, lambda req: dict(state), events.append,
            round11_timing=True, clock=clock)
        req = NS(rid='timed', req_pool_idx=1, mamba_pool_idx=1, output_ids=[7],
                 origin_input_ids=[11], done=False, finished=lambda: req.done)
        boundary.before([req])
        boundary.observe(0,[2,3])
        statistics = boundary.hybrid_statistics(topk=2)
        boundary.prefill_complete(statistics)
        self.assertFalse(any(row.get('status') == 'ROUND11_DYNAMIC_TIMING' for row in events))

        req.done=True
        req.req_pool_idx=req.mamba_pool_idx=None
        boundary.after()
        timing_events = [row for row in events if row.get('status') == 'ROUND11_DYNAMIC_TIMING']
        self.assertEqual(len(timing_events), 1)
        timing = timing_events[0]['timing_ns']
        phases = {
            'route_statistics', 'consumer_drain_sync', 'state_fence_before',
            'selection', 'migration_copy_and_sync', 'mapping_publication',
            'state_fence_after', 'cleanup', 'total_exposed',
        }
        self.assertEqual(set(timing), phases)
        self.assertTrue(all(type(value) is int and value >= 0 for value in timing.values()))
        child = sum(value for name, value in timing.items() if name != 'total_exposed')
        self.assertLessEqual(child, timing['total_exposed'])
        statuses = [row.get('status') for row in events]
        self.assertLess(statuses.index('ROUND11_DYNAMIC_TIMING'),
                        statuses.index('PREFILL_END_REQUEST_DRAINED'))

    def test_round11_timing_is_opt_in(self):
        b,req,adapter,state,stats,events=self.setup_case()
        b.prefill_complete(stats)
        req.done=True
        req.req_pool_idx=req.mamba_pool_idx=None
        b.after()
        self.assertFalse(any(row.get('status') == 'ROUND11_DYNAMIC_TIMING' for row in events))


if __name__=='__main__':unittest.main()
