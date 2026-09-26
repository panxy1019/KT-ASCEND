import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from migration_bridge.round9.runtime.prefill_binding import prefill_after, prefill_model_ready
from migration_bridge.round9.runtime.prefill_end_boundary import PrefillEndBoundary


class PrefillBindingTests(unittest.TestCase):
    def test_hybrid_after_commits_boundary_route_statistics(self):
        boundary = object.__new__(PrefillEndBoundary)
        boundary.state = 'PREFILL_ACTIVE'
        calls = []
        boundary.hybrid_statistics = lambda **kwargs: calls.append(('statistics', kwargs)) or {'tokens': 64}
        boundary.prefill_complete = lambda statistics: calls.append(('complete', statistics))
        runtime = NS(after=lambda: None)
        scheduler = NS(
            tp_worker=NS(model_runner=NS(_round9_prefill=runtime)),
            _round9_dynamic_boundary=boundary,
        )
        self.assertEqual(prefill_after(scheduler), {'tokens': 64})
        self.assertEqual(calls, [
            ('statistics', {'topk': 10}),
            ('complete', {'tokens': 64}),
        ])

    def test_hybrid_after_does_not_recommit_during_decode(self):
        boundary = object.__new__(PrefillEndBoundary)
        boundary.state = 'DECODE_ACTIVE'
        boundary.hybrid_statistics = lambda **kwargs: self.fail('decode cannot republish prefill')
        boundary.prefill_complete = lambda statistics: self.fail('decode cannot migrate')
        scheduler = NS(
            tp_worker=NS(model_runner=NS(_round9_prefill=NS(after=lambda: None))),
            _round9_dynamic_boundary=boundary,
        )
        self.assertIsNone(prefill_after(scheduler))

    def test_timeline_binds_only_after_private_transport_preallocation(self):
        source = inspect.getsource(prefill_model_ready)
        self.assertLess(
            source.index('transport = layerwise.LayerwiseTransport(backend, reservation)'),
            source.index("bind_expert_timeline(backend, binding['source_identity'], args.model_path)"),
        )

    def test_timeline_wraps_outer_transport_without_changing_gated_core(self):
        from migration_bridge.round9.runtime.prefill_binding import bind_expert_timeline
        events=[]; calls=[]
        adapter=NS(copied_bytes=0)
        def copy(layer,moves):
            calls.append(('copy',layer,tuple(moves)));adapter.copied_bytes+=128
        adapter.copy_experts=copy
        method=NS(apply=lambda layer,dispatch: calls.append(('apply',layer,dispatch)) or 'result')
        backend=NS(adapter=adapter,method=method,
                   synchronize=lambda identity:calls.append(('sync',identity)))
        identity=NS(allocation_epoch=7)
        with patch('migration_bridge.round9.runtime.working_set_timeline.enabled',return_value=True), patch(
                'migration_bridge.round9.runtime.working_set_timeline.event',
                side_effect=lambda category,phase,**fields:events.append((category,phase,fields))):
            self.assertTrue(bind_expert_timeline(backend,'source','revision'))
            backend.adapter.copy_experts(3,[(0,9),(1,10)])
            self.assertEqual(backend.method.apply('layer','dispatch'),'result')
            backend.synchronize(identity)
        self.assertEqual([event[1] for event in events],
                         ['shadow_prepare','compute_enqueue','device_wait'])
        self.assertEqual(events[0][2]['logical_experts'],[9,10])
        self.assertEqual(events[0][2]['bytes'],128)

    def test_timeline_wraps_read_only_slotted_copy_adapter(self):
        from migration_bridge.round9.runtime.prefill_binding import bind_expert_timeline

        class SlottedAdapter:
            __slots__ = ('copied_bytes', 'calls')

            def __init__(self):
                self.copied_bytes = 0
                self.calls = []

            def copy_experts(self, layer, moves):
                self.calls.append((layer, tuple(moves)))
                self.copied_bytes += 64

            def synchronize(self):
                self.calls.append('adapter_sync')

        adapter = SlottedAdapter()
        method = NS(apply=lambda layer, dispatch: 'result')
        backend = NS(
            adapter=adapter,
            method=method,
            synchronize=lambda identity: None,
        )
        with patch(
            'migration_bridge.round9.runtime.working_set_timeline.enabled',
            return_value=True,
        ), patch('migration_bridge.round9.runtime.working_set_timeline.event'):
            self.assertTrue(bind_expert_timeline(backend, 'source', 'revision'))
            backend.adapter.copy_experts(2, [(0, 7)])
            backend.adapter.synchronize()
        self.assertEqual(adapter.calls, [(2, ((0, 7),)), 'adapter_sync'])

    def test_real_fused_moe_keyword_call_contract(self):
        from migration_bridge.round9.runtime.prefill_binding import bind_expert_method
        calls=[]
        method=NS(apply=lambda **kwargs: calls.append(kwargs) or 'hybrid')
        runtime=NS(active=False, apply=lambda *args:'transport')
        bind_expert_method(method,3,runtime,lambda **kwargs:kwargs)
        dispatch=NS(hidden_states='h',topk_output=NS(topk_ids='ids',topk_weights='weights'))
        self.assertEqual(method.apply(layer='owner',dispatch_output=dispatch),'hybrid')
        self.assertEqual(calls[0],dict(layer='owner',dispatch_output=dispatch))
        runtime.active=True
        self.assertEqual(method.apply(layer='owner',dispatch_output=dispatch),dict(hidden_states='transport'))

    def test_default_off_does_not_touch_model_or_import_device(self):
        with patch('migration_bridge.round9.runtime.prefill_binding.load_framework', return_value=None):
            self.assertIsNone(prefill_model_ready(object()))

    def test_model_runner_seam_precedes_kv_pool_and_is_unique(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        sys.path.insert(0, str(tools))
        try:
            from install_framework_overlay import patch_model_runner
            original = 'class Runner:\n    def initialize(self):\n        self.load_model()\n        self.init_memory_pool()\n'
            result = patch_model_runner(original)
            compile(result, 'runner.py', 'exec')
            self.assertLess(result.index('prefill_model_ready(self)'), result.index('self.init_memory_pool()'))
            self.assertLess(result.index('self.init_memory_pool()'), result.index('context_pools_ready(self)'))
            with self.assertRaises(ValueError): patch_model_runner(result)
            with self.assertRaises(ValueError): patch_model_runner('no valid seam')
        finally: sys.path.remove(str(tools))

    def test_unsupported_scheduler_rejected_before_device_import(self):
        config = dict(prefill=dict(threshold_tokens=1))
        runner = NS(server_args=NS(disable_overlap_schedule=False))
        with patch('migration_bridge.round9.runtime.prefill_binding.load_framework', return_value=config):
            with self.assertRaisesRegex(RuntimeError, 'serial'):
                prefill_model_ready(runner)

    def test_kv_profile_subtracts_workspace_before_gdn_and_token_sizing(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        sys.path.insert(0, str(tools))
        try:
            from install_framework_overlay import patch_kv_profiler
            original = ('class Runner:\n    def size(self):\n        rest_memory = 8\n'
                        '        if self.mambaish_config is not None:\n'
                        '            rest_memory = self.handle_max_mamba_cache(rest_memory)\n'
                        '        return rest_memory\n')
            patched = patch_kv_profiler(original); scope = {}; exec(patched, scope)
            runner = scope['Runner'](); runner.mambaish_config = None
            self.assertEqual(runner.size(), 8)
            runner._round9_prefill_workspace_bytes = 2 * 1024**3
            self.assertEqual(runner.size(), 6)
            with self.assertRaises(ValueError): patch_kv_profiler(patched)
        finally: sys.path.remove(str(tools))


if __name__ == '__main__': unittest.main()
