import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from migration_bridge.round9.runtime.framework_hooks import RuntimeCollector, framework_before, prepare_masks
from migration_bridge.round9.runtime.prefill_telemetry import PrefillCollector


class FrameworkHooksTests(unittest.TestCase):
    def runtime(self, directory):
        c = object.__new__(RuntimeCollector)
        c.collector = PrefillCollector(layers=2, experts=4, topk=2)
        c.req = c.pending_req = c.record = None
        c.records = []
        c.broken = False
        c.residents = [[0, 1], [0, 1]]
        c.path = Path(directory) / 'stats.json'
        return c

    def batch(self, req, extend=True):
        return NS(reqs=[req], forward_mode=NS(is_extend=lambda: extend))

    def req(self, rid, start=0):
        return NS(rid=rid, last_extend_prefix_len=start, origin_input_ids=[7, 8],
                  extend_input_len=1, finished_reason=None)

    def execute(self, c):
        for layer in (0, 1): c.collector.observe(layer, [[2, 3]])
        c.after()

    def test_off_hooks_do_not_touch_scheduler_or_wrapper(self):
        with patch.dict('os.environ', {}, clear=True):
            prepare_masks(None, None)
            framework_before(None, None)

    def test_actual_scheduler_fields_publish_only_complete_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.runtime(directory); req = self.req('r')
            c.before(self.batch(req)); self.execute(c)
            self.assertFalse(c.path.exists())
            req.last_extend_prefix_len = 1
            c.before(self.batch(req)); self.execute(c)
            document = json.loads(c.path.read_text())
            self.assertEqual(document['statistics']['tokens'], 2)
            self.assertEqual(document['requests'][0]['request_id'], 'r')
            c.before(self.batch(req, extend=False)); c.after()
            self.assertEqual(document, json.loads(c.path.read_text()))
            req.last_extend_prefix_len = 0
            with self.assertRaises(RuntimeError): c.before(self.batch(req))

    def test_request_switch_requires_prior_abort_and_drops_only_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.runtime(directory); old = self.req('old'); new = self.req('new')
            c.before(self.batch(old)); self.execute(c)
            with self.assertRaises(RuntimeError): c.before(self.batch(new))
            old.finished_reason = type('FINISH_ABORT', (), {})()
            c.before(self.batch(new)); self.execute(c)
            new.last_extend_prefix_len = 1
            c.before(self.batch(new)); self.execute(c)
            result = json.loads(c.path.read_text())
            self.assertEqual(result['statistics']['completed_phases'], 1)
            self.assertEqual([r['request_id'] for r in result['requests']], ['new'])


if __name__ == '__main__': unittest.main()
