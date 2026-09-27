import unittest

from migration_bridge.round9.tests.framework_smoke_checks import drained_request_ids


class DrainReceiptTests(unittest.TestCase):
    def test_prefill_commit_is_not_request_drain(self):
        event = dict(status='PREFILL_END_COMMIT', request_id='r', generation=1,
                     changed_layers=48, copied_experts=100, plans=[])
        for mode in ('prefill_end', 'request_boundary'):
            self.assertEqual(drained_request_ids([event], mode), set())

    def test_prefill_final_drain_requires_no_second_migration(self):
        event = dict(status='PREFILL_END_REQUEST_DRAINED', request_id='r',
                     generation=1, changed_layers=0, copied_experts=0)
        self.assertEqual(drained_request_ids([event], 'prefill_end'), {'r'})
        event['copied_experts'] = 1
        self.assertEqual(drained_request_ids([event], 'prefill_end'), set())

    def test_existing_boundary_receipt_is_accepted(self):
        event = dict(request_id='r', generation=1, changed_layers=1,
                     copied_experts=2, plans=[{}])
        self.assertEqual(drained_request_ids([event], 'request_boundary'), {'r'})
        self.assertEqual(drained_request_ids([{'request_id': 'r'}], 'request_boundary'), set())

    def test_binding_or_other_request_does_not_complete_target(self):
        events = [dict(status='BOUND_NOT_FULL_MODEL_QUALIFIED', generation=0),
                  dict(status='PREFILL_END_REQUEST_DRAINED', request_id='other',
                       changed_layers=0, copied_experts=0)]
        self.assertNotIn('target', drained_request_ids(events, 'prefill_end'))
