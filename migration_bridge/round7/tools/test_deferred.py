import unittest
from deferred_probe import audit_trace


def event(layer,budget,count,pending):
    return dict(event='submit',layer_idx=layer,max_deferred_experts_per_token=budget,
        deferred_cpu_routes=count,outgoing_pending=pending,rows=1,immediate_cpu_routes=4)


class Deferred(unittest.TestCase):
    def test_actual_d1_and_terminal_drain(self):
        result=audit_trace([event(25,1,1,True),event(26,0,0,False)])
        self.assertEqual(result['deferred_cpu_routes'],1)

    def test_flag_without_execution_is_not_pass(self):
        with self.assertRaises(AssertionError):audit_trace([event(26,0,0,False)])

    def test_terminal_pending_rejected(self):
        with self.assertRaises(AssertionError):audit_trace([event(25,1,1,True),event(26,1,1,True)])
