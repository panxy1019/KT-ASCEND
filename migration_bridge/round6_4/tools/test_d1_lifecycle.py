import unittest
from run_d1_lifecycle import audit_events


def events(groups=500):
    result = []
    for index in range(groups):
        deferred = int(index == 0)
        result.extend([
            dict(event="submit",layer_idx=7,rows=1,immediate_cpu_routes=1,deferred_cpu_routes=deferred,
                 immediate_cpu_experts=[16],deferred_cpu_experts=[38] if deferred else [],outgoing_pending=bool(deferred)),
            dict(event="pending_state",layer_idx=7,pending_layers=[7] if deferred else []),
            dict(event="sync",layer_idx=7,allow_pending=deferred),
            dict(event="sync_complete",layer_idx=7,allow_pending=deferred),
            dict(event="submit",layer_idx=8,rows=1,immediate_cpu_routes=1,deferred_cpu_routes=0,
                 immediate_cpu_experts=[8],deferred_cpu_experts=[],outgoing_pending=False)])
        if deferred:
            result.append(dict(event="previous_pending_consumed",producer_layer=7,consumer_layer=8))
        result.extend([dict(event="pending_state",layer_idx=8,pending_layers=[]),
                       dict(event="sync",layer_idx=8,allow_pending=0),
                       dict(event="sync_complete",layer_idx=8,allow_pending=0)])
    for i,row in enumerate(result):
        row["trace_sequence"] = i+1
    return result


class LifecycleTests(unittest.TestCase):
    def test_real_tasks_not_token_proxy(self):
        report = audit_events(events())
        self.assertEqual(report["physical_forwards"],500)
        self.assertEqual(report["real_cpu_tasks_with_selected_routes"],1001)
        self.assertEqual(report["worker_sync_complete"],1000)

    def test_insufficient_or_missing_transition_fails(self):
        with self.assertRaisesRegex(AssertionError,"INSUFFICIENT"):
            audit_events(events(2))
        data = events()
        data[-1]["event"] = "unknown"
        with self.assertRaisesRegex(ValueError,"TRANSITION"):
            audit_events(data)


if __name__ == "__main__":
    unittest.main()
