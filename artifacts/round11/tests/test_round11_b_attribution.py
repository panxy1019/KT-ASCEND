import unittest

from migration_bridge.round9.runtime.round11_b_attribution import (
    observe,
    resolve_device_spans,
    validate_scope,
)


class FakeEvent:
    def __init__(self, complete=True, elapsed=3.5):
        self.complete = complete
        self.elapsed = elapsed
        self.records = 0

    def record(self):
        self.records += 1

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        self.last_other = other
        return self.elapsed


class Round11BAttributionTests(unittest.TestCase):
    def test_observer_preserves_provider_result_and_exception(self):
        calls = []
        rows = []

        def provider(value):
            calls.append(value)
            if value < 0:
                raise ValueError("provider failure")
            return value + 1

        clock = iter((100, 130, 200, 250)).__next__
        wrapped = observe(provider, rows.append, "GDN", 0,
                          lambda: {"request_id": "r", "mode": "decode"},
                          clock=clock)
        self.assertEqual(wrapped(4), 5)
        with self.assertRaisesRegex(ValueError, "provider failure"):
            wrapped(-1)
        self.assertEqual(calls, [4, -1])
        self.assertEqual([row["outcome"] for row in rows], ["return", "error"])
        self.assertEqual([row["duration_ns"] for row in rows], [30, 50])

    def test_device_resolution_is_deferred_and_never_synchronizes(self):
        start = FakeEvent(complete=True, elapsed=7.25)
        end = FakeEvent(complete=False)
        rows = []
        wrapped = observe(lambda: 9, rows.append, "QSA_ATTENTION", 3,
                          lambda: {"request_id": "r", "mode": "decode"},
                          device_factory=lambda: (start, end),
                          clock=iter((100, 140)).__next__)
        self.assertEqual(wrapped(), 9)
        batch = {"spans": rows, "observer_errors": []}
        self.assertFalse(resolve_device_spans(batch))
        self.assertEqual(rows[0]["device_status"], "PENDING")
        end.complete = True
        self.assertTrue(resolve_device_spans(batch))
        self.assertEqual(rows[0]["device_status"], "COMPLETE")
        self.assertEqual(rows[0]["device_ms"], 7.25)
        self.assertNotIn("_device", rows[0])

    def test_host_only_span_resolves_without_device_event(self):
        rows = []
        observe(lambda: None, rows.append, "PLE_EXPOSED_WAIT", 1,
                lambda: {"request_id": "r", "mode": "extend"},
                clock=iter((10, 20)).__next__)()
        batch = {"spans": rows, "observer_errors": []}
        self.assertTrue(resolve_device_spans(batch))
        self.assertEqual(rows[0]["device_status"], "NOT_APPLICABLE")
        self.assertIsNone(rows[0]["device_ms"])

    def test_scope_rejects_dynamic_or_wrong_backend(self):
        good = {
            "storage": "MXFP4",
            "placement": "static",
            "capacity": 32,
            "workers": 128,
            "device": "npu",
            "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        }
        validate_scope(good)
        for key, value in (("placement", "request_boundary"),
                           ("workers", 64), ("storage", "BF16")):
            with self.assertRaises(ValueError):
                validate_scope({**good, key: value})


if __name__ == "__main__":
    unittest.main()
