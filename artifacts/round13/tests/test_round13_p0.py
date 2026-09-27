import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_round13_p0_snapshot import patch_launcher, patch_scheduler
from round13_attribution import (
    observe, resolve_device_spans, resolve_dispatch_output, validate_scope,
)
from round13_p0_driver import build_schedule, timing_metrics


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
        self.other = other
        return self.elapsed


class Round13P0Tests(unittest.TestCase):
    def test_observer_preserves_result_and_defers_event_query(self):
        rows = []
        start, end = FakeEvent(), FakeEvent(complete=False)
        wrapped = observe(lambda value: value + 1, rows.append, "GDN", 0,
                          lambda: {"request_id": "r", "mode": "decode"},
                          device_factory=lambda: (start, end),
                          clock=iter((10, 30)).__next__)
        self.assertEqual(wrapped(4), 5)
        batch = {"spans": rows, "observer_errors": []}
        self.assertFalse(resolve_device_spans(batch))
        self.assertEqual(rows[0]["device_status"], "PENDING")
        end.complete = True
        self.assertTrue(resolve_device_spans(batch))
        self.assertEqual(rows[0]["device_ms"], 3.5)
        self.assertNotIn("_device", rows[0])

    def test_scope_is_exact(self):
        scope = {"storage": "MXFP4", "placement": "static", "capacity": 32,
                 "workers": 128, "device": "npu",
                 "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540"}
        validate_scope(scope)
        with self.assertRaises(ValueError):
            validate_scope({**scope, "capacity": 64})

    def test_expert_dispatch_accepts_runtime_keyword_call(self):
        marker = object()
        self.assertIs(resolve_dispatch_output((), {"layer": object(),
                                                   "dispatch_output": marker}), marker)
        self.assertIs(resolve_dispatch_output((object(), marker), {}), marker)
        with self.assertRaises(TypeError):
            resolve_dispatch_output((object(),), {})

    def test_schedule_is_seven_generations_with_separate_dev(self):
        fixture = {"cases": {
            "edge128": {"input_ids": list(range(128))},
            "edge512": {"input_ids": list(range(512))},
            "canary_math391": {"input_ids": list(range(27))},
        }}
        dev = {"status": "FROZEN_ROUND13_DEV_TOKEN_IDS", "prompts": [
            {"id": "sched-dev-01", "input_ids": [1, 2]},
            {"id": "sched-dev-02", "input_ids": [3, 4, 5]},
        ]}
        rows = build_schedule(fixture, dev, block="A")
        self.assertEqual(len(rows), 7)
        self.assertEqual([row["phase"] for row in rows].count("dev"), 2)
        self.assertEqual([row["case_id"] for row in rows if row["phase"] == "measurement"],
                         ["edge128", "edge512"])
        self.assertTrue(all(row["request"]["sampling_params"]["ignore_eos"] for row in rows))

    def test_timing_uses_real_stream_arrivals(self):
        self.assertEqual(timing_metrics(10, 40, [20, 30, 35]), {
            "ttft_seconds": 1e-08,
            "e2e_seconds": 3e-08,
            "tpot_seconds": 7.5e-09,
        })

    def test_patchers_are_fail_closed_and_compile(self):
        scheduler = (
            "                from sglang.srt.layers.round9.framework_hooks import framework_before\n"
            "                framework_before(self, batch)\n"
            "                result = self.run_batch(batch)\n"
            "                self.process_batch_result(batch, result)\n"
        )
        self.assertIn("round13_before", patch_scheduler(scheduler))
        launcher = "\n".join((
            "if a.version != 177 or a.cost_mode != 'OFF':",
            " pass",
            "name='kt-round12a-kt-b-v177';out=root/'evidence'/'20260927-round12a-kt-b-v177'",
            "if budget.get('stage')!='ROUND12A_ABBA':raise RuntimeError('wrong Round12a ABBA budget')",
            "args+=['-e','ROUND9_HYBRID_PROBE=1']",
        )) + "\n"
        patched = patch_launcher(launcher, 184)
        self.assertIn("ROUND13_ATTRIBUTION_ENABLE=1", patched)
        self.assertIn("ROUND13_P0_PROFILE", patched)
        with self.assertRaises(ValueError):
            patch_launcher(patched, 184)


if __name__ == "__main__":
    unittest.main()
