import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import round10_t4_driver as driver


class T4DriverTests(unittest.TestCase):
    def test_frozen_schedule_is_five_generations_one_abort(self):
        fixture = json.loads((ROOT / "round10-t1-fixture.json").read_text())
        rows = driver.build_schedule(fixture)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(row["phase"] == "cancel" for row in rows), 1)
        self.assertTrue(all(row["expected_path"] == "HYBRID" for row in rows))

    def test_commit_requires_one_generation_transition_and_drain(self):
        events = [
            {"status": "PREFILL_END_COMMIT", "request_id": "r", "live_state_preserved": True,
             "first_output_token_generation": 2, "decode_generation": 3, "generation": 3,
             "changed_layers": 1, "copied_experts": 2, "plans": [{}]},
            {"status": "PREFILL_END_REQUEST_DRAINED", "request_id": "r"},
        ]
        self.assertEqual(driver.validate_commit("r", events)["commit"]["generation"], 3)
        for broken in (
            events[:1],
            [dict(events[0], live_state_preserved=False), events[1]],
            [dict(events[0], decode_generation=4), events[1]],
        ):
            with self.assertRaises(RuntimeError):
                driver.validate_commit("r", broken)

    def test_cancel_rejects_commit_and_requires_discard_drain(self):
        event = {"status": "DYNAMIC_REQUEST_ABORT_DRAINED", "request_id": "r",
                 "changed_layers": 0, "copied_experts": 0}
        self.assertEqual(driver.validate_cancel_dynamic("r", [event]), event)
        with self.assertRaises(RuntimeError):
            driver.validate_cancel_dynamic("r", [event, {"status": "PREFILL_END_COMMIT", "request_id": "r"}])


if __name__ == "__main__":
    unittest.main()
