import unittest

from artifacts.round11.round11_t5_driver import build_schedule, collect_measurement_batches


class Round11T5DriverTests(unittest.TestCase):
    def test_schedule_is_four_generations_and_two_measurements(self):
        fixture = {
            "cases": {
                "edge128": {"input_ids": list(range(128))},
                "canary_math391": {"input_ids": list(range(27))},
            }
        }
        schedule = build_schedule(fixture, block="A")
        self.assertEqual(len(schedule), 4)
        self.assertEqual([row["phase"] for row in schedule],
                         ["warmup", "measurement", "measurement", "drain"])
        self.assertEqual([row["decode_tokens"] for row in schedule], [4, 32, 64, 4])
        self.assertTrue(all(row["request"]["stream"] for row in schedule))
        self.assertEqual(sum(row["phase"] == "measurement" for row in schedule), 2)

    def test_trace_collection_requires_both_modes_and_complete_sample(self):
        required = []
        for component, layers in (("GDN", (0, 24, 46)),
                                  ("QSA_ATTENTION", (3, 23, 47)),
                                  ("QSA_INDEXER", (3, 23, 47))):
            required.extend({"component": component, "layer": layer,
                             "device_status": "COMPLETE", "observer_errors": []}
                            for layer in layers)
        required.extend([
            {"component": "PLE_PREPARE", "layer": 1,
             "device_status": "NOT_APPLICABLE", "observer_errors": []},
            {"component": "PLE_EXPOSED_WAIT", "layer": 1,
             "device_status": "NOT_APPLICABLE", "observer_errors": []},
            {"component": "PLE_CONSUMER", "layer": 1,
             "device_status": "COMPLETE", "observer_errors": []},
        ])
        batches = [
            {"status": "ROUND11_B_BATCH", "request_id": "r", "mode": mode,
             "batch_sequence": index, "dropped_spans": 0,
             "observer_errors": [], "spans": [dict(row) for row in required]}
            for index, mode in enumerate(("extend", "decode"))
        ]
        actual = collect_measurement_batches(batches, {"r"})
        self.assertEqual(len(actual), 2)
        bad = [dict(row) for row in batches]
        bad[1] = {**bad[1], "spans": bad[1]["spans"][:-1]}
        with self.assertRaisesRegex(RuntimeError, "sample coverage"):
            collect_measurement_batches(bad, {"r"})


if __name__ == "__main__":
    unittest.main()
