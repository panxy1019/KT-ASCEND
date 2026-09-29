import unittest

from tools.kt_perf_closure.build_gap_accounting import partition_timeline


class BuildGapAccountingTests(unittest.TestCase):
    def test_partitions_known_intervals_and_names_dependency_gaps(self):
        intervals = [
            {"component": "GDN", "layer": 0, "start_ns": 10, "end_ns": 20},
            {
                "component": "ROUTED_EXPERT",
                "layer": 0,
                "start_ns": 30,
                "end_ns": 50,
            },
            {
                "component": "PLE_CONSUMER",
                "layer": 1,
                "start_ns": 53,
                "end_ns": 56,
            },
            {
                "component": "QSA_ATTENTION",
                "layer": 1,
                "start_ns": 60,
                "end_ns": 70,
            },
            {
                "component": "ROUTED_EXPERT",
                "layer": 1,
                "start_ns": 75,
                "end_ns": 90,
            },
        ]

        result = partition_timeline(0, 100, intervals)

        self.assertEqual(result["known_ns"], 58)
        self.assertEqual(result["gap_ns"], 42)
        self.assertEqual(
            result["gaps"],
            {
                "scheduler_to_attention": 10,
                "attention_to_expert_same_layer": 15,
                "expert_to_ple": 3,
                "ple_to_attention": 4,
                "expert_to_scheduler": 10,
            },
        )

    def test_rejects_overlapping_top_level_intervals(self):
        intervals = [
            {"component": "GDN", "layer": 0, "start_ns": 10, "end_ns": 30},
            {
                "component": "ROUTED_EXPERT",
                "layer": 0,
                "start_ns": 20,
                "end_ns": 40,
            },
        ]

        with self.assertRaisesRegex(ValueError, "overlap"):
            partition_timeline(0, 50, intervals)


if __name__ == "__main__":
    unittest.main()
