import json
import tempfile
import unittest
from pathlib import Path

from round18.expert_pool_analysis import (
    RouteEvent,
    frequency_resident_plan,
    load_round9_route_tape,
    simulate_pool,
)


class ExpertPoolAnalysisTests(unittest.TestCase):
    def test_lru_and_belady_count_unique_event_misses(self):
        events = [
            RouteEvent(layer=0, experts=(1,)),
            RouteEvent(layer=0, experts=(2,)),
            RouteEvent(layer=0, experts=(3,)),
            RouteEvent(layer=0, experts=(1,)),
            RouteEvent(layer=0, experts=(2,)),
            RouteEvent(layer=0, experts=(3,)),
        ]

        lru = simulate_pool(events, capacity=2, policy="lru")
        belady = simulate_pool(events, capacity=2, policy="belady")

        self.assertEqual(lru.total_occurrences, 6)
        self.assertEqual(lru.unique_requests, 6)
        self.assertEqual(lru.miss_loads, 6)
        self.assertEqual(belady.miss_loads, 4)
        self.assertLessEqual(belady.miss_loads, lru.miss_loads)

    def test_static_frequency_is_per_layer(self):
        events = [
            RouteEvent(layer=0, experts=(0, 1)),
            RouteEvent(layer=0, experts=(0, 2)),
            RouteEvent(layer=1, experts=(7, 8)),
            RouteEvent(layer=1, experts=(7, 9)),
        ]

        result = simulate_pool(events, capacity=2, policy="static_frequency")

        self.assertEqual(result.miss_loads, 2)
        self.assertEqual(result.per_layer[0].resident_final, (0, 1))
        self.assertEqual(result.per_layer[1].resident_final, (7, 8))

    def test_event_larger_than_capacity_is_rejected(self):
        events = [RouteEvent(layer=0, experts=(0, 1, 2))]
        with self.assertRaisesRegex(ValueError, "working set"):
            simulate_pool(events, capacity=2, policy="lru")

    def test_dynamic_policy_accepts_frequency_warm_start(self):
        events = [
            RouteEvent(layer=0, experts=(0,)),
            RouteEvent(layer=0, experts=(0,)),
            RouteEvent(layer=0, experts=(1,)),
        ]
        initial = frequency_resident_plan(events, capacity=1)
        result = simulate_pool(
            events, capacity=1, policy="belady", initial_resident=initial
        )
        self.assertEqual(initial, {0: (0,)})
        self.assertEqual(result.hits, 2)
        self.assertEqual(result.miss_loads, 1)

    def test_round9_group_tape_restores_logical_ids_per_row(self):
        rows = [
            {
                "kind": "header",
                "schema": 1,
            },
            {
                "kind": "group",
                "logical_layer": 3,
                "logical_experts": [10, 20, 30],
                "remapped_ids": [[2, 0], [1, 2]],
                "rows": [4, 5],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
            events = load_round9_route_tape(path)

        self.assertEqual(
            events,
            [
                RouteEvent(layer=3, experts=(30, 10), token=4),
                RouteEvent(layer=3, experts=(20, 30), token=5),
            ],
        )


if __name__ == "__main__":
    unittest.main()
