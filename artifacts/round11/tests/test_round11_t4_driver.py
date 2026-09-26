import json
from pathlib import Path
import tempfile
import unittest

from artifacts.round11.round11_t4_driver import (
    build_schedule,
    consume_stream,
    validate_dynamic_events,
    wait_dynamic_events,
)


class FakeResponse:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        for value in self.values:
            if value == "[DONE]":
                yield b"data: [DONE]\n"
            else:
                yield ("data: " + json.dumps(value) + "\n").encode()


class Round11T4DriverTests(unittest.TestCase):
    def test_stream_requires_single_token_cumulative_growth(self):
        values = []
        for index in range(4):
            values.append(
                {
                    "output_ids": list(range(index + 1)),
                    "meta_info": {
                        "output_token_logprobs": [(-0.1, token, None) for token in range(index + 1)],
                        "finish_reason": {"type": "length"} if index == 3 else None,
                        "completion_tokens": index + 1,
                    },
                }
            )
        clock = iter((10, 20, 30, 40)).__next__
        result = consume_stream(FakeResponse(values + ["[DONE]"]), 4, clock=clock)
        self.assertEqual(result["output_ids"], [0, 1, 2, 3])
        self.assertEqual(result["token_arrival_monotonic_ns"], [10, 20, 30, 40])
        self.assertTrue(result["finite_logprobs"])
        self.assertEqual(result["finish_reason"], "length")

    def test_stream_rejects_batched_or_nonprefix_output(self):
        values = [
            {"output_ids": [1], "meta_info": {"output_token_logprobs": [(-1.0, 1, None)]}},
            {"output_ids": [1, 2, 3], "meta_info": {"output_token_logprobs": [(-1.0, 1, None)] * 3}},
        ]
        with self.assertRaisesRegex(RuntimeError, "one token"):
            consume_stream(FakeResponse(values), 3, clock=lambda: 1)

    def test_schedule_has_one_warmup_and_four_measurements(self):
        fixture = {"cases": {"edge64": {"input_ids": list(range(64))}}}
        schedule = build_schedule(fixture, mode="STATIC", block="A")
        self.assertEqual([row["decode_tokens"] for row in schedule], [32, 32, 64, 128, 256])
        self.assertEqual(schedule[0]["phase"], "warmup")
        self.assertTrue(all(row["request"]["stream"] for row in schedule))
        self.assertTrue(all(row["request"]["sampling_params"]["ignore_eos"] for row in schedule))

    def test_dynamic_event_contract_uses_exclusive_phase_receipt(self):
        rid = "r11-t4-dynamic-a-32"
        timing = {
            "route_statistics": 1_000_000,
            "consumer_drain_sync": 2_000_000,
            "state_fence_before": 3_000_000,
            "selection": 4_000_000,
            "migration_copy_and_sync": 10_000_000,
            "mapping_publication": 2_000_000,
            "state_fence_after": 3_000_000,
            "cleanup": 5_000_000,
            "total_exposed": 30_000_000,
        }
        events = [
            {
                "status": "PREFILL_END_COMMIT",
                "request_id": rid,
                "live_state_preserved": True,
                "changed_layers": 48,
                "copied_experts": 96,
            },
            {"status": "ROUND11_DYNAMIC_TIMING", "request_id": rid, "timing_ns": timing},
            {"status": "PREFILL_END_REQUEST_DRAINED", "request_id": rid},
        ]
        result = validate_dynamic_events(rid, events)
        self.assertEqual(result["timing_ns"]["total_exposed"], 30_000_000)
        events[1]["timing_ns"]["total_exposed"] = 10_000_000
        with self.assertRaisesRegex(RuntimeError, "exclusive"):
            validate_dynamic_events(rid, events)

    def test_dynamic_event_wait_allows_bounded_async_drain_append(self):
        rid = "r11-t4-dynamic-a-01"
        commit = {"status": "PREFILL_END_COMMIT", "request_id": rid}
        timing = {"status": "ROUND11_DYNAMIC_TIMING", "request_id": rid}
        drain = {"status": "PREFILL_END_REQUEST_DRAINED", "request_id": rid}
        snapshots = iter(([commit], [commit, timing], [commit, timing, drain]))
        sleeps = []
        result = wait_dynamic_events(
            Path("unused.jsonl"),
            0,
            rid,
            timeout=1,
            poll=0.01,
            read_fn=lambda _path: next(snapshots),
            clock=lambda: 0,
            sleep=lambda seconds: sleeps.append(seconds),
        )
        self.assertEqual(result, [commit, timing, drain])
        self.assertEqual(sleeps, [0.01, 0.01])


if __name__ == "__main__":
    unittest.main()
