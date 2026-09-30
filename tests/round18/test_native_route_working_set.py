from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from round18.native_route_working_set import (
    BF16_EXPERT_BYTES,
    analyze_native_route_working_set,
)


class NativeRouteWorkingSetTests(unittest.TestCase):
    def _write_fixture(self, directory: Path) -> tuple[Path, Path]:
        trace_path = directory / "native-route-trace.jsonl"
        result_path = directory / "route-result.json"
        events: list[dict[str, object]] = []
        requests: list[dict[str, object]] = []
        workload_ranges = {
            "W1": (range(0, 40), range(0, 10), 2),
            "W2": (range(20, 60), range(20, 30), 2),
            "W4": (range(40, 80), range(40, 50), 4),
        }
        for workload, (prefill_ids, decode_ids, completion_tokens) in (
            workload_ranges.items()
        ):
            start = len(events)
            prefill_rows = [
                list(prefill_ids)[offset : offset + 10]
                for offset in range(0, 40, 10)
            ]
            for phase_rows in (prefill_rows, [list(decode_ids)]):
                for layer in range(48):
                    events.append(
                        {
                            "schema": 1,
                            "kind": "native_route_event",
                            "event_index": len(events),
                            "logical_layer": layer,
                            "generation": 1,
                            "token_rows": len(phase_rows),
                            "top_k": 10,
                            "topk_ids": phase_rows,
                            "topk_weights": [
                                [0.1 for _ in row] for row in phase_rows
                            ],
                        }
                    )
            end = len(events)
            requests.append(
                {
                    "workload": workload,
                    "request_id": f"request-{workload.lower()}",
                    "input_tokens": 4,
                    "decode_tokens": completion_tokens,
                    "completion_tokens": completion_tokens,
                    "route_event_start": start,
                    "route_event_end_exclusive": end,
                    "route_event_count": end - start,
                    "route_token_rows": sum(
                        int(event["token_rows"]) for event in events[start:end]
                    ),
                }
            )
        trace_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "status": "PASS_ROUND18_FRESH_NATIVE_ROUTE_TRACE",
                    "logical_experts": 512,
                    "top_k": 10,
                    "layer_count": 48,
                    "route_event_count": len(events),
                    "requests": requests,
                }
            ),
            encoding="utf-8",
        )
        return result_path, trace_path

    def test_w1_plan_is_evaluated_on_split_w2_w4_holdouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path, trace_path = self._write_fixture(Path(directory))

            report = analyze_native_route_working_set(result_path, trace_path)

        self.assertEqual(report["status"], "PASS_ROUND18_NATIVE_ROUTE_ANALYSIS")
        self.assertEqual(report["development_workload"], "W1")
        self.assertEqual(report["holdout_workloads"], ["W2", "W4"])
        self.assertEqual(report["capacities"], [32, 64, 96, 128])
        q32 = report["by_capacity"]["Q32"]
        self.assertEqual(q32["w1_static_frequency_plan"]["0"], list(range(32)))
        prefill = q32["holdouts"]["W2"]["Prefill"]
        self.assertEqual(prefill["routed_model_tokens"], 4)
        self.assertEqual(prefill["generated_output_tokens"], 2)
        self.assertEqual(
            prefill["invocation_working_set"],
            {"fit_ratio": 0.0, "min": 40, "median": 40.0, "max": 40},
        )
        static = prefill["policies"]["static_frequency"]
        self.assertEqual(static["miss_experts"], 28 * 48)
        self.assertEqual(static["bf16_bytes"], 28 * 48 * BF16_EXPERT_BYTES)
        self.assertEqual(
            static["bf16_bytes_per_generated_output_token"],
            28 * 48 * BF16_EXPERT_BYTES / 2,
        )
        self.assertEqual(
            static["bf16_bytes_per_routed_model_token"],
            28 * 48 * BF16_EXPERT_BYTES / 4,
        )
        self.assertEqual(static["per_layer"]["0"]["miss_experts"], 28)
        self.assertEqual(prefill["policies"]["lru"]["miss_experts"], 28 * 48)
        self.assertEqual(prefill["policies"]["belady"]["miss_experts"], 28 * 48)
        self.assertEqual(
            q32["holdouts"]["W2"]["Decode"]["invocation_working_set"]
            ["fit_ratio"],
            1.0,
        )
        self.assertEqual(
            report["interpretation"]["dynamic_policy_scope"],
            "SEQUENTIAL_TOKEN_ROW_ORACLE_NOT_NATIVE_SINGLE_CALL_RUNNABLE",
        )

    def test_q64_fits_the_synthetic_prefill_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path, trace_path = self._write_fixture(Path(directory))
            report = analyze_native_route_working_set(result_path, trace_path)

        stats = report["by_capacity"]["Q64"]["holdouts"]["W2"]["Prefill"]
        self.assertEqual(stats["invocation_working_set"]["fit_ratio"], 1.0)

    def test_static_miss_experts_deduplicate_within_one_native_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path, trace_path = self._write_fixture(Path(directory))
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            w2_start = 96
            repeated = list(range(40, 50))
            for event in rows[w2_start : w2_start + 48]:
                event["topk_ids"] = [repeated, repeated, repeated, repeated]
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            report = analyze_native_route_working_set(result_path, trace_path)

        policies = report["by_capacity"]["Q32"]["holdouts"]["W2"]["Prefill"]
        self.assertEqual(policies["policies"]["static_frequency"]["miss_experts"], 10 * 48)
        self.assertEqual(policies["policies"]["lru"]["miss_experts"], 10 * 48)

    def test_rejects_missing_layer_and_non_top10_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, trace_path = self._write_fixture(root)
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            rows[0]["logical_layer"] = 47
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "all 48 layers"):
                analyze_native_route_working_set(result_path, trace_path)

            rows[0]["logical_layer"] = 0
            rows[0]["top_k"] = 9
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Top-10"):
                analyze_native_route_working_set(result_path, trace_path)

    def test_rejects_overlapping_or_out_of_order_request_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path, trace_path = self._write_fixture(Path(directory))
            payload = json.loads(result_path.read_text())
            payload["requests"][1]["route_event_start"] = 95
            payload["requests"][1]["route_event_count"] += 1
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contiguous request event ranges"):
                analyze_native_route_working_set(result_path, trace_path)


if __name__ == "__main__":
    unittest.main()
