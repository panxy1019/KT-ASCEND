import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/round14_a_driver.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_a_driver", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures():
    return {
        "round10": {"cases": {"edge64": {"input_ids": list(range(64))}}},
        "round11": {"cases": {
            "edge128": {"input_ids": list(range(128))},
            "edge512": {"input_ids": list(range(512))},
            "canary_marker": {"input_ids": [1, 2], "validator": {"kind": "marker"}},
            "canary_math391": {"input_ids": [3, 4], "validator": {"kind": "math391"}},
            "canary_json": {"input_ids": [5, 6], "validator": {"kind": "json"}},
            "canary_quote": {"input_ids": [7, 8], "validator": {"kind": "quote"}},
            "canary_transform": {"input_ids": [9, 10], "validator": {"kind": "transform"}},
        }},
    }


class Round14ADriverTest(unittest.TestCase):
    def test_schedule_is_w2_first_backend_identical_and_exactly_24(self):
        api = load_module()
        kt = api.build_schedule(fixtures(), version=235, backend="KT", block="A")
        native = api.build_schedule(
            fixtures(), version=236, backend="NATIVE_TP8", block="A"
        )
        self.assertEqual(len(kt), 24)
        self.assertEqual(len(native), 24)
        self.assertEqual(
            [row["workload"] for row in kt[:20]],
            [name for name in ("W2", "W0", "W1", "W4", "W3") for _ in range(4)],
        )
        for left, right in zip(kt[:20], native[:20]):
            self.assertEqual(left["request"]["input_ids"], right["request"]["input_ids"])
            self.assertEqual(left["decode_tokens"], right["decode_tokens"])
            self.assertEqual(left["request"]["sampling_params"], right["request"]["sampling_params"])
        self.assertTrue(all(row["request"]["stream"] for row in kt[:20]))
        self.assertTrue(all(row["request"]["sampling_params"]["ignore_eos"] for row in kt[:20]))
        self.assertTrue(all(row["kind"] == "quality" for row in kt[20:]))
        self.assertTrue(all(not row["request"]["stream"] for row in kt[20:]))
        self.assertTrue(all(not row["request"]["sampling_params"]["ignore_eos"] for row in kt[20:]))
        self.assertEqual({row["server_profile"] for row in kt}, {"H513_EXPERIMENTAL"})
        self.assertEqual({row["server_profile"] for row in native}, {"NATIVE_TP8_NO_HYBRID_THRESHOLD"})

    def test_roles_are_fixed_k_n_n_k_and_version_mismatch_fails(self):
        api = load_module()
        expected = {
            235: ("KT", "A"),
            236: ("NATIVE_TP8", "A"),
            237: ("NATIVE_TP8", "B"),
            238: ("KT", "B"),
            239: ("NATIVE_TP8", "A"),
            240: ("NATIVE_TP8", "B"),
            241: ("NATIVE_TP8", "A"),
            242: ("NATIVE_TP8", "B"),
            247: ("NATIVE_TP8", "A"),
            248: ("NATIVE_TP8", "B"),
            251: ("NATIVE_TP8", "A"),
            252: ("NATIVE_TP8", "B"),
        }
        self.assertEqual(api.ROLES, expected)
        for version, (backend, block) in expected.items():
            rows = api.build_schedule(
                fixtures(), version=version, backend=backend, block=block
            )
            self.assertEqual(len(rows), 24)
        with self.assertRaisesRegex(ValueError, "role/version mismatch"):
            api.build_schedule(fixtures(), version=235, backend="NATIVE_TP8", block="A")

    def test_measurement_window_uses_actual_tokens_and_full_wall_window(self):
        api = load_module()
        summary = api.measurement_window(
            [
                {"phase": "measurement", "completion_tokens": 128},
                {"phase": "measurement", "completion_tokens": 127},
                {"phase": "measurement", "completion_tokens": 126},
            ],
            started_ns=1_000_000_000,
            ended_ns=101_000_000_000,
        )
        self.assertEqual(summary["successful_requests"], 3)
        self.assertEqual(summary["actual_output_tokens"], 381)
        self.assertAlmostEqual(summary["serial_output_tokens_per_second"], 3.81)
        self.assertAlmostEqual(summary["requests_per_second"], 0.03)
        with self.assertRaisesRegex(ValueError, "three measurements"):
            api.measurement_window(
                [{"phase": "measurement", "completion_tokens": 128}],
                started_ns=0,
                ended_ns=1,
            )

    def test_kt_qsa_requires_current_prefill_and_decode_paths(self):
        api = load_module()
        rid = "request"
        rows = []
        for layer in range(3, 48, 4):
            rows.append({
                "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                "request_id": rid,
                "layer": layer,
                "strategy": "VECTORIZED",
                "query_tokens": 32,
            })
            rows.append({
                "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                "request_id": rid,
                "layer": layer,
                "strategy": "DIRECT_BF16_ROW",
                "query_tokens": 1,
            })
        result = api.validate_kt_qsa(rid, rows)
        self.assertEqual(result["prefill_strategy"], "VECTORIZED")
        self.assertEqual(result["decode_strategy"], "DIRECT_BF16_ROW")
        self.assertEqual(result["prefill_layers"], list(range(3, 48, 4)))
        self.assertEqual(result["decode_layers"], list(range(3, 48, 4)))
        broken = [dict(row) for row in rows]
        broken[-1]["strategy"] = "ROW"
        with self.assertRaisesRegex(RuntimeError, "QSA path identity"):
            api.validate_kt_qsa(rid, broken)

    def test_native_rank_evidence_covers_all_ranks_and_layers(self):
        api = load_module()
        events = {}
        qsa_layers = list(range(3, 48, 4))
        gdn_layers = [layer for layer in range(48) if layer not in qsa_layers]
        for rank in range(8):
            rows = []
            for layer in range(48):
                rows.append({
                    "status": "NATIVE_TP8_LAYER_PHYSICAL",
                    "rank": rank,
                    "layer": layer,
                    "w13_device": "npu",
                    "w2_device": "npu",
                    "dtype": "torch.bfloat16",
                    "w13_bytes": 2,
                    "w2_bytes": 2,
                })
            for layer in qsa_layers:
                rows.append({
                    "status": "NATIVE_TP8_OLD_QSA_HIT",
                    "rank": rank,
                    "layer": layer,
                    "query_tokens": 128,
                    "query_heads": 3,
                    "kv_heads": 1,
                })
            for layer in gdn_layers:
                rows.append({
                    "status": "NATIVE_TP8_GDN_HIT",
                    "rank": rank,
                    "layer": layer,
                    "key_heads": 2,
                    "value_heads": 6,
                    "z_heads": 6,
                    "conv_state_shape": [1],
                    "recurrent_state_shape": [1],
                    "conv_state_device": "npu",
                    "recurrent_state_device": "npu",
                })
            events[rank] = rows
        result = api.validate_native_rank_evidence(events)
        self.assertEqual(result["ranks"], list(range(8)))
        self.assertEqual(result["old_qsa_branch"], "QUALIFIED_ORIGINAL")
        broken = {rank: list(rows) for rank, rows in events.items()}
        broken[7] = [
            row for row in broken[7]
            if not (row["status"] == "NATIVE_TP8_OLD_QSA_HIT" and row["layer"] == 47)
        ]
        with self.assertRaisesRegex(RuntimeError, "old-QSA"):
            api.validate_native_rank_evidence(broken)

    def test_ple_validation_requires_request_drain_on_kt_and_all_native_ranks(self):
        api = load_module()
        rid = "request"
        kt = api.validate_kt_ple(rid, [
            {"status": "PLE_ASYNC_BOUND"},
            {"status": "PLE_BATCH_DRAINED", "request_id": rid, "async_engaged": True},
        ])
        self.assertEqual(kt, {"drained": 1, "async": 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(8):
                rows = [
                    {"status": "PLE_ASYNC_BOUND"},
                    {"status": "PLE_BATCH_DRAINED", "request_id": rid,
                     "async_engaged": True},
                ]
                (root / f"ple-async-events-rank{rank}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            native = api.validate_native_ple(root, rid)
            self.assertEqual(sorted(native), list(range(8)))
            self.assertTrue(all(native[rank]["async_batches"] == 1 for rank in native))
            (root / "ple-async-events-rank7.jsonl").write_text(
                json.dumps({"status": "PLE_ASYNC_BOUND"}) + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "rank 7"):
                api.validate_native_ple(root, rid)

    def test_stream_metrics_use_first_and_last_real_token_arrivals(self):
        api = load_module()
        metrics = api.stream_metrics(
            {"token_arrival_monotonic_ns": [2_000_000_000, 4_000_000_000,
                                             8_000_000_000]},
            started_ns=1_000_000_000,
            ended_ns=9_000_000_000,
        )
        self.assertEqual(metrics["ttft_seconds"], 1.0)
        self.assertEqual(metrics["e2e_seconds"], 8.0)
        self.assertEqual(metrics["tpot_seconds"], 3.0)
        with self.assertRaisesRegex(RuntimeError, "arrival"):
            api.stream_metrics(
                {"token_arrival_monotonic_ns": []},
                started_ns=1,
                ended_ns=2,
            )

    def test_run_freezes_all_windows_full_tokens_and_24_attempts(self):
        api = load_module()
        counter = {"value": 0}

        def execute(row, _out, backend):
            counter["value"] += 1
            started = counter["value"] * 10_000_000_000
            ended = started + 2_000_000_000
            if row["kind"] == "quality":
                return {
                    "started_ns": started,
                    "ended_ns": ended,
                    "quality_pass": True,
                    "output_ids": [1, 2, 3],
                    "completion_tokens": 3,
                    "finite_output_logprobs": True,
                    "finish_reason": "stop",
                    "backend_evidence": {"backend": backend},
                }
            tokens = row["decode_tokens"]
            arrivals = [started + (index + 1) * 10_000_000 for index in range(tokens)]
            return {
                "started_ns": started,
                "ended_ns": ended,
                "output_ids": [3897, 248046] + list(range(max(0, tokens - 2))),
                "token_arrival_monotonic_ns": arrivals,
                "finite_logprobs": True,
                "finish_reason": "length",
                "completion_tokens": tokens,
                "saw_done_marker": True,
                "backend_evidence": {"backend": backend},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture10 = root / "fixture10.json"
            fixture11 = root / "fixture11.json"
            fixture10.write_text(json.dumps(fixtures()["round10"]))
            fixture11.write_text(json.dumps(fixtures()["round11"]))
            result = api.run(
                235, "KT", "A", root, fixture10, fixture11,
                request_executor=execute,
            )
            self.assertEqual(result["status"], "PASS_ROUND14_A_DRIVER_RAW")
            self.assertEqual(result["generation_attempts"], 24)
            self.assertEqual(result["abort_controls"], 0)
            self.assertEqual(set(result["measurement_windows"]),
                             {"W0", "W1", "W2", "W3", "W4"})
            self.assertEqual(result["measurement_windows"]["W2"]["actual_output_tokens"],
                             384)
            self.assertEqual(len([row for row in result["results"]
                                  if row.get("quality_pass")]), 4)
            self.assertTrue((root / "round14-a-raw.json").is_file())


if __name__ == "__main__":
    unittest.main()
