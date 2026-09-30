from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from round18.native_route_trace import NativeRouteTraceRecorder


class _Tensor:
    def __init__(self, values):
        self.values = values
        self.detach_calls = 0
        self.cpu_calls = 0

    def detach(self):
        self.detach_calls += 1
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self

    def tolist(self):
        return self.values


class NativeRouteTraceTests(unittest.TestCase):
    def test_records_one_bounded_jsonl_event_with_route_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            recorder = NativeRouteTraceRecorder(path, max_events=2)
            ids = _Tensor([[1, 2], [3, 4]])
            weights = _Tensor([[0.6, 0.4], [0.7, 0.3]])
            hidden = type("Hidden", (), {"shape": (2, 2560)})()

            recorder.record(
                layer_name="language_model.model.layers.7.mlp.experts",
                generation=9,
                hidden_states=hidden,
                topk_ids=ids,
                topk_weights=weights,
                identity_key=("round18", 1, 2, "layer7", 9),
            )

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["event_index"], 0)
            self.assertEqual(row["logical_layer"], 7)
            self.assertEqual(row["generation"], 9)
            self.assertEqual(row["token_rows"], 2)
            self.assertEqual(row["topk_ids"], [[1, 2], [3, 4]])
            self.assertEqual(row["topk_weights"], [[0.6, 0.4], [0.7, 0.3]])
            self.assertEqual(ids.detach_calls, 1)
            self.assertEqual(weights.cpu_calls, 1)

    def test_rejects_event_overflow_before_writing_extra_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            recorder = NativeRouteTraceRecorder(path, max_events=1)
            kwargs = {
                "layer_name": "model.layers.0.mlp.experts",
                "generation": 1,
                "hidden_states": type("Hidden", (), {"shape": (1, 2560)})(),
                "topk_ids": _Tensor([[1]]),
                "topk_weights": _Tensor([[1.0]]),
                "identity_key": ("identity",),
            }
            recorder.record(**kwargs)
            with self.assertRaisesRegex(RuntimeError, "event limit"):
                recorder.record(**kwargs)
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_rejects_mismatched_topk_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = NativeRouteTraceRecorder(Path(directory) / "trace.jsonl")
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                recorder.record(
                    layer_name="model.layers.0.mlp.experts",
                    generation=1,
                    hidden_states=type("Hidden", (), {"shape": (1, 2560)})(),
                    topk_ids=_Tensor([[1, 2]]),
                    topk_weights=_Tensor([[1.0]]),
                    identity_key=("identity",),
                )


if __name__ == "__main__":
    unittest.main()
