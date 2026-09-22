"""Exercise the real dumper AST without requiring a local torch install."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round6_2/tests"))
import test_capture_boundary as boundary
from test_capture_boundary import Batch, Logits, Tensor


class CompactCaptureTests(boundary.CaptureBoundaryTests):
    def test_compact_keeps_declared_diagnostic_but_drops_other_leaf_outputs(self):
        with patch.dict(os.environ, {"KT_R63_COMPACT_CAPTURE": "1", "KT_R63_FULL_DIAGNOSTIC_SEQUENCES": "0"}):
            d = self.dumper(True)
        for value in (7, 8):
            batch = Batch(value)
            d.begin_complete_forward(None, (batch,), {})
            d._dump_hook("leaf", False)(None, (), Tensor(5))
            d.end_complete_forward(None, (), {}, Logits(9))
        self.assertIn("leaf", self.saved[0])
        self.assertNotIn("leaf", self.saved[1])
        self.assertIn("capture.next_token_logits", self.saved[1])
        self.assertTrue(self.saved[0]["capture.transaction"]["r63_full_diagnostic"])
        self.assertFalse(self.saved[1]["capture.transaction"]["r63_full_diagnostic"])

    def test_index_exclusive_and_refers_to_committed_file(self):
        path = Path(self.tmp.name) / "index.jsonl"
        with patch.dict(os.environ, {"KT_R63_CAPTURE_INDEX": str(path)}):
            d = self.dumper(True)
            self.addCleanup(d._r63_index.close)
            with self.assertRaises(FileExistsError):
                self.dumper(True)
        self.assertEqual(path.read_text(), "")
        d.begin_complete_forward(None, (Batch(7),), {})
        d.end_complete_forward(None, (), {}, Logits(9))
        records = [json.loads(s) for s in path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sequence"], 0)
        self.assertTrue(Path(records[0]["file"]).is_file())
        self.assertFalse(list(Path(self.tmp.name).rglob("*.incomplete")))

    def test_compact_requires_complete_boundary(self):
        with patch.dict(os.environ, {"KT_R63_COMPACT_CAPTURE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "REQUIRES_COMPLETE"):
                self.dumper(False)
