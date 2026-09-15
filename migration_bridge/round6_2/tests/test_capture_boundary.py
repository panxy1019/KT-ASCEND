"""Dependency-free tests of real dumper class; not an A3 tensor test."""
import ast
import copy
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "third_party/sglang/python/sglang/srt/debug_utils/tensor_dump_forward_hook.py"


class Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def clone(self):
        return Tensor(self.value)


class Batch:
    def __init__(self, value):
        self.input_ids = Tensor(value)
        self.seq_lens = Tensor(1)
        self.positions = Tensor(0)


class Logits:
    def __init__(self, value):
        self.next_token_logits = Tensor(value)


class CaptureBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.saved = []
        tree = ast.parse(SOURCE.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TensorDumper")
        namespace = dict(os=os, Path=Path, Optional=__import__('typing').Optional,
                         List=list, logger=logging.getLogger(__name__),
                         ForwardBatch=Batch, LogitsProcessorOutput=Logits,
                         PPProxyTensors=type("Proxy", (), {}),
                         torch=SimpleNamespace(Tensor=Tensor, save=lambda value, path: self.saved.append(copy.deepcopy(value))))
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.cls = namespace["TensorDumper"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def dumper(self, enabled):
        with patch.dict(os.environ, {"KT_R62_COMPLETE_FORWARD_DUMP": str(int(enabled))}):
            return self.cls(self.tmp.name, [999], 1, 0, 0)

    def test_legacy_flush_precedes_current_logits(self):
        d = self.dumper(False)
        d.add_tensor("logits_processor", Logits(99))
        d._dump_hook("model", True)(None, (Batch(7),), Tensor(70))
        d.add_tensor("logits_processor", Logits(77))
        self.assertEqual(self.saved[0]["logits_processor"].value, 99)
        self.assertEqual(self.saved[0]["model.forward_batch_info.input_ids"].value, 7)

    def test_complete_outer_forward_aligns_two_calls(self):
        d = self.dumper(True)
        for token in (7, 11):
            batch = Batch(token)
            d.begin_complete_forward(None, (), {"forward_batch": batch})
            d._dump_hook("model", True)(None, (batch,), Tensor(token * 10))
            d.add_tensor("logits_processor", Logits(token * 11))
            batch.input_ids.value = -1
            d.end_complete_forward(None, (), {}, Logits(token * 11))
        self.assertEqual(len(self.saved), 2)
        for row, token in zip(self.saved, (7, 11)):
            self.assertEqual(row["capture.forward_batch_info.input_ids"].value, token)
            self.assertEqual(row["capture.next_token_logits"].value, token * 11)
        self.assertEqual(d._current_tensors, {})

    def test_unflushed_state_fails(self):
        d = self.dumper(True)
        d.add_tensor("stale", Tensor(1))
        with self.assertRaisesRegex(RuntimeError, "unflushed"):
            d.begin_complete_forward(None, (Batch(7),), {})

    def test_wrong_outer_output_fails(self):
        with self.assertRaisesRegex(RuntimeError, "LogitsProcessorOutput"):
            self.dumper(True).end_complete_forward(None, (), {}, Tensor(1))


if __name__ == "__main__":
    unittest.main()
