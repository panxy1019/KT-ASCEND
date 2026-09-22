"""Dependency-free tests of real dumper class; not an A3 tensor test."""
import ast
import copy
import logging
import hashlib
import json
from functools import wraps
from contextvars import ContextVar
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
        self.shape = (1,)
        self.dtype = "mock"

    def cpu(self):
        return self

    def clone(self):
        return Tensor(self.value)

    def contiguous(self):
        return self

    def view(self, dtype):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return str(self.value).encode()


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
        def save(value, path):
            self.saved.append(copy.deepcopy(value))
            Path(path).touch()
        namespace = dict(os=os, hashlib=hashlib, json=json,
                         _active_capture=ContextVar("test_capture", default=None),
                         Path=Path, Optional=__import__('typing').Optional,
                         List=list, logger=logging.getLogger(__name__),
                         ForwardBatch=Batch, LogitsProcessorOutput=Logits,
                         PPProxyTensors=type("Proxy", (), {}),
                         torch=SimpleNamespace(Tensor=Tensor, uint8=object(), save=save))
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.cls = namespace["TensorDumper"]
        bind = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_bind_round62_forward")
        namespace["wraps"] = wraps
        exec(compile(ast.Module(body=[bind], type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.bind = namespace["_bind_round62_forward"]
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

    def test_direct_forward_bypassing_module_call_is_captured(self):
        d = self.dumper(True)
        model = SimpleNamespace(forward=lambda batch: Logits(batch.input_ids.value * 11))
        self.bind(model, d)
        for token in (7, 11):
            self.assertEqual(model.forward(Batch(token)).next_token_logits.value, token * 11)
        self.assertEqual(len(self.saved), 2)
        self.assertEqual(self.saved[1]["capture.forward_batch_info.input_ids"].value, 11)

    def test_direct_reference_forward_is_captured_once(self):
        d = self.dumper(False)
        model = SimpleNamespace(forward=lambda batch: Logits(77))
        self.bind(model, d, reference=True)
        batch = Batch(7)
        batch.rids = ["reference"]
        model.forward(batch)
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(self.saved[0]["capture.transaction"]["batch"]["request_ids"], ["reference"])

    def test_wrong_outer_output_fails(self):
        d = self.dumper(True)
        d.begin_complete_forward(None, (Batch(7),), {})
        with self.assertRaisesRegex(RuntimeError, "LogitsProcessorOutput"):
            d.end_complete_forward(None, (), {}, Tensor(1))

    def test_duplicate_sequence_fails(self):
        d = self.dumper(True)
        d.begin_complete_forward(None, (Batch(7),), {})
        d.end_complete_forward(None, (), {}, Logits(7))
        d._forward_pass_id = 0
        with self.assertRaisesRegex(RuntimeError, "duplicate forward"):
            d.begin_complete_forward(None, (Batch(8),), {})

    def test_output_without_entry_fails(self):
        with self.assertRaisesRegex(RuntimeError, "without matching entry"):
            self.dumper(True).end_complete_forward(None, (), {}, Logits(1))


if __name__ == "__main__":
    unittest.main()
