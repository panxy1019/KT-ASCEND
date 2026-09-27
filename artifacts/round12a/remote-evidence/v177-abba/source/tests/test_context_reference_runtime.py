import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from migration_bridge.round9.runtime import context_reference_runtime as api


class ContextReferenceProfileTests(unittest.TestCase):
    def profile(self, mode="BF16_COMPACT"):
        return {
            "schema": 1,
            "mode": mode,
            "device_gate": {"path": "/round9/evidence/context.json", "sha256": "a" * 64},
        }

    def test_profile_is_explicit_hashed_and_default_off(self):
        for mode in ("BF16_COMPACT", "FP8_HOST_REFERENCE"):
            self.assertEqual(api.parse_context_reference_profile(self.profile(mode))["mode"], mode)
        for override in ({"schema": True}, {"mode": "FP8_NATIVE"},
                         {"device_gate": None}, {"extra": True}):
            document = self.profile()
            document.update(override)
            with self.assertRaises(ValueError):
                api.parse_context_reference_profile(document)
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(api.load_context_reference_profile())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(self.profile()))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            env = {"ROUND9_CONTEXT_REFERENCE_PROFILE": str(path),
                   "ROUND9_CONTEXT_REFERENCE_SHA256": digest}
            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(api.load_context_reference_profile()["mode"], "BF16_COMPACT")
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "changed"):
                    api.load_context_reference_profile()


class ContextReferenceLifecycleTests(unittest.TestCase):
    def runtime(self):
        self.calls = []
        self.events = []
        def consume(identity, mode, layer, q, keys, values, selected, positions, reference):
            self.calls.append((identity, mode, layer, positions))
            return f"output-{layer}"
        return api.ContextReferenceRuntime(
            "FP8_HOST_REFERENCE", layers=(3, 7), consume_rows=consume,
            event=self.events.append,
        )

    def test_chunks_keep_allocation_epoch_and_reused_id_gets_new_epoch(self):
        runtime = self.runtime(); first = object()
        runtime.begin("same", first)
        self.assertEqual(runtime.consume(3, None, None, None, None, [0], None), "output-3")
        self.assertEqual(runtime.consume(7, None, None, None, None, [0], None), "output-7")
        runtime.after(first, finished=False)
        first_epoch = self.calls[0][0].allocation_epoch

        runtime.begin("same", first)
        runtime.consume(3, None, None, None, None, [1], None)
        runtime.consume(7, None, None, None, None, [1], None)
        runtime.after(first, finished=True)
        self.assertEqual(self.calls[-1][0].allocation_epoch, first_epoch)

        reused = object()
        runtime.begin("same", reused)
        runtime.consume(3, None, None, None, None, [0], None)
        runtime.consume(7, None, None, None, None, [0], None)
        runtime.after(reused, finished=True)
        self.assertGreater(self.calls[-1][0].allocation_epoch, first_epoch)
        self.assertEqual([event["status"] for event in self.events], [
            "CONTEXT_REFERENCE_BATCH_DRAINED",
            "CONTEXT_REFERENCE_BATCH_DRAINED", "CONTEXT_REFERENCE_REQUEST_DRAINED",
            "CONTEXT_REFERENCE_BATCH_DRAINED", "CONTEXT_REFERENCE_REQUEST_DRAINED",
        ])

    def test_layer_order_switch_abort_and_failure_are_fail_closed(self):
        runtime = self.runtime(); request = object()
        runtime.begin("r", request)
        with self.assertRaisesRegex(RuntimeError, "ordered"):
            runtime.consume(7, None, None, None, None, [0], None)
        self.assertEqual(runtime.state, "BROKEN_RELOAD_REQUIRED")

        runtime = self.runtime(); request = object(); runtime.begin("r", request)
        with self.assertRaisesRegex(RuntimeError, "active"):
            runtime.begin("other", object())
        runtime.abort(request)
        self.assertEqual(runtime.state, "READY")
        self.assertEqual(self.events[-1]["status"], "CONTEXT_REFERENCE_ABORTED")

        runtime = self.runtime(); request = object(); runtime.begin("r", request)
        runtime.consume(3, None, None, None, None, [0], None)
        with self.assertRaisesRegex(RuntimeError, "all QSA layers"):
            runtime.after(request, finished=False)
        self.assertEqual(runtime.state, "BROKEN_RELOAD_REQUIRED")


class ContextReferenceTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("PyTorch is required for QSA consumer tests") from error
        cls.qsa = importlib.import_module("migration_bridge.round9.runtime.qsa")
        from migration_bridge.round9.runtime.context_profile import ContextStateIdentity
        cls.identity = ContextStateIdentity("request", None, 3, "QSA_KV")

    def tensors(self):
        torch = self.torch
        torch.manual_seed(11)
        q = torch.randn(2, 24, 256, dtype=torch.float32).to(torch.bfloat16)
        keys = torch.randn(8, 2, 256, dtype=torch.float32).to(torch.bfloat16)
        values = torch.randn(8, 2, 256, dtype=torch.float32).to(torch.bfloat16)
        selected = torch.tensor([[0, 1, 3, -1], [5, 1, 5, 7]], dtype=torch.int64)
        positions = torch.tensor([3, 7], dtype=torch.int64)
        return q, keys, values, selected, positions

    def test_bf16_compact_real_consumer_is_exact_for_rows_duplicates_and_padding(self):
        q, keys, values, selected, positions = self.tensors()
        expected = self.qsa.sparse_gqa(q, keys, values, selected)
        actual = api._consume_reference_rows(
            self.identity, "BF16_COMPACT", 3, q, keys, values,
            selected, positions, self.qsa.sparse_gqa)
        self.assertTrue(self.torch.equal(actual, expected))

    def test_fp8_host_reference_real_consumer_is_finite_and_bounded(self):
        q, keys, values, selected, positions = self.tensors()
        expected = self.qsa.sparse_gqa(q, keys, values, selected)
        actual = api._consume_reference_rows(
            self.identity, "FP8_HOST_REFERENCE", 3, q, keys, values,
            selected, positions, self.qsa.sparse_gqa)
        self.assertTrue(bool(self.torch.isfinite(actual).all()))
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        self.assertLess(float(error), 0.10)


if __name__ == "__main__":
    unittest.main()
