"""Contracts for the opt-in real-prefill route-group fixture capture."""
import builtins
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch


class Backend:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def execute_group(self, layer, group, hidden, weights, identity):
        self.calls.append((layer, identity.allocation_epoch))
        if self.fail:
            raise ValueError("provider failure")
        torch = __import__("torch")
        rows = torch.tensor(group.rows, dtype=torch.int64)
        return tuple(hidden.index_select(0, rows) + 1)


def group(rows=(0, 1), logical=(3, 7), slots=((0, 1), (1, 0))):
    return NS(rows=rows, logical_experts=logical, remapped_ids=slots)


def identity(request="request-a", epoch=4, allocation=9):
    return NS(request_id=request, placement_epoch=epoch,
              allocation_epoch=allocation)


class PrefillGroupCaptureOffTests(unittest.TestCase):
    def test_default_off_does_not_import_torch_replace_backend_or_touch_paths(self):
        from migration_bridge.round9.runtime.prefill_group_capture import bind_capture
        backend = Backend()
        original = backend.execute_group
        recorder = NS(request_key=("r", 0, 64), chunk=(0, 32))
        real_import = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("OFF capture imported torch")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {}, clear=True), patch.object(
                builtins, "__import__", side_effect=reject_torch):
            capture = bind_capture(backend, recorder)
        self.assertFalse(capture.enabled)
        self.assertIs(backend.execute_group.__func__, original.__func__)


class PrefillGroupCapturePureContractTests(unittest.TestCase):
    def test_enabled_binding_records_host_only_route_tape_without_torch(self):
        from migration_bridge.round9.runtime.prefill_group_capture import bind_capture
        class HostBackend:
            def execute_group(self, layer, group, hidden, weights, identity):
                return ("provider",)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "real-prefill"
            backend = HostBackend()
            recorder = NS(request_key=("request-a", 4, 64), chunk=(0, 32))
            capture = bind_capture(
                backend, recorder,
                {"ROUND9_PREFILL_GROUP_CAPTURE": str(prefix)},
            )
            real_import = builtins.__import__
            def reject_torch(name, *args, **kwargs):
                if name == "torch":
                    raise AssertionError("host route tape imported torch")
                return real_import(name, *args, **kwargs)
            with patch.object(builtins, "__import__", side_effect=reject_torch):
                value = backend.execute_group(3, group(), object(), object(), identity())
            self.assertEqual(value, ("provider",))
            rows=[json.loads(line) for line in Path(str(prefix)+".routes.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["kind"], "header")
            self.assertEqual(rows[1]["kind"], "group")
            self.assertEqual(rows[1]["sequence"], 0)
            self.assertEqual(rows[1]["recorded_group_count"], 1)
            self.assertEqual(rows[1]["request_id"], "request-a")
            self.assertEqual(rows[1]["chunk"], [0, 32])
            self.assertEqual(rows[1]["logical_layer"], 3)
            self.assertEqual(rows[1]["remapped_ids"], [[0, 1], [1, 0]])
            self.assertFalse(capture.route_tape.failed)

    def test_route_tape_records_limit_failure_once_instead_of_dropping(self):
        from migration_bridge.round9.runtime.prefill_group_capture import RouteTape
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"routes.jsonl"
            tape=RouteTape(path,max_groups=1,max_bytes=2048)
            metadata={"request_id":"r","placement_epoch":0,"input_tokens":64,
                "chunk":[0,32],"logical_layer":0,"allocation_epoch":1,
                "rows":[0],"logical_experts":[3],"remapped_ids":[[0]]}
            tape.record(metadata)
            tape.record({**metadata,"allocation_epoch":2,"rows":[1]})
            before=path.read_bytes();tape.record({**metadata,"allocation_epoch":3,"rows":[2]})
            self.assertEqual(path.read_bytes(),before)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["kind"] for row in rows],["header","group","failure"])
            self.assertEqual(rows[-1]["status"],"FAILED_ROUTE_TAPE_LIMIT")
            self.assertEqual(rows[-1]["recorded_group_count"],1)
            self.assertTrue(tape.failed)

    def test_decode_group_is_ignored_without_importing_torch(self):
        from migration_bridge.round9.runtime.prefill_group_capture import bind_capture
        class DecodeBackend:
            def execute_group(self, layer, group, hidden, weights, identity):
                return ("provider-result",)
        with tempfile.TemporaryDirectory() as directory:
            backend = DecodeBackend()
            recorder = NS(request_key=("request-a", 4, 64), chunk=None)
            capture = bind_capture(
                backend, recorder,
                {"ROUND9_PREFILL_GROUP_CAPTURE": str(Path(directory) / "fixture")},
            )
            real_import = builtins.__import__
            def reject_torch(name, *args, **kwargs):
                if name == "torch":
                    raise AssertionError("decode capture imported torch")
                return real_import(name, *args, **kwargs)
            with patch.object(builtins, "__import__", side_effect=reject_torch):
                value = backend.execute_group(0, group(), object(), object(), identity())
            self.assertEqual(value, ("provider-result",))
            self.assertFalse(capture.failed)
            self.assertFalse(capture.complete)

    def test_group_metadata_reconstructs_logical_ids_and_token_span(self):
        from migration_bridge.round9.runtime.prefill_group_capture import group_metadata
        row = group_metadata(
            0, group(rows=(5, 6), logical=(3, 7), slots=((0, 1), (1, 0))),
            identity(allocation=9), ("request-a", 4, 64), (0, 32),
        )
        self.assertEqual(row["logical_experts"], [3, 7])
        self.assertEqual(row["slot_ids"], [[0, 1], [1, 0]])
        self.assertEqual(row["logical_ids"], [[3, 7], [7, 3]])
        self.assertEqual(row["rows"], [5, 6])
        self.assertEqual(row["token_span"], [5, 7])
        self.assertEqual(row["request_token_span"], [5, 7])

    def test_group_metadata_rejects_bounds_slots_and_identity_mismatch(self):
        from migration_bridge.round9.runtime.prefill_group_capture import group_metadata
        cases = (
            (group(rows=tuple(range(33)), slots=tuple((0, 1) for _ in range(33))), "32"),
            (group(rows=(1, 0)), "contiguous"),
            (group(slots=((0, 2), (1, 0))), "slot"),
        )
        for bad_group, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                group_metadata(0, bad_group, identity(), ("request-a", 4, 64), (0, 32))
        with self.assertRaisesRegex(ValueError, "identity"):
            group_metadata(0, group(), identity(request="other"),
                           ("request-a", 4, 64), (0, 32))

    def test_sequence_accepts_exactly_two_adjacent_groups_for_one_chunk(self):
        from migration_bridge.round9.runtime.prefill_group_capture import CaptureSequence
        sequence = CaptureSequence()
        one = {"request_id": "r", "placement_epoch": 2,
               "allocation_epoch": 8, "chunk": [0, 32], "logical_layer": 0}
        two = {**one, "allocation_epoch": 9}
        self.assertEqual(sequence.accept(one), 0)
        self.assertEqual(sequence.accept(two), 1)
        self.assertTrue(sequence.complete)
        self.assertIsNone(sequence.accept({**two, "allocation_epoch": 10}))
        for change, message in (
            ({"request_id": "other"}, "identity"),
            ({"chunk": [32, 32]}, "chunk"),
            ({"allocation_epoch": 10}, "adjacent"),
        ):
            candidate = {**one, "allocation_epoch": 9, **change}
            broken = CaptureSequence(); broken.accept(one)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                broken.accept(candidate)

    def test_enabled_path_contract_requires_absolute_distinct_fresh_files(self):
        from migration_bridge.round9.runtime.prefill_group_capture import CapturePaths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"ROUND9_PREFILL_GROUP_CAPTURE": str(root / "real-prefill")}
            paths = CapturePaths.from_environment(env)
            self.assertEqual(paths.tensor, root / "real-prefill.pt")
            self.assertEqual(paths.receipt, root / "real-prefill.receipt.json")
            self.assertEqual(paths.routes, root / "real-prefill.routes.jsonl")
            with self.assertRaisesRegex(ValueError, "absolute"):
                CapturePaths.from_environment({"ROUND9_PREFILL_GROUP_CAPTURE": "relative"})
            paths.tensor.write_bytes(b"occupied")
            with self.assertRaises(FileExistsError):
                CapturePaths.from_environment(env)


class PrefillGroupCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = __import__("torch")
        except ImportError:
            raise unittest.SkipTest("numeric capture contracts require real torch")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.tensor_path = root / "groups.pt"
        self.receipt_path = root / "groups.receipt.json"
        self.env = {
            "ROUND9_PREFILL_GROUP_CAPTURE": str(root / "groups"),
        }
        self.backend = Backend()
        self.recorder = NS(request_key=("request-a", 4, 64), chunk=(0, 32))

    def tearDown(self):
        self.directory.cleanup()

    def bind(self):
        from migration_bridge.round9.runtime.prefill_group_capture import bind_capture
        with patch.dict(os.environ, self.env, clear=True):
            return bind_capture(self.backend, self.recorder)

    def tensors(self):
        hidden = self.torch.arange(16, dtype=self.torch.bfloat16).reshape(4, 4)
        weights = self.torch.tensor(
            [[.1, .9], [.2, .8], [.3, .7], [.4, .6]],
            dtype=self.torch.float32,
        )
        return hidden, weights

    def test_two_adjacent_real_results_publish_exact_tensor_file_then_sha_receipt(self):
        capture = self.bind()
        hidden, weights = self.tensors()
        first = group()
        second = group(rows=(2, 3), logical=(11, 13), slots=((1, 0), (0, 1)))
        first_result = self.backend.execute_group(
            0, first, hidden, weights, identity(allocation=9))
        second_result = self.backend.execute_group(
            0, second, hidden, weights, identity(allocation=10))

        self.assertTrue(capture.complete)
        self.assertEqual(len(self.backend.calls), 2)
        receipt = json.loads(self.receipt_path.read_text())
        raw = self.tensor_path.read_bytes()
        self.assertEqual(receipt["status"], "COMPLETE_REAL_CONSUMER_GROUP_CAPTURE")
        self.assertEqual(receipt["tensor_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(receipt["request"], {
            "request_id": "request-a", "placement_epoch": 4,
            "input_tokens": 64, "chunk": [0, 32],
        })
        self.assertTrue(receipt["diagnostic_device_copy_and_sync"])
        self.assertEqual([row["allocation_epoch"] for row in receipt["groups"]], [9, 10])
        self.assertEqual(receipt["groups"][1]["token_span"], [2, 4])

        payload = self.torch.load(self.tensor_path, weights_only=True)
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(len(payload["groups"]), 2)
        saved = payload["groups"][0]
        self.assertTrue(self.torch.equal(saved["hidden_states"], hidden[:2]))
        self.assertTrue(self.torch.equal(saved["topk_weights"], weights[:2]))
        self.assertTrue(self.torch.equal(
            saved["topk_slot_ids"], self.torch.tensor([[0, 1], [1, 0]], dtype=self.torch.int32)))
        self.assertTrue(self.torch.equal(
            saved["topk_logical_ids"], self.torch.tensor([[3, 7], [7, 3]], dtype=self.torch.int64)))
        expected = self.torch.stack(first_result).float()
        self.assertEqual(saved["reference_contribution_fp32"].dtype, self.torch.float32)
        self.assertTrue(self.torch.equal(saved["reference_contribution_fp32"], expected))

    def test_wrong_request_or_nonadjacent_allocation_fails_closed_without_receipt(self):
        capture = self.bind()
        hidden, weights = self.tensors()
        self.backend.execute_group(0, group(), hidden, weights, identity(allocation=9))
        self.backend.execute_group(
            0, group(rows=(2, 3)), hidden, weights,
            identity(request="request-b", allocation=11),
        )
        self.assertTrue(capture.failed)
        self.assertIn("identity", capture.failure)
        self.assertEqual(len(self.backend.calls), 2)
        self.assertFalse(self.tensor_path.exists())
        self.assertFalse(self.receipt_path.exists())

    def test_chunk_switch_fails_closed_and_does_not_interrupt_provider(self):
        capture = self.bind()
        hidden, weights = self.tensors()
        self.backend.execute_group(0, group(), hidden, weights, identity(allocation=9))
        self.recorder.chunk = (32, 32)
        value = self.backend.execute_group(
            0, group(rows=(2, 3)), hidden, weights, identity(allocation=10))
        self.assertEqual(len(value), 2)
        self.assertTrue(capture.failed)
        self.assertIn("chunk", capture.failure)
        self.assertFalse(self.receipt_path.exists())

    def test_layer_and_row_bounds_are_fail_closed(self):
        for layer, bad_group, message in (
            (1, group(), None),
            (0, group(rows=tuple(range(33)), slots=tuple((0, 1) for _ in range(33))), "32"),
            (0, group(rows=(1, 0)), "contiguous"),
            (0, group(slots=((0, 2), (1, 0))), "slot"),
        ):
            with self.subTest(layer=layer, message=message):
                self.tearDown(); self.setUp()
                capture = self.bind()
                hidden = self.torch.zeros((33, 4), dtype=self.torch.bfloat16)
                weights = self.torch.ones((33, 2), dtype=self.torch.bfloat16)
                self.backend.execute_group(layer, bad_group, hidden, weights, identity())
                if message is None:
                    self.assertFalse(capture.failed)
                else:
                    self.assertTrue(capture.failed)
                    self.assertIn(message, capture.failure)
                self.assertFalse(self.receipt_path.exists())

    def test_capture_never_masks_provider_exception_and_never_overcaptures(self):
        capture = self.bind()
        hidden, weights = self.tensors()
        self.backend.execute_group(0, group(), hidden, weights, identity(allocation=9))
        self.backend.execute_group(
            0, group(rows=(2, 3)), hidden, weights, identity(allocation=10))
        before = self.tensor_path.read_bytes()
        self.backend.execute_group(0, group(), hidden, weights, identity(allocation=11))
        self.assertEqual(self.tensor_path.read_bytes(), before)
        self.assertEqual(len(self.backend.calls), 3)
        self.backend.fail = True
        with self.assertRaisesRegex(ValueError, "provider failure"):
            self.backend.execute_group(0, group(), hidden, weights, identity(allocation=12))
        self.assertTrue(capture.complete)

    def test_enabled_paths_must_be_fresh_files(self):
        from migration_bridge.round9.runtime.prefill_group_capture import bind_capture
        self.tensor_path.write_bytes(b"occupied")
        with patch.dict(os.environ, self.env, clear=True), self.assertRaises(FileExistsError):
            bind_capture(self.backend, self.recorder)


if __name__ == "__main__":
    unittest.main()
