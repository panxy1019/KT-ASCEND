import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT))
    return module


class Round13BSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load("build_round13_b")
        cls.component_builder = load("build_round13_b_component")
        cls.model_builder = load("build_round13_b_snapshot")
        cls.preflight = load("round13_b_offline_preflight")
        cls.driver = load("round13_b_model_driver")
        cls.parent = ROOT / "frozen-parent-v177-copy"
        cls.fixture10 = json.loads(
            (ROOT / "frozen-inputs/round10-t1-fixture.json").read_text()
        )
        cls.fixture11 = json.loads(
            (ROOT / "frozen-inputs/round11-t1-fixture.json").read_text()
        )

    def test_decode_candidate_is_one_fail_closed_runtime_change(self):
        path = self.parent / "source/runtime/context_reference_runtime.py"
        original = path.read_text()
        patched = self.builder.patch_context_runtime(original)
        compile(patched, "context_reference_runtime.py", "exec")
        self.assertIn("qualified_prefill", patched)
        self.assertIn("qualified_decode", patched)
        self.assertIn('"DIRECT_BF16_ROW"', patched)
        self.assertIn("q.shape[0] == 1", patched)
        self.assertIn("q.shape[0] > 1", patched)
        self.assertIn("_consume_reference_rows", patched)
        with self.assertRaises(ValueError):
            self.builder.patch_context_runtime(patched)

    def test_source_gate_accepts_only_candidate_runtime_tuple(self):
        runtime = self.builder.patch_context_runtime(
            (self.parent / "source/runtime/context_reference_runtime.py").read_text()
        ).encode()
        runtime_sha = hashlib.sha256(runtime).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source/runtime").mkdir(parents=True)
            (root / "overlay-python/sglang/srt/layers/round9").mkdir(parents=True)
            shutil.copy2(self.parent / "ROUND12_QSA_SOURCE_BINDING.json", root)
            shutil.copy2(
                self.parent / "source/runtime/context_reference_gate.py",
                root / "source/runtime/context_reference_gate.py",
            )
            shutil.copy2(
                self.parent
                / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
                root / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
            )
            delta = self.builder.bind_candidate_runtime(root, runtime_sha)
            binding = json.loads(
                (root / "ROUND12_QSA_SOURCE_BINDING.json").read_text()
            )
            self.assertEqual(
                binding["candidate_context_reference_runtime_sha256"], runtime_sha
            )
            self.assertEqual(delta["candidate_runtime_sha256"], runtime_sha)
            for relative in (
                "source/runtime/context_reference_gate.py",
                "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
            ):
                gate = (root / relative).read_text()
                self.assertIn(runtime_sha, gate)
                self.assertNotIn(delta["baseline_runtime_sha256"], gate)
            with self.assertRaises(ValueError):
                self.builder.bind_candidate_runtime(root, "0" * 64)

    @unittest.skipIf(torch is None, "local system Python has no Torch")
    def test_qualified_decode_uses_original_device_tensors_and_is_causal(self):
        patched = self.builder.patch_context_runtime(
            (self.parent / "source/runtime/context_reference_runtime.py").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "candidate_runtime"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "context_profile.py").write_text(
                "class ContextStateIdentity:\n"
                "    def __init__(self,*args): self.args=args\n"
            )
            (package / "context_reference_runtime.py").write_text(patched)
            sys.path.insert(0, str(root))
            try:
                runtime = __import__(
                    "candidate_runtime.context_reference_runtime",
                    fromlist=["_consume_approved_candidate"],
                )
                q = torch.randn(1, 24, 256, dtype=torch.bfloat16)
                keys = torch.randn(4, 2, 256, dtype=torch.bfloat16)
                values = torch.randn_like(keys)
                selected = torch.tensor([[0, 1, 2, 3, -1]])
                positions = torch.tensor([3])
                seen = {}

                def consumer(got_q, got_keys, got_values, got_selected):
                    seen.update(
                        q=got_q,
                        keys=got_keys,
                        values=got_values,
                        selected=got_selected,
                    )
                    return got_q.clone()

                result, strategy = runtime._consume_approved_candidate(
                    object(), "BF16_COMPACT", 3, q, keys, values,
                    selected, positions, consumer,
                )
                self.assertEqual(strategy, "DIRECT_BF16_ROW")
                self.assertIs(seen["keys"], keys)
                self.assertIs(seen["values"], values)
                self.assertTrue(torch.equal(result, q))
                future = selected.clone()
                future[0, 0] = 4
                with self.assertRaisesRegex(ValueError, "future logical token"):
                    runtime._consume_approved_candidate(
                        object(), "BF16_COMPACT", 3, q, keys, values,
                        future, positions, consumer,
                    )
            finally:
                sys.path.remove(str(root))
                for name in list(sys.modules):
                    if name == "candidate_runtime" or name.startswith(
                        "candidate_runtime."
                    ):
                        del sys.modules[name]

    def test_component_bundle_freezes_two_independent_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            self.component_builder.build(
                parent=self.parent, inputs=ROOT, output=output
            )
            contract = json.loads((output / "CONTRACT.json").read_text())
            self.assertEqual(contract["versions"], {"215": "A", "216": "B"})
            self.assertEqual(contract["contexts"], [64, 128, 512])
            self.assertEqual(contract["performance_gate_candidate_over_baseline"], 0.9)
            self.assertNotEqual(
                contract["candidate_runtime_sha256"],
                contract["baseline_runtime_sha256"],
            )
            expected = json.loads((output / "source-sha256.json").read_text())
            actual = {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*"))
                if path.is_file()
                and path.name != "source-sha256.json"
                and "__pycache__" not in path.parts
            }
            self.assertEqual(actual, expected)

    def test_model_schedules_match_six_process_budget(self):
        fixtures = {"round10": self.fixture10, "round11": self.fixture11}
        for version, (implementation, mode, block) in self.driver.ROLES.items():
            if mode == "smoke":
                rows = self.driver.smoke_schedule(fixtures, version, implementation)
                self.assertEqual(len(rows), 8)
                self.assertEqual(sum(row["phase"] == "cancel" for row in rows), 1)
                self.assertEqual(
                    {row["case_id"] for row in rows},
                    {
                        "canary_marker",
                        "canary_math391",
                        "canary_json",
                        "canary_quote",
                        "canary_transform",
                        "edge128",
                        "cancel256",
                    },
                )
            else:
                rows = self.driver.performance_schedule(
                    fixtures, version, implementation, block
                )
                self.assertEqual(len(rows), 20)
                self.assertEqual(
                    [
                        sum(
                            row["workload"] == workload
                            and row["phase"] == "measurement"
                            for row in rows
                        )
                        for workload in self.driver.WORKLOADS
                    ],
                    [3, 3, 3, 3, 3],
                )

    def test_qsa_strategy_contract_distinguishes_only_decode(self):
        events = []
        for layer in self.driver.QSA_LAYERS:
            events.extend(
                [
                    {
                        "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                        "request_id": "r",
                        "layer": layer,
                        "strategy": "VECTORIZED",
                    },
                    {
                        "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                        "request_id": "r",
                        "layer": layer,
                        "strategy": "DIRECT_BF16_ROW",
                    },
                ]
            )
        receipt = self.driver.validate_qsa("r", events, "candidate")
        self.assertEqual(receipt["decode_strategy"], "DIRECT_BF16_ROW")
        with self.assertRaises(RuntimeError):
            self.driver.validate_qsa("r", events, "baseline")

    def test_model_snapshots_change_only_candidate_runtime_tuple(self):
        class Args:
            parent = self.parent
            inputs = ROOT

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            built = {}
            for version in (217, 218, 219, 220, 221, 222):
                args = Args()
                args.version = version
                args.output = directory / str(version)
                self.model_builder.build(args)
                contract = json.loads(
                    (args.output / "ROUND13_MODEL_CONTRACT.json").read_text()
                )
                built[version] = contract
                expected = json.loads(
                    (args.output / "round13-source-sha256.json").read_text()
                )
                actual = self.model_builder.source_manifest(args.output)
                self.assertEqual(actual, expected)
                self.assertEqual(contract["profile_threshold"], 513)
                self.assertEqual(contract["default_threshold"], 129)
                self.assertEqual(
                    json.loads((args.output / "framework-profile.json").read_text())["prefill"]["threshold_tokens"],
                    513,
                )
                receipt = (
                    ROOT.parent
                    / "round12a/remote-evidence/v174-abba/context-reference-device-gate.json"
                )
                checked = self.preflight.run(args.output, receipt)
                self.assertEqual(
                    checked["status"], "PASS_ROUND13_B_OFFLINE_PREFLIGHT"
                )
                self.assertEqual(
                    checked["source_gate"]["candidate_context_reference_runtime_sha256"],
                    contract["runtime_sha256"],
                )
            baseline_runtime = {
                built[version]["runtime_sha256"] for version in (217, 219, 222)
            }
            candidate_runtime = {
                built[version]["runtime_sha256"] for version in (218, 220, 221)
            }
            self.assertEqual(len(baseline_runtime), 1)
            self.assertEqual(len(candidate_runtime), 1)
            self.assertNotEqual(baseline_runtime, candidate_runtime)
            self.assertTrue(
                all(built[version]["runtime_delta_sha256"] is None for version in (217, 219, 222))
            )
            self.assertTrue(
                all(built[version]["runtime_delta_sha256"] for version in (218, 220, 221))
            )


if __name__ == "__main__":
    unittest.main()
