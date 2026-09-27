import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load("round13_a_model_driver")
B = load("build_round13_a_snapshot")


class Round13ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = ROOT / "frozen-parent-v177-copy"
        cls.fixture10 = __import__("json").loads(
            (ROOT / "frozen-inputs/round10-t1-fixture.json").read_text())
        cls.fixture11 = __import__("json").loads(
            (ROOT / "frozen-inputs/round11-t1-fixture.json").read_text())

    def test_every_arm_has_exact_five_workloads_and_budget(self):
        fixtures = {"round10": self.fixture10, "round11": self.fixture11}
        for version, (strategy, block) in A.EXPECTED_ORDER.items():
            rows = A.performance_schedule(fixtures, version, strategy, block)
            quality = A.quality_schedule(fixtures, version, strategy, block)
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(quality), 4)
            self.assertEqual({row["workload"] for row in rows}, set(A.WORKLOADS))
            for workload in A.WORKLOADS:
                group = [row for row in rows if row["workload"] == workload]
                self.assertEqual([row["phase"] for row in group],
                                 ["warmup", "measurement", "measurement", "measurement"])
            self.assertEqual(sum(row["phase"] == "cancel" for row in quality),
                             1 if block == "A" else 0)

    def test_workload_lengths_and_output_budgets_are_frozen(self):
        fixtures = {"round10": self.fixture10, "round11": self.fixture11}
        rows = A.performance_schedule(fixtures, 196, "uniform", "A")
        expected = {
            "W0": (64, 2), "W1": (128, 16), "W2": (128, 128),
            "W3": (512, 16), "W4": (64, 32),
        }
        for row in rows:
            self.assertEqual((row["input_tokens"], row["decode_tokens"]),
                             expected[row["workload"]])
            self.assertTrue(row["request"]["stream"])
            self.assertTrue(row["request"]["sampling_params"]["ignore_eos"])

    def test_parent_patchers_compile_and_are_fail_closed(self):
        framework = self.parent / "overlay-python/sglang/srt/layers/round9/framework_hooks.py"
        qwen = self.parent / "overlay-python/sglang/srt/models/qwen4_exp.py"
        launcher = self.parent / "source/tools/launch_hybrid.py"
        prefill = self.parent / "overlay-python/sglang/srt/layers/round9/prefill_binding.py"
        hybrid_probe = self.parent / "overlay-python/sglang/srt/layers/round9/hybrid_probe.py"
        patched_framework = B.patch_framework(framework.read_text())
        patched_qwen = B.patch_qwen(qwen.read_text())
        patched_prefill = B.patch_prefill_binding(prefill.read_text())
        patched_hybrid_probe = B.patch_hybrid_probe(hybrid_probe.read_text())
        patched_launcher = B.patch_launcher(
            launcher.read_text(), 196, "uniform", "A", "uniform-a")
        compile(patched_framework, "framework_hooks.py", "exec")
        compile(patched_qwen, "qwen4_exp.py", "exec")
        compile(patched_prefill, "prefill_binding.py", "exec")
        compile(patched_hybrid_probe, "hybrid_probe.py", "exec")
        compile(patched_launcher, "launch_hybrid.py", "exec")
        self.assertIn("collect_physical_layers", patched_framework)
        self.assertIn("require_component_gate", patched_qwen)
        self.assertIn("H513 Hybrid <=512", patched_prefill)
        self.assertIn("layers[min(layers)]", patched_prefill)
        self.assertIn("observed_layers", patched_hybrid_probe)
        self.assertIn("ROUND13_PLACEMENT_STRATEGY=uniform", patched_launcher)
        self.assertIn("digest(root/'round13-a-uniform-a-v196/PLACEMENT_PROFILES.json')",
                      patched_launcher)
        with self.assertRaises(ValueError):
            B.patch_framework(patched_framework)

    def test_qsa_source_binding_accepts_only_the_exact_placement_consumer(self):
        qwen = self.parent / "overlay-python/sglang/srt/models/qwen4_exp.py"
        candidate = B.patch_qwen(qwen.read_text()).encode()
        candidate_sha = hashlib.sha256(candidate).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source/runtime").mkdir(parents=True)
            (root / "overlay-python/sglang/srt/layers/round9").mkdir(parents=True)
            shutil.copy2(self.parent / "ROUND12_QSA_SOURCE_BINDING.json", root)
            shutil.copy2(self.parent / "source/runtime/context_reference_gate.py",
                         root / "source/runtime/context_reference_gate.py")
            shutil.copy2(
                self.parent / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
                root / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py")
            delta = B.bind_round13_qsa_consumer(root, candidate_sha)
            binding = json.loads((root / "ROUND12_QSA_SOURCE_BINDING.json").read_text())
            self.assertEqual(binding["candidate_qwen4_exp_sha256"], candidate_sha)
            self.assertEqual(delta["round13_qwen4_exp_sha256"], candidate_sha)
            gate = (root / "source/runtime/context_reference_gate.py").read_text()
            self.assertIn(candidate_sha, gate)
            self.assertNotIn(delta["round12_qwen4_exp_sha256"], gate)
            overlay_gate = (root / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py").read_text()
            self.assertIn(candidate_sha, overlay_gate)
            self.assertEqual(len(delta["round13_context_reference_gate_sha256"]), 2)
            with self.assertRaises(ValueError):
                B.bind_round13_qsa_consumer(root, "0" * 64)


if __name__ == "__main__":
    unittest.main()
