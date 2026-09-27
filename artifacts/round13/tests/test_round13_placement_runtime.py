import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("r13_runtime", ROOT / "round13_placement_runtime.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class Config:
    def __init__(self, layer, row):
        class Scalar:
            def __init__(self, value): self.value = value
            def item(self): return self.value
        class Indices:
            def __init__(self, values): self.values = values
            def reshape(self, *shape): return self
            def tolist(self): return self.values
        class Mask:
            def __init__(self, values): self.values = values
            def sum(self): return Scalar(len(self.values))
            def nonzero(self): return Indices(self.values)
        self.layer_idx = layer
        self.gpu_experts_mask = Mask(list(row))
        self.cpuinfer_threads = 128
        self.threadpool_count = 4
        self.numa_nodes = [2, 3, 4, 7]
        self.method = "LLAMAFILE"
        self.max_deferred_experts_per_token = 0
        self.kt_enable_dynamic_expert_update = False
        self.gpu_prefill_token_threshold = 0


class TestRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "PLACEMENT_PROFILES.json"
        cls.sha = hashlib.sha256(cls.path.read_bytes()).hexdigest()

    def test_load_all_frozen_profiles(self):
        expected = {
            "uniform": (32, 32), "frequency": (0, 512),
            "front_loading": (0, 512), "random_seed42": (18, 44),
        }
        for name, bounds in expected.items():
            profile = R.load_profile(self.path, self.sha, name)
            self.assertEqual(sum(profile["counts"]), 1536)
            self.assertEqual((min(profile["counts"]), max(profile["counts"])), bounds)

    def test_sha_and_budget_fail_closed(self):
        with self.assertRaises(RuntimeError):
            R.load_profile(self.path, "0" * 64, "uniform")
        document = json.loads(self.path.read_text())
        document["profiles"]["uniform"]["selected_logical_experts"][0].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(document))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                R.load_profile(path, digest, "uniform")

    def test_q0_q512_and_nonuniform_layer_contract(self):
        profile = R.load_profile(self.path, self.sha, "frequency")
        self.assertEqual(R.validate_layer_config(None, 0, profile, None)["provider"], "PURE_NPU")
        self.assertEqual(R.validate_layer_config(Config(3, []), 3, profile, None)["provider"], "KTEP_Q0")
        self.assertEqual(R.validate_layer_config(
            Config(2, profile["rows"][2]), 2, profile, None)["resident_capacity"], 242)
        with self.assertRaises(RuntimeError):
            R.validate_layer_config(Config(0, profile["rows"][0]), 0, profile, None)

    def test_provider_identity(self):
        profile = R.load_profile(self.path, self.sha, "front_loading")
        wrapped = lambda method, name: method == "wrapped" and name == "kt_ep"
        self.assertEqual(R.require_method_identity("plain", 0, profile, wrapped)["provider"], "PURE_NPU")
        self.assertEqual(R.require_method_identity("wrapped", 3, profile, wrapped)["provider"], "KTEP")
        with self.assertRaises(RuntimeError):
            R.require_method_identity("wrapped", 0, profile, wrapped)

    def test_component_gate_is_complete_and_numeric(self):
        gate = ROOT / "A_COMPONENT_GATE.json"
        digest = hashlib.sha256(gate.read_bytes()).hexdigest()
        loaded = R.require_component_gate(gate, digest)
        self.assertEqual(set(loaded["results"]), {
            "front_q512", "front_q0", "frequency_q242", "random_q44"})


if __name__ == "__main__":
    unittest.main()
