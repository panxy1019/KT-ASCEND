from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


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


class Round14aResidentPlacementTest(unittest.TestCase):
    def setUp(self):
        from round14a_resident_profiles import build_document
        self.document = build_document()

    def test_fplus1_and_uplus42_have_the_same_2016_expert_budget(self):
        profiles = self.document["profiles"]
        self.assertEqual(set(profiles), {"fplus1", "uplus42"})
        self.assertEqual(profiles["fplus1"]["per_layer_counts"], [512] + [32] * 47)
        self.assertEqual(profiles["uplus42"]["per_layer_counts"], [42] * 48)
        for profile in profiles.values():
            self.assertEqual(profile["budget_experts"], 2016)
            self.assertEqual(sum(profile["per_layer_counts"]), 2016)
            self.assertEqual(profile["nominal_bf16_bytes"], 2016 * 9830400)

    def test_profile_document_and_rows_are_fail_closed(self):
        from round14a_resident_runtime import load_profile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(self.document, indent=2) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_profile(path, digest, "fplus1")
            self.assertEqual(loaded["counts"], [512] + [32] * 47)
            with self.assertRaises(RuntimeError):
                load_profile(path, "0" * 64, "fplus1")
            damaged = json.loads(path.read_text())
            damaged["profiles"]["uplus42"]["selected_logical_experts"][0].pop()
            path.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                load_profile(path, hashlib.sha256(path.read_bytes()).hexdigest(), "uplus42")

    def test_layer_provider_contract_covers_pure_npu_and_mixed(self):
        from round14a_resident_runtime import load_profile, validate_layer_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(self.document, indent=2) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fplus = load_profile(path, digest, "fplus1")
            self.assertEqual(validate_layer_config(None, 0, fplus)["provider"], "PURE_NPU")
            self.assertEqual(
                validate_layer_config(Config(1, fplus["rows"][1]), 1, fplus)["provider"],
                "KTEP_MIXED",
            )
            uplus = load_profile(path, digest, "uplus42")
            self.assertEqual(
                validate_layer_config(Config(47, uplus["rows"][47]), 47, uplus)["resident_capacity"],
                42,
            )
            with self.assertRaises(RuntimeError):
                validate_layer_config(None, 47, uplus)

    def test_profile_hashes_are_stable_and_distinct(self):
        profiles = self.document["profiles"]
        self.assertNotEqual(profiles["fplus1"]["profile_sha256"],
                            profiles["uplus42"]["profile_sha256"])
        for profile in profiles.values():
            rows = profile["selected_logical_experts"]
            encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(profile["profile_sha256"], hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    unittest.main()
