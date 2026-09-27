import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "artifacts/round14/build_round14_b_snapshot.py"
PARENT = ROOT / "artifacts/round14/a-snapshots/v235"
POLICY = ROOT / "artifacts/round14/round14_bounded_policy.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_round14_b_snapshot", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14BSnapshotBuilderTest(unittest.TestCase):
    def test_contract_profiles_are_strict_and_versioned(self):
        api = load_builder()
        h512 = api.contract(243)
        self.assertEqual(h512["profile"], "H512")
        self.assertEqual(h512["limits"], {
            "max_input_tokens": 512,
            "max_output_tokens": 256,
            "max_total_tokens": 768,
            "hybrid_threshold_tokens": 513,
        })
        h768 = api.contract(246)
        self.assertEqual(h768["profile"], "H768")
        self.assertEqual(h768["limits"]["hybrid_threshold_tokens"], 769)
        self.assertEqual(api.contract(253)["block"], "A")
        self.assertEqual(api.contract(254)["block"], "B")
        self.assertEqual(api.contract(255)["profile"], "H768")
        self.assertEqual(api.contract(256)["profile"], "H768")
        self.assertEqual(api.contract(257)["block"], "A")
        self.assertEqual(api.contract(258)["block"], "B")
        self.assertEqual(api.contract(259)["profile"], "H768")
        self.assertEqual(api.contract(260)["profile"], "H768")
        self.assertEqual(api.contract(261)["block"], "A")
        self.assertEqual(api.contract(262)["block"], "B")
        self.assertEqual(api.contract(263)["profile"], "H768")
        self.assertEqual(api.contract(264)["profile"], "H768")
        self.assertEqual(api.contract(265)["block"], "A")
        self.assertEqual(api.contract(266)["block"], "B")
        self.assertEqual(api.contract(267)["profile"], "H768")
        self.assertEqual(api.contract(268)["profile"], "H768")
        self.assertEqual(api.contract(269)["block"], "A")
        self.assertEqual(api.contract(270)["block"], "B")
        self.assertEqual(api.contract(273)["block"], "B")
        with self.assertRaises(ValueError):
            api.contract(242)

    def test_tokenizer_patch_gates_single_and_batch_before_object_creation(self):
        api = load_builder()
        source = (
            PARENT
            / "overlay-python/sglang/srt/managers/tokenizer_manager.py"
        ).read_text()
        patched = api.render_tokenizer_manager(source)
        self.assertEqual(
            patched.count("from sglang.srt.round14_bounded_policy import"), 1
        )
        self.assertEqual(patched.count("enforce_from_environment("), 2)
        single_validate = patched.index("self._validate_one_request(obj, input_ids)")
        single_gate = patched.index("enforce_from_environment(obj, input_ids)")
        single_create = patched.index("return self._create_tokenized_object(", single_gate)
        self.assertLess(single_validate, single_gate)
        self.assertLess(single_gate, single_create)
        batch_validate = patched.index(
            "self._validate_one_request(obj[i], input_ids_list[i])"
        )
        batch_gate = patched.index(
            "enforce_from_environment(obj[i], input_ids_list[i])"
        )
        batch_create = patched.index("self._create_tokenized_object(", batch_gate)
        self.assertLess(batch_validate, batch_gate)
        self.assertLess(batch_gate, batch_create)

    def test_build_freezes_profile_policy_and_manifest(self):
        api = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            output = temp / "v243"
            driver = temp / "round14_b_driver.py"
            driver.write_text("#!/usr/bin/env python3\n")
            report = api.build(
                PARENT, output, 243,
                driver,
                POLICY,
                ROOT / "artifacts/round14/ROUND14_B_REQUESTS.json",
            )
            self.assertEqual(report["status"], "PASS_ROUND14_B_SNAPSHOT_BUILD")
            profile = json.loads((output / "ROUND14_H_BOUNDED_PROFILE.json").read_text())
            self.assertEqual(profile["hybrid_threshold_tokens"], 513)
            framework = json.loads((output / "framework-profile.json").read_text())
            self.assertEqual(framework["prefill"]["threshold_tokens"], 513)
            manager = output / "overlay-python/sglang/srt/managers/tokenizer_manager.py"
            self.assertIn("enforce_from_environment", manager.read_text())
            self.assertTrue((output / "ROUND14_B_REQUESTS.json").exists())
            manifest = json.loads((output / "round14-source-sha256.json").read_text())
            self.assertEqual(api.tree(output), manifest)
            sys.path.insert(0, str(output / "source/tools"))
            try:
                from install_candidate_overlay import tree_digest, tree_files
                overlay = output / "overlay-python"
                receipt = json.loads(
                    (overlay / "round9-candidate-overlay.json").read_text()
                )
                self.assertEqual(
                    receipt["candidate_tree_sha256"],
                    tree_digest(tree_files(overlay)),
                )
                self.assertEqual(
                    receipt["round14_b"]["status"],
                    "INSTALLED_PRE_DISPATCH_H_BOUNDED_GATE",
                )
            finally:
                sys.path.remove(str(output / "source/tools"))

    def test_repair_snapshot_expands_bounded_host_probe_capacity(self):
        api = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            output = temp / "v273"
            driver = temp / "round14_b_driver.py"
            driver.write_text("#!/usr/bin/env python3\n")
            api.build(
                PARENT, output, 273,
                driver,
                POLICY,
                ROOT / "artifacts/round14/ROUND14_B_REQUESTS.json",
            )
            probe = (
                output
                / "overlay-python/sglang/srt/layers/round9/hybrid_probe.py"
            ).read_text()
            self.assertIn("if state['rows'] >= 8192:", probe)
            self.assertNotIn("if state['rows'] >= 4096:", probe)
            delta = json.loads((output / "ROUND14_B_SOURCE_DELTA.json").read_text())
            self.assertEqual(delta["hybrid_probe_max_rows"], 8192)


if __name__ == "__main__":
    unittest.main()
