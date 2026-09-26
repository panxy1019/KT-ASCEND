import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "build_round11_t4_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_round11_t4_snapshot", MODULE)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class BuildRound11T4SnapshotTests(unittest.TestCase):
    def test_launcher_patch_binds_role_budget_and_timing(self):
        root = Path(__file__).resolve().parents[2]
        source = (
            root
            / "round10"
            / "t6-snapshots"
            / "v107"
            / "source"
            / "tools"
            / "launch_hybrid.py"
        ).read_text()
        patched = builder.patch_launcher(source, mode="DYNAMIC", block="A", version=128)
        self.assertIn("kt-round11-t4-dynamic-v128", patched)
        self.assertIn("ROUND11_T4_PROCESS", patched)
        self.assertIn("ROUND11_DYNAMIC_TIMING_ENABLE", patched)
        self.assertIn("ROUND11_DYNAMIC_TIMING_SOURCE_PROOF.json", patched)
        self.assertIn("unified_diff_sha256", patched)
        self.assertIn("round11-qualified-dynamic.py", patched)
        self.assertNotIn("if a.version != 107", patched)

    def test_role_contract_rejects_version_drift(self):
        self.assertEqual(builder.role_name("STATIC", "A", 120), "kt-round11-t4-static-v120")
        self.assertEqual(builder.role_name("STATIC", "A", 124), "kt-round11-t4-static-v124")
        self.assertEqual(builder.role_name("DYNAMIC", "A", 128), "kt-round11-t4-dynamic-v128")
        self.assertEqual(builder.role_name("DYNAMIC", "B", 129), "kt-round11-t4-dynamic-v129")
        self.assertEqual(builder.role_name("DYNAMIC", "A", 132), "kt-round11-t4-dynamic-v132")
        self.assertEqual(builder.role_name("DYNAMIC", "B", 133), "kt-round11-t4-dynamic-v133")
        self.assertEqual(builder.role_name("DYNAMIC", "A", 134), "kt-round11-t4-dynamic-v134")
        self.assertEqual(builder.role_name("DYNAMIC", "B", 135), "kt-round11-t4-dynamic-v135")
        with self.assertRaisesRegex(ValueError, "identity"):
            builder.role_name("STATIC", "A", 121)

    def test_timing_source_proof_binds_exact_qualified_diff(self):
        root = Path(__file__).resolve().parents[2]
        qualified = root / "round10" / "t6-snapshots" / "v107" / "source" / "runtime" / "dynamic.py"
        instrumented = root.parent / "migration_bridge" / "round9" / "runtime" / "dynamic.py"
        proof = builder.build_timing_source_proof(qualified, instrumented)
        self.assertEqual(proof["status"], "PASS_ROUND11_TIMING_ONLY_SOURCE_DERIVATION")
        self.assertEqual(proof["qualified_dynamic_sha256"], builder.QUALIFIED_DYNAMIC_SHA256)
        self.assertEqual(proof["instrumented_dynamic_sha256"], builder.INSTRUMENTED_DYNAMIC_SHA256)
        self.assertEqual(proof["unified_diff_sha256"], builder.TIMING_DIFF_SHA256)
        self.assertEqual(proof["semantic_change"], "MONOTONIC_TIMING_OBSERVER_ONLY")

    def test_dynamic_profile_rebind_changes_only_profile_binding(self):
        parent = Path(__file__).resolve().parents[2] / "round10" / "t6-snapshots" / "v107"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rebound").mkdir()
            for name in ("candidate-profile.json", "bank-receipt.json"):
                shutil.copy2(parent / "rebound" / name, root / "rebound" / name)
            original = json.loads((root / "rebound/bank-receipt.json").read_text())
            proof = builder.rebind_dynamic_candidate_profile(root)
            profile = json.loads((root / "rebound/candidate-profile.json").read_text())
            rebound = json.loads((root / "rebound/bank-receipt.json").read_text())
            self.assertEqual(profile["placement"], "request_boundary")
            self.assertEqual(rebound["profile_sha256"], builder.DYNAMIC_PROFILE_CANONICAL_SHA256)
            self.assertEqual(rebound["banks"], original["banks"])
            self.assertEqual(rebound["catalog_sha256"], original["catalog_sha256"])
            self.assertEqual(rebound["source_identity"], original["source_identity"])
            self.assertEqual(proof["bank_payload_sha256"], builder.BANK_PAYLOAD_SHA256)
            self.assertEqual(proof["status"], "PASS_DYNAMIC_PROFILE_ONLY_REBIND")

    def test_supervisor_publishes_cleanup_bound_final_result(self):
        source = builder.supervisor_source(mode="DYNAMIC", block="B", version=122)
        self.assertIn("PASS_ROUND11_T4_DRIVER_RAW", source)
        self.assertIn("PASS_ROUND11_T4_PROCESS", source)
        self.assertIn("post-cleanup-proc-mem.txt", source)
        self.assertIn("guard_exit_ack", source)
        self.assertIn("round11-t4-final.json", source)

    def test_supervisor_uses_hfinal_edge64_fixture_and_reports_missing_raw(self):
        source = builder.supervisor_source(mode="STATIC", block="A", version=124)
        self.assertIn("/round10-t1-fixture.json", source)
        self.assertNotIn("/round11-t1-fixture.json", source)
        self.assertIn("T4 driver exited before publishing raw result", source)

    def test_hfinal_parent_fixture_has_exact_64_token_case(self):
        fixture = (
            Path(__file__).resolve().parents[2]
            / "round10"
            / "t6-snapshots"
            / "v107"
            / "round10-t1-fixture.json"
        )
        document = json.loads(fixture.read_text())
        ids = document["cases"]["edge64"]["input_ids"]
        self.assertEqual(len(ids), 64)
        self.assertTrue(all(type(value) is int for value in ids))


if __name__ == "__main__":
    unittest.main()
