import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1] / "build_round11_t6_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_round11_t6_snapshot", MODULE)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class Round11T6SnapshotBuilderTests(unittest.TestCase):
    def test_roles_are_fixed_abba(self):
        self.assertEqual(builder.role(147), ("BASELINE", "A"))
        self.assertEqual(builder.role(148), ("FINAL", "A"))
        self.assertEqual(builder.role(149), ("FINAL", "B"))
        self.assertEqual(builder.role(150), ("BASELINE", "B"))
        with self.assertRaisesRegex(ValueError, "identity"):
            builder.role(151)

    def test_launcher_patch_has_t6_identity_and_budget(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "round11/t4-snapshots/v127/source/tools/launch_hybrid.py"
        patched = builder.patch_launcher(path.read_text(), version=148)
        self.assertIn("kt-round11-t6-final-v148", patched)
        self.assertIn("20260926-round11-t6-v148", patched)
        self.assertIn("ROUND11_T6_PROCESS", patched)
        self.assertNotIn("if a.version != 127", patched)
        compile(patched, "launch_hybrid.py", "exec")

    def test_supervisor_binds_six_generations_and_cleanup(self):
        source = builder.supervisor_source(version=147)
        self.assertIn("generation_attempts')!=6", source)
        self.assertIn("PASS_ROUND11_T6_PROCESS", source)
        self.assertIn("post-cleanup-proc-mem.txt", source)
        compile(source, "round11_t6_supervisor.py", "exec")


if __name__ == "__main__":
    unittest.main()
