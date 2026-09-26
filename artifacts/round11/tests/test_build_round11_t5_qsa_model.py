import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1] / "build_round11_t5_qsa_model.py"
SPEC = importlib.util.spec_from_file_location("build_round11_t5_qsa_model", MODULE)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class Round11T5QSAModelBuilderTests(unittest.TestCase):
    def test_training_identity_is_the_frozen_full_container_id(self):
        self.assertEqual(
            builder.TRAINING_ID,
            "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef",
        )

    def test_qwen_patch_changes_only_sparse_gqa_dispatch(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "round11/t4-snapshots/v127/overlay-python/sglang/srt/models/qwen4_exp.py"
        source = path.read_text()
        patched = builder.patch_qwen(source)
        self.assertIn("sparse_gqa_candidate", patched)
        self.assertEqual(patched.count("sparse_gqa_candidate"), 3)
        self.assertNotEqual(source, patched)

    def test_launcher_patch_binds_single_v145_identity(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "round11/t4-snapshots/v127/source/tools/launch_hybrid.py"
        patched = builder.patch_launcher(path.read_text())
        self.assertIn("kt-round11-t5-qsa-candidate-v145", patched)
        self.assertIn("20260926-round11-t5-qsa-model-v145", patched)
        self.assertIn("ROUND11_T5_QSA_MODEL", patched)
        self.assertNotIn("if a.version != 127", patched)

    def test_v146_rebind_keeps_base_gate_and_requires_v144_candidate_proof(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "round11/t4-snapshots/v127/source/tools/launch_hybrid.py"
        patched = builder.patch_launcher(path.read_text(), version=146)
        patched = builder.patch_context_gate(patched)
        self.assertIn("kt-round11-t5-qsa-candidate-v146", patched)
        self.assertIn("round11-qualified-qsa.py", patched)
        self.assertIn("PASS_ROUND11_QSA_GATE_REBIND", patched)
        self.assertIn("QSA_COMPONENT_PROMOTE_TO_REQUEST", patched)
        self.assertIn("PASS_ROUND11_T5_QSA_COMPONENT", patched)
        compile(patched, "launch_hybrid.py", "exec")

    def test_supervisor_requires_four_generations_and_cleanup(self):
        source = builder.supervisor_source()
        self.assertIn("generation_attempts')!=4", source)
        self.assertIn("PASS_ROUND11_T5_QSA_MODEL_PROCESS", source)
        self.assertIn("post-cleanup-proc-mem.txt", source)
        self.assertIn("guard_exit_ack", source)
        compile(source, "round11_t5_qsa_model_supervisor.py", "exec")

    def test_repaired_supervisor_binds_v146(self):
        source = builder.supervisor_source(version=146)
        self.assertIn("qsa-candidate-v146", source)
        self.assertIn("'--version','146'", source)
        compile(source, "round11_t5_qsa_model_supervisor.py", "exec")


if __name__ == "__main__":
    unittest.main()
