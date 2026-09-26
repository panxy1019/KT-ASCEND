import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_round10_t1_snapshot as builder


class SnapshotBuilderTests(unittest.TestCase):
    def test_launcher_patch_is_exact_and_single_process_identity(self):
        source = "\n".join(
            [
                "if a.version != 90 or a.cost_mode != 'OFF':",
                " raise RuntimeError('fixed cost trial identities required')",
                "name=f'kt-round9-full-model-v{a.version}';out=root/'evidence'/f'20260925-plan15-threshold-v{a.version}'",
                " if budget.get('stage')!='PLAN15_HYBRID_THRESHOLD_CANDIDATE129':raise RuntimeError('wrong independent threshold budget')",
                "  starts=budget.setdefault('starts',[])",
                "  if len(starts)>=1 or any(x['mode']==a.cost_mode for x in starts):raise RuntimeError('start budget spent; no retries')",
                "  if a.cost_mode=='ON' and (len(starts)!=1 or starts[0]['mode']!='OFF' or not budget.get('off_verified')):",
                "   raise RuntimeError('ON requires reviewed OFF result and cleanup')",
            ]
        ) + "\n"
        patched = builder.patch_launcher(source, process="A", version=91)
        self.assertIn("a.version != 91", patched)
        self.assertIn("kt-round10-t1-a-v91", patched)
        self.assertIn("20260926-round10-t1-v91", patched)
        self.assertIn("ROUND10_T1_PROCESS_A", patched)
        self.assertNotIn("a.cost_mode=='ON'", patched)
        with self.assertRaisesRegex(ValueError, "precondition"):
            builder.patch_launcher(source.replace("version != 90", "version != 89"), process="A", version=91)

    def test_contract_rejects_wrong_version_process_pair(self):
        self.assertEqual(builder.identity("A", 91)["container_name"], "kt-round10-t1-a-v91")
        self.assertEqual(builder.identity("B", 92)["container_name"], "kt-round10-t1-b-v92")
        self.assertEqual(builder.identity("A", 93)["container_name"], "kt-round10-t1-a-v93")
        self.assertEqual(builder.identity("B", 94)["container_name"], "kt-round10-t1-b-v94")
        self.assertEqual(builder.identity("R", 95)["container_name"], "kt-round10-t1-r-v95")
        with self.assertRaises(ValueError):
            builder.identity("A", 92)

    def test_guard_patch_allows_only_bounded_round10_t1_names(self):
        source = """import re
def owned_container_name(name):
    return re.fullmatch(
        r'kt-round9-(?:full-model|c128-expert|layerwise|ple|context-tier)-v[0-9]+',
        name) is not None
message = 'guard may only control an explicitly allowed owned Round9 container'
"""
        patched = builder.patch_guard(source)
        namespace = {}
        exec(patched, namespace)
        self.assertTrue(namespace["owned_container_name"]("kt-round10-t1-a-v93"))
        self.assertTrue(namespace["owned_container_name"]("kt-round10-t1-b-v92"))
        self.assertTrue(namespace["owned_container_name"]("kt-round10-t1-r-v95"))
        self.assertFalse(namespace["owned_container_name"]("kt-round10-t1-a-v93-other"))
        self.assertFalse(namespace["owned_container_name"]("mzy-train"))

    def test_generated_controller_and_supervisor_are_valid_python(self):
        values = builder.identity("A", 91)
        compile(builder.controller_source(values, "0" * 64), "host_controller.py", "exec")
        compile(builder.supervisor_source(), "round10_t1_supervisor.py", "exec")


if __name__ == "__main__":
    unittest.main()
