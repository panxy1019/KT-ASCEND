import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import round10_admission as admission


class AdmissionTests(unittest.TestCase):
    def test_numa_meminfo_parses_node_prefixed_kernel_format(self):
        raw = "Node 2 MemTotal:       264107980 kB\nNode 2 MemFree:         50331649 kB\n"
        self.assertEqual(
            admission.parse_numa_meminfo(raw),
            {"MemTotal": 264107980, "MemFree": 50331649},
        )

    def test_smt_min_mean_uses_every_sibling(self):
        cores = [
            {"node": 2, "siblings": [10, 11], "excluded_explicit_container_affinity": False},
            {"node": 3, "siblings": [20, 21], "excluded_explicit_container_affinity": False},
        ]
        idle = {"10": 0.99, "11": 0.95, "20": 0.9, "21": 0.8}
        self.assertAlmostEqual(admission.conservative_mean(cores, idle), (0.95 + 0.8) / 2)

    def test_validate_rejects_busy_window_or_unknown_device_pid(self):
        document = admission.synthetic_document_for_test()
        document["windows"][0]["idle"] = {
            cpu: 0.1 for cpu in document["windows"][0]["idle"]
        }
        with self.assertRaisesRegex(RuntimeError, "idle"):
            admission.validate_documents([document, admission.synthetic_document_for_test()])
        first = admission.synthetic_document_for_test()
        first["windows"][0]["physical12_processes"] = {"999": 100}
        with self.assertRaisesRegex(RuntimeError, "unknown physical12"):
            admission.validate_documents([first, admission.synthetic_document_for_test()])

    def test_two_60_second_documents_pass_exact_contract(self):
        first = admission.synthetic_document_for_test()
        second = admission.synthetic_document_for_test()
        got = admission.validate_documents([first, second])
        self.assertEqual(got["status"], "PASS")
        self.assertEqual(got["windows"], 12)
        self.assertEqual(got["selected_physical_cores"], 160)
        self.assertGreater(got["min_mean_idle"], 0.8)


if __name__ == "__main__":
    unittest.main()
