from pathlib import Path
import unittest

from tools.kt_perf_closure.build_full_host_snapshot import (
    patch_launcher,
    patch_observer,
)


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "artifacts/kt-perf-root-cause/v506-source-samples"


class FullHostSnapshotTests(unittest.TestCase):
    def test_observer_records_all_attention_parents_but_only_samples_device_events(self):
        source = (SAMPLES / "round13_attribution.py").read_text()
        patched = patch_observer(source)
        self.assertIn("GDN_LAYERS = tuple(layer for layer in range(48)", patched)
        self.assertIn("QSA_LAYERS = tuple(range(3, 48, 4))", patched)
        self.assertIn("GDN_DEVICE_LAYERS = (0, 24, 46)", patched)
        self.assertIn("device=layer in GDN_DEVICE_LAYERS", patched)
        self.assertIn("device=layer in QSA_DEVICE_LAYERS", patched)

    def test_launcher_normalizes_only_the_bound_observer_diff(self):
        source = (SAMPLES / "launch_hybrid.py").read_text()
        patched = patch_launcher(
            source,
            run_version=508,
            baseline_observer_sha="1" * 64,
            diagnostic_observer_sha="2" * 64,
        )
        self.assertIn("a.version != 508", patched)
        self.assertIn("kt-root-cause-full-host-a-v508", patched)
        self.assertIn("candidate_files.get(_observer_relative)!='" + "2" * 64, patched)
        self.assertIn("candidate_files[_observer_relative]='" + "1" * 64, patched)


if __name__ == "__main__":
    unittest.main()
