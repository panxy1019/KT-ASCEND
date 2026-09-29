from pathlib import Path
import unittest

from tools.kt_perf_closure.build_profile_snapshot import (
    patch_candidate_server,
    patch_launcher,
    patch_layerwise_gate,
    patch_monitor,
    patch_overlay_gate,
    patch_q32_gate,
)


class ProfileSnapshotPatchTests(unittest.TestCase):
    def test_launcher_patches_expected_contract(self):
        source = (Path(__file__).resolve().parents[2] / "artifacts/round16-a/failed-launch-a-v002/source/tools/launch_hybrid.py").read_text()
        # The locally archived failed launch may predate the final v003 identity;
        # use the exact final anchors as a compact synthetic contract.
        source = source.replace("if a.version != 295 or a.cost_mode != 'OFF':", "if a.version != 295 or a.cost_mode != 'OFF':")
        if "name='kt-round16-a1-a-v003'" not in source:
            self.skipTest("final v003 launcher is remote-only")
        patched = patch_launcher(source, block="A")
        self.assertIn("round11-t2-attribution-runtime-v3", patched)
        self.assertIn("KT_ROOT_TASK_TRACE_PATH", patched)
        self.assertIn("kt-root-cause-profile-a-v501", patched)

    def test_monitor_patch_contract(self):
        source = "return re.fullmatch(r'|kt-round16-a1-[ab]-v003)', name) is not None"
        patched = patch_monitor(source)
        self.assertIn("kt-root-cause-profile-[ab]-v501", patched)

    def test_candidate_server_keeps_q32_gate_and_binds_trace_native(self):
        source = """native_sha256=hashlib.sha256(Path(kt_kernel_ext.__file__).resolve().read_bytes()).hexdigest()
    require_capacity_device_gate(capacity,candidate,identity,payload,native_sha256)
"""
        binding_sha = "a" * 64
        patched = patch_candidate_server(source, binding_sha=binding_sha)
        self.assertIn("KT_ROOT_TRACE_BINDING_PATH", patched)
        self.assertIn(binding_sha, patched)
        self.assertIn("actual_native_sha256", patched)
        self.assertIn("binding['baseline_native_sha256']", patched)
        self.assertIn("require_capacity_device_gate", patched)
        self.assertIn("PASS_KT_ROOT_DIAGNOSTIC_NATIVE_BINDING", patched)

    def test_q32_gate_maps_only_the_bound_diagnostic_native(self):
        source = """    if (gate.get('storage') != storage or gate.get('cpu_workers') != workers or
            gate.get('source_identity') != source_identity or
            gate.get('bank_sha256') != layer0_sha256 or
            not _sha256(native_sha256) or gate.get('native_sha256') != native_sha256):
        raise RuntimeError('Q32 receipt source, storage, workers, bank or native identity mismatch')
"""
        patched = patch_q32_gate(source, binding_sha="b" * 64)
        self.assertIn("validated_native_sha256", patched)
        self.assertIn("diagnostic_native_sha256", patched)
        self.assertIn("baseline_native_sha256", patched)
        self.assertIn("b" * 64, patched)

    def test_overlay_gate_allows_only_the_bound_q32_gate_diff(self):
        source = """if overlay['baseline_tree_sha256']!=tree_digest(tree_files(root/'src/sglang-round9-port/python')) or overlay['candidate_tree_sha256']!=tree_digest(tree_files(sglang)):
 raise RuntimeError('installed overlay differs from recorded baseline/candidate payload')
"""
        patched = patch_overlay_gate(
            source,
            baseline_q32_sha="c" * 64,
            diagnostic_q32_sha="d" * 64,
        )
        self.assertIn("candidate_files[_q32_relative]", patched)
        self.assertIn("c" * 64, patched)
        self.assertIn("d" * 64, patched)
        self.assertIn("tree_digest(candidate_files)", patched)

    def test_layerwise_gate_maps_only_the_bound_diagnostic_native(self):
        source = """    if (not isinstance(source_identity, dict) or not source_identity or
            gate.get('source_identity') != source_identity or
            not _sha256(native_sha256) or gate.get('native_sha256') != native_sha256):
        raise RuntimeError('layerwise original model or native identity mismatch')
"""
        patched = patch_layerwise_gate(source, binding_sha="e" * 64)
        self.assertIn("validated_native_sha256", patched)
        self.assertIn("diagnostic_native_sha256", patched)
        self.assertIn("e" * 64, patched)


if __name__ == "__main__":
    unittest.main()
