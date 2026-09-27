import hashlib
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from round12a_offloader_patch import (  # noqa: E402
    BASE_OFFLOADER_SHA256,
    patch_offloader_source,
    unique_items_by_identity,
)


class Round12aOffloaderPatchTests(unittest.TestCase):
    def test_unique_items_keeps_first_alias_and_all_unique_values(self):
        tied = object()
        other = object()
        state = {
            "linear_attn.A_log": tied,
            "linear_attn.attn.A_log": tied,
            "linear_attn.dt_bias": other,
        }
        self.assertEqual(
            unique_items_by_identity(state),
            [
                ("linear_attn.A_log", tied),
                ("linear_attn.dt_bias", other),
            ],
        )

    def test_patch_is_exact_and_preserves_default_tied_semantics(self):
        source_path = ROOT / "fixtures/offloader_v1_base.py"
        source = source_path.read_text()
        self.assertEqual(
            hashlib.sha256(source_path.read_bytes()).hexdigest(),
            BASE_OFFLOADER_SHA256,
        )

        patched = patch_offloader_source(source)
        compile(patched, "offloader.py", "exec")
        self.assertIn("module.state_dict(keep_vars=True)", patched)
        self.assertIn("_device_state_preserving_ties(module, device)", patched)
        self.assertNotIn("tie_weights=False", patched)
        self.assertEqual(
            patched.count("def _device_state_preserving_ties("), 1
        )
        self.assertNotEqual(
            hashlib.sha256(patched.encode()).hexdigest(),
            BASE_OFFLOADER_SHA256,
        )

    def test_patch_rejects_unknown_or_already_patched_source(self):
        source = (ROOT / "fixtures/offloader_v1_base.py").read_text()
        patched = patch_offloader_source(source)
        with self.assertRaises(ValueError):
            patch_offloader_source(patched)
        with self.assertRaises(ValueError):
            patch_offloader_source(source.replace("class OffloaderV1", "class Drift"))

    def test_component_uses_exact_candidate_and_tied_contract(self):
        candidate = ROOT / "fixtures/offloader_v1_tied_candidate.py"
        self.assertEqual(
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006",
        )
        probe = (ROOT / "round12a_tied_offloader_probe.py").read_text()
        controller = (ROOT / "round12a_tied_offloader_controller.py").read_text()
        self.assertIn("PASS_ROUND12A_TIED_OFFLOADER_COMPONENT", probe)
        self.assertIn("tied_identity_preserved=True", probe)
        self.assertIn("tie_weights_false_used=False", probe)
        self.assertIn(
            "candidate / 'offloader.py'", controller
        )
        self.assertIn(
            "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006",
            controller,
        )

    def test_component_snapshot_binds_its_own_bounded_guard(self):
        builder = (ROOT / "build_round12a_component.py").read_text()
        controller = (ROOT / "round12a_tied_offloader_controller.py").read_text()
        self.assertIn("VERSION = 172", builder)
        self.assertIn("VERSION = 172", controller)
        self.assertIn('"round12a_monitor_owned.py"', builder)
        self.assertIn('candidate / "round12a_monitor_owned.py"', controller)
        self.assertNotIn('parser.add_argument("--monitor"', controller)
        self.assertNotIn("args.monitor", controller)

    def test_npu_fixture_constructs_tied_objects_on_the_target_device(self):
        probe = (ROOT / "round12a_tied_offloader_probe.py").read_text()
        self.assertIn('TiedProbeModule(width, device="npu:0")', probe)
        self.assertNotIn('TiedProbeModule(width).to("npu:0")', probe)

    def test_component_cleanup_releases_keep_vars_state_reference(self):
        probe = (ROOT / "round12a_tied_offloader_probe.py").read_text()
        self.assertIn("state = None", probe)
        self.assertIn("del state, reference, value, wrapped, module, offloader", probe)


if __name__ == "__main__":
    unittest.main()
