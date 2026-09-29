import hashlib
import unittest


class Round17ProfileSnapshotTests(unittest.TestCase):
    def test_patch_launcher_binds_new_identity_and_normalizes_observer(self):
        from tools.round17.build_profile_snapshot import patch_launcher

        old = "a" * 64
        new = "b" * 64
        source = """if a.version != 509 or a.cost_mode != 'OFF':
 raise RuntimeError('fixed cost trial identities required')
name='kt-root-cause-full-host-a-v509';out=root/'evidence'/'20260929-kt-root-cause-full-host-a-v509'
KT_ROOT_TRACE_BINDING_PATH=/round9/kt-root-cause-full-host-a-v509/TRACE_NATIVE_BINDING.json
_observer_relative='sglang/srt/layers/round9/round13_attribution.py'
if candidate_files.get(_observer_relative)!='ec9e4c88a00251ae635dca9cee3eca4dbec168ddb6df2cfb4f01babe833ea49f':
 raise RuntimeError('full-host observer differs from bound source')
candidate_files[_observer_relative]='a4bd6c90bc05ec09e51d14194fb2eaf38e00dcb5578d48b91bfdfc1f4b392cc0'
ROUND16_A1_BLOCK_ID=round16-a1-fresh-profile-a-v001
"""
        patched = patch_launcher(
            source,
            version=510,
            block="B",
            snapshot_name="round17-profile-a-v510",
            evidence_name="20260929-round17-profile-a-v510",
            diagnostic_observer_sha256=new,
            baseline_observer_sha256=old,
        )
        self.assertIn("a.version != 510", patched)
        self.assertIn("name='round17-profile-a-v510'", patched)
        self.assertIn(new, patched)
        self.assertIn(old, patched)
        self.assertIn("ROUND16_A1_BLOCK_ID=round16-a1-fresh-profile-b-v001", patched)

    def test_patch_block_changes_role_without_changing_schedule(self):
        from tools.round17.build_profile_snapshot import patch_block

        value = {
            "schema": 1,
            "block": "A",
            "block_id": "round16-a1-fresh-profile-a-v001",
            "snapshot_id": "round16-a1-fresh-profile-a-v001-snapshot",
            "schedule_sha256": "c" * 64,
        }
        patched = patch_block(value, block="B", version=511)
        self.assertEqual(patched["block"], "B")
        self.assertEqual(patched["block_id"], "round16-a1-fresh-profile-b-v001")
        self.assertEqual(patched["schedule_sha256"], "c" * 64)
        self.assertEqual(patched["round17_version"], 511)

    def test_patch_monitor_adds_only_the_new_owned_name(self):
        from tools.round17.build_profile_snapshot import patch_monitor

        source = (
            "r'|kt-round16-a1-[ab]-v003|kt-root-cause-profile-[ab]-v506|"
            "kt-root-cause-full-host-a-v509)', name) is not None"
        )
        patched = patch_monitor(source, snapshot_name="round17-profile-a-v510")
        self.assertIn("|round17-profile-a-v510)", patched)


if __name__ == "__main__":
    unittest.main()
