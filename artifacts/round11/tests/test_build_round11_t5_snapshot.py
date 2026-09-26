import unittest
import re

from artifacts.round11.build_round11_t5_snapshot import (
    controller_source,
    patch_launcher,
    patch_scheduler,
)
from artifacts.round11.tests.test_round11_monitor_owned import SCOPE as GUARD_SCOPE


class BuildRound11T5SnapshotTests(unittest.TestCase):
    def test_scheduler_hook_brackets_run_batch_once(self):
        source = (
            "                from sglang.srt.layers.round9.framework_hooks import framework_before\n"
            "                framework_before(self, batch)\n"
            "                result = self.run_batch(batch)\n"
            "                self.process_batch_result(batch, result)\n"
        )
        actual = patch_scheduler(source)
        self.assertEqual(actual.count("round11_b_before(self, batch)"), 1)
        self.assertEqual(actual.count("round11_b_after(self)"), 1)
        self.assertLess(actual.index("round11_b_before"), actual.index("self.run_batch"))
        self.assertLess(actual.index("self.run_batch"), actual.index("round11_b_after(self)"))
        with self.assertRaisesRegex(ValueError, "precondition"):
            patch_scheduler(actual)

    def test_launcher_enables_only_bounded_observer_and_rebinds_identity(self):
        source = "\n".join(
            (
                "if a.version != 127 or a.cost_mode != 'OFF':",
                " pass",
                "name='kt-round11-t4-static-v127';out=root/'evidence'/'20260926-round11-t4-v127'",
                "for env in ('ROUND9_DYNAMIC_EVENTS=/out/dynamic-events.jsonl',f'ROUND9_DYNAMIC_BOUNDARY_ENABLE={int(a.placement==\"request_boundary\")}',f'ROUND11_DYNAMIC_TIMING_ENABLE={int(a.placement==\"request_boundary\")}'):args+=['-e',env]",
                "if budget.get('stage')!='ROUND11_T4_PROCESS':raise RuntimeError('wrong Round11 T4 budget')",
            )
        )
        actual = patch_launcher(source, block="A", version=136)
        self.assertIn("kt-round11-t5-hotspot-v136", actual)
        self.assertNotIn("hotspot-a-v136", actual)
        self.assertIn("ROUND11_B_ATTRIBUTION_ENABLE=1", actual)
        self.assertIn("ROUND11_B_SPANS=/out/round11-b-spans.jsonl", actual)
        self.assertIn("ROUND11_T5_PROCESS", actual)
        self.assertNotIn("request_boundary')!=0", actual)

    def test_generated_container_name_is_accepted_by_existing_guard(self):
        source = controller_source(block="A", version=139)
        name = re.search(r"name='([^']+)'", source).group(1)
        self.assertEqual(name, "kt-round11-t5-hotspot-v139")
        self.assertTrue(GUARD_SCOPE["owned_container_name"](name))


if __name__ == "__main__":
    unittest.main()
