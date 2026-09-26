import sys
import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import build_round10_t4_snapshot as builder


class T4SnapshotBuilderTests(unittest.TestCase):
    def test_launcher_patch_freezes_dynamic_identity_and_budget(self):
        source="\n".join((
            "if a.version != 95 or a.cost_mode != 'OFF':",
            "name='kt-round10-t1-r-v95';out=root/'evidence'/'20260926-round10-t1-v95'",
            " if budget.get('stage')!='ROUND10_T1_PROCESS_R':raise RuntimeError('wrong Round10 T1 budget')",
        ))
        actual=builder.patch_launcher(source,102)
        self.assertIn("a.version != 102",actual)
        self.assertIn("kt-round10-t4-d-v102",actual)
        self.assertIn("ROUND10_T4_D1",actual)
        with self.assertRaisesRegex(ValueError,'precondition'):
            builder.patch_launcher(source.replace('version != 95','version != 94'),102)

    def test_guard_accepts_only_t4_dynamic_identity(self):
        source="""import re
def owned_container_name(name):
    return re.fullmatch(
        r'(?:kt-round9-(?:full-model|c128-expert|layerwise|ple|context-tier)-v[0-9]+'
        r'|kt-round10-t1-[abr]-v[0-9]+)', name) is not None
"""
        namespace={};exec(builder.patch_guard(source),namespace)
        self.assertTrue(namespace['owned_container_name']('kt-round10-t4-d-v102'))
        self.assertFalse(namespace['owned_container_name']('kt-round10-t4-x-v102'))

    def test_generated_control_plane_compiles(self):
        compile(builder.controller_source(102,'0'*64),'controller.py','exec')
        compile(builder.supervisor_source(102),'supervisor.py','exec')


if __name__=='__main__':unittest.main()
