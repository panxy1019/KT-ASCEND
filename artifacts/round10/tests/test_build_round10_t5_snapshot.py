import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import build_round10_t5_snapshot as builder

class T5BuilderTests(unittest.TestCase):
    def test_device_identity_patches_are_exact(self):
        g='physical.value != 15\npathlib.Path("/dev/davinci15")\n'
        self.assertIn('physical.value != 12',builder.patch_gdn(g))
        q="phy.value==15\npathlib.Path('/dev/davinci15')\n"
        self.assertIn('phy.value==12',builder.patch_qsa(q))
        p=("admission.get('physical')!=15\nphysical.value!=15\n{Path('/dev/davinci15')}\n"
           "physical_device=15,\n"
           "source_sha256=source_sha,torch_version=str(torch.__version__),torch_npu_version=str(torch_npu.__version__))\n")
        actual=builder.patch_ple(p)
        self.assertIn("physical_device=12",actual)
        self.assertIn("expected_physical_device=12",actual)
        self.assertNotIn('davinci15',actual)

    def test_ple_gate_patch_keeps_history_default_and_allows_explicit_device(self):
        source=("def require_ple_device_gate(path, expected_sha, *, source_identity, source_sha256,\n"
                "                            torch_version, torch_npu_version):\n"
                "    exact=dict(physical_device=15, value=True)\n")
        actual=builder.patch_ple_gate(source)
        self.assertIn("expected_physical_device=15",actual)
        self.assertIn("physical_device=expected_physical_device",actual)

    def test_guard_and_controller_are_bounded(self):
        source="""import re
def owned_container_name(name):
    return re.fullmatch(
        r'(?:kt-round9-(?:full-model|c128-expert|layerwise|ple|context-tier)-v[0-9]+'
        r'|kt-round10-t1-[abr]-v[0-9]+'
        r'|kt-round10-t4-d-v[0-9]+)', name) is not None
"""
        scope={};exec(builder.patch_guard(source),scope)
        self.assertTrue(scope['owned_container_name']('kt-round10-t5-components-v103'))
        self.assertFalse(scope['owned_container_name']('kt-round10-t5-components-v103-x'))
        compile(builder.controller_source(103,'0'*64),'controller.py','exec')

    def test_recovery_controller_reuses_only_frozen_successes(self):
        prior={'root':'/evidence/v103','gdn_sha256':'a'*64,'qsa_sha256':'b'*64}
        source=builder.controller_source(104,'0'*64,prior)
        compile(source,'controller.py','exec')
        self.assertIn('PASS_PRIOR_T5_RESULTS_VERIFIED',source)
        self.assertNotIn('gdn_device_probe.py --model',source)
        self.assertNotIn('qsa_device_probe.py --model',source)
        self.assertIn('ple_async_device_probe.py --model',source)
        self.assertNotIn('/models/{rev}',source)
        self.assertNotIn('{receipt}',source)
        self.assertIn('/models/de4b8e4d43b917e7706784d8bb445c9af86a3540',source)
        self.assertLess(source.index("if state['ExitCode']!=0"),source.index("out/'result.json'"))

if __name__=='__main__':unittest.main()
