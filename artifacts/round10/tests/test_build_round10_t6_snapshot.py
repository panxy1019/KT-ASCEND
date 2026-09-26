import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import build_round10_t6_snapshot as b

class T6BuilderTests(unittest.TestCase):
    def test_identity_patches_are_bounded(self):
        launcher="if a.version != 95 or a.cost_mode != 'OFF':\nname='kt-round10-t1-r-v95';out=root/'evidence'/'20260926-round10-t1-v95'\nif budget.get('stage')!='ROUND10_T1_PROCESS_R':raise RuntimeError('wrong Round10 T1 budget')\n"
        actual=b.patch_launcher(launcher);self.assertIn('107',actual);self.assertIn('ROUND10_T6_FINAL',actual)
        guard="r'|kt-round10-t1-[abr]-v[0-9]+)', name) is not None"
        self.assertIn('kt-round10-t6-final',b.patch_guard(guard))

    def test_supervisor_accepts_only_final_identity_and_budget(self):
        source=b.patch_supervisor(b.t1.supervisor_source());compile(source,'supervisor.py','exec')
        self.assertIn("('F',107)",source);self.assertIn("'F':8",source);self.assertIn("'F':1",source)
        self.assertIn('PASS_ROUND10_T6_PROCESS',source)

    def test_controller_reads_the_final_supervisor_receipt(self):
        source=b.patch_controller("report['status']=json.loads((out/'round10-t1-supervisor.json').read_text())['status']")
        self.assertIn("round10-t6-supervisor.json",source)
        self.assertNotIn("round10-t1-supervisor.json",source)

if __name__=='__main__':unittest.main()
