import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).parents[3]
TOOL=ROOT/'migration_bridge/round9/tools/candidate_workflow.py'

class WorkflowTests(unittest.TestCase):
    def test_profile_and_plan_do_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=root/'profile.json'
            r=subprocess.run([sys.executable,str(TOOL),'profile','--workers','128','--storage','BF16','--placement','static','--numa-nodes','2,3,6,7','--output',str(profile)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            topology=root/'cpus.json';topology.write_text(json.dumps(dict(quota=136,rows=[dict(cpu=n*100+i,core=f'{n}:{i}',numa=n) for n in (2,3,6,7) for i in range(40)])))
            r=subprocess.run([sys.executable,str(TOOL),'plan','--profile',str(profile),'--topology',str(topology),'--output',str(root/'plan.json')],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            report=json.loads((root/'plan.json').read_text());self.assertEqual(report['status'],'HOST_PLAN_ONLY_DEVICE_ADMISSION_PENDING')
            self.assertEqual(report['server_args']['kt_threadpool_count'],4)
            self.assertFalse(report['launch_permitted'])

if __name__=='__main__':unittest.main()
