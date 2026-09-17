"""R16 first, frozen 32 requests/restart/observer. Never starts CMMLU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from common import sha, write
from serial_session import session
sys.path.append('/workspace/r67-tools-v5/migration_bridge/round6_7/tools')
from run_serial_sweep import request, route_summary

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--operator-summary',type=Path,required=True)
p.add_argument('--profile',choices=('R16','R8'),default='R16')
p.add_argument('--performance-summary',type=Path);a=p.parse_args()
if a.profile=='R8':
    assert a.performance_summary is not None,'R8_PERFORMANCE_PREDECESSOR'
    perf=json.loads(a.performance_summary.read_text())
    assert perf['status']=='PASS' and perf['verdict']=='BENEFICIAL','R8_NOT_YET_ALLOWED'
op=json.loads(a.operator_summary.read_text())
assert op['status']=='PASS' and len(op['runs'])==52,'OPERATOR_PREDECESSOR'
audit=json.loads(a.operator_summary.with_name('audit.json').read_text())
assert audit['status']=='PASS' and audit['summary_sha256']==sha(a.operator_summary),'OPERATOR_AUDIT'
assert sha('/workspace/c64-dev/runtime/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so')==op['native_sha256']
assert sha('/workspace/c64-dev/runtime/kt_kernel/utils/llamafile.py')==op['wrapper_sha256']
a.output.mkdir(parents=True,exist_ok=False)
base=Path('/workspace/results/r65-r67-serial-v5')
for name in (a.profile+'.json',a.profile+'.pt'):shutil.copy2(base/'profiles'/name,a.output/name)
shutil.copy2(base/'compact.json',a.output/'compact.json')
profile=a.output/(a.profile+'.json');manifest=json.loads((a.output/'compact.json').read_text())
assert len(manifest['prompts'])==8 and len(manifest['budgets'])==4
os.environ.update(C64_TP='2',KT_R67_CPU_SET='128-191',KT_R67_NPU_ID='0')
os.environ.pop('KT_C64_CAPTURE_DIR',None)
report=dict(status='RUNNING',phase='COMPACT',profile=a.profile,started=time.time(),references={},requests=0,
    profile_sha256=sha(profile),compact_sha256=sha(a.output/'compact.json'),
    operator_summary_sha256=sha(a.operator_summary),workers=64,numa=[4,5],performance='CONTENDED_NOT_FORMAL')
write(a.output/'progress.json',report)
try:
    with session(a.output/'compact',profile) as root:
        for row in manifest['prompts']:
            anchor=None
            for i,budget in enumerate(manifest['budgets']):
                _,ids,_=request(root,row,f'c64-{row["id"]}-{i}',budget)
                if anchor is None:anchor=ids
                assert ids==anchor[:budget],'SAME_PATH_OR_PREFIX'
                report['requests']+=1;write(a.output/'progress.json',report)
            report['references'][row['id']]=anchor
            write(a.output/'progress.json',report)
    report['phase']='RESTART';write(a.output/'progress.json',report)
    with session(a.output/'restart',profile) as root:
        for row in manifest['prompts']:
            _,ids,_=request(root,row,f'c64-restart-{row["id"]}',16)
            assert ids==report['references'][row['id']],'RESTART_MISMATCH'
    report['phase']='OBSERVER';write(a.output/'progress.json',report)
    with session(a.output/'observer',profile,observer=True) as root:
        for row in manifest['prompts']:
            _,ids,_=request(root,row,f'c64-observer-{row["id"]}',8)
            assert ids==report['references'][row['id']][:8],'OBSERVER_PARITY'
        report['routing']=route_summary(root/'routing.jsonl',profile)
        structure=[json.loads(x) for x in (root/'physical-structure.jsonl').read_text().splitlines()]
        assert {r['layer'] for r in structure}==set(range(1,27)),'PHYSICAL_LAYOUT'
        report['structure']=structure
    report.update(status='PASS',phase='COMPLETE')
except BaseException as exc:
    report.update(status='FAIL',error=repr(exc));raise
finally:
    report['ended']=time.time();write(a.output/'progress.json',report);write(a.output/'summary.json',report)
