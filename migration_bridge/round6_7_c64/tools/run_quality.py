"""Long quality is opt-in only after C64 system and beneficial isolated perf."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from common import sha,write,product_gate
from serial_session import session  # establish the private lifecycle module
sys.path.append('/workspace/r67-tools-v5/migration_bridge/round6_7/tools')
from run_serial_sweep import quality,paired

p=argparse.ArgumentParser();p.add_argument('--canary',type=Path,required=True)
p.add_argument('--performance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--r8-smoke',type=Path,required=True)
p.add_argument('--user-authorized-quality-screen',action='store_true')
a=p.parse_args()
c=json.loads(a.canary.read_text());perf=json.loads(a.performance.read_text())
assert c['status']=='PASS' and perf['status']=='PASS','QUALITY_ENTRY_GATE'
if a.user_authorized_quality_screen:
    assert perf.get('formal_performance') is False,'EXPLICIT_EXPLORATORY_REPORT_REQUIRED'
else:
    assert perf.get('formal_performance',True) and perf['verdict']=='BENEFICIAL','BENEFICIAL_ISOLATED_PERFORMANCE_REQUIRED'
assert json.loads(a.r8_smoke.read_text())['status']=='PASS','R8_SMOKE_PREDECESSOR'
mem=json.loads(a.performance.with_name('memory-audit.json').read_text())
assert mem['status']=='PASS' and mem['performance_sha256']==sha(a.performance),'MEMORY_PREDECESSOR'
assert sha('/workspace/c64-dev/runtime/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so')==c['native_sha256']
assert sha('/workspace/c64-dev/runtime/kt_kernel/utils/llamafile.py')==c['wrapper_sha256']
name=c['profile'];assert name in ('R16','R8')
base=Path('/workspace/results/r65-r67-serial-v5')
a.output.mkdir(parents=True,exist_ok=False)
for name_in in ('cmmlu.json',):shutil.copy2(base/name_in,a.output/name_in)
manifest=a.output/'cmmlu.json'
assert sha(manifest)=='bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04'
profile=a.canary.parent/(name+'.json')
assert sha(profile)==c['profile_sha256']
os.environ.update(C64_TP='2',KT_R67_CPU_SET='128-191',KT_R67_NPU_ID='0')
os.environ.pop('KT_C64_CAPTURE_DIR',None)
freeze=dict(canary_sha256=sha(a.canary),performance_sha256=sha(a.performance),
    quality_entry_policy='USER_AUTHORIZED_SCREEN' if a.user_authorized_quality_screen else 'BENEFICIAL_ISOLATED_PERFORMANCE',
    formal_performance=perf.get('formal_performance',True),
    r8_smoke_sha256=sha(a.r8_smoke),memory_audit_sha256=sha(a.performance.with_name('memory-audit.json')),
    manifest_sha256=sha(manifest),profile_sha256=sha(profile),product_floor_pp=-5,
    reference_sha256=sha(base/'R64/quality.json'),workers=64,numa=[4,5],batch=1,
    native_sha256=sha('/workspace/c64-dev/runtime/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'))
write(a.output/'QUALITY_FREEZE.json',freeze)
report=dict(status='RUNNING',profile=name)
try:
    result=quality(a.output,profile,a.output,json.loads(manifest.read_text()))
    ref=json.loads((base/'R64/quality.json').read_text())
    assert sha(base/'R64/quality.json')==freeze['reference_sha256']
    assert sha(manifest)==freeze['manifest_sha256']
    gate=product_gate(result['correct'],ref['correct'])
    report.update(status='PASS' if gate['pass_5pp'] and result['invalid']==0 else 'FAIL',
        correct=result['correct'],baseline_correct=ref['correct'],invalid=result['invalid'],
        **gate,paired=paired(result['rows'],ref['rows']))
except BaseException as exc:
    report.update(status='FAIL',error=repr(exc));raise
finally:write(a.output/'summary.json',report)
