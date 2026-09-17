"""User-authorized contended timing followed by gated quality; no other jobs stopped."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
from common import write

root=Path('/workspace/c64-dev'); tools=Path(__file__).parent
lock=(root/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(status='RUNNING',started=time.time(),steps=[],formal_performance=False,
           authorization='USER_2026_09_17_START_WITH_C32_RUNNING')
progress=root/'authorized-screen-progress.json'
def run(name,script,*args):
    state['phase']=name;write(progress,state)
    with (root/(name+'.log')).open('x') as log:
        subprocess.run(['bash',str(tools/'with_c64_env.sh'),str(tools/script),*map(str,args)],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=86400)
    state['steps'].append(name);write(progress,state)
try:
    canary=root/'r16-canary-v3/summary.json'
    perf=root/'performance-contended-v1/summary.json'
    run('authorized-performance','run_performance.py','--output',perf.parent,'--canary',canary,'--contended-exploratory')
    run('authorized-memory','audit_memory.py','--performance',perf)
    r8=root/'r8-canary-authorized-v1/summary.json'
    run('authorized-r8-canary','run_canary.py','--output',r8.parent,'--operator-summary',root/'operators-v5/summary.json',
        '--profile','R8','--performance-summary',perf,'--user-authorized-quality-screen')
    run('authorized-r8-smoke','r8_smoke_extension.py','--canary',r8)
    for name,c in [('R16',canary),('R8',r8)]:
        run('authorized-quality-'+name,'run_quality.py','--canary',c,'--performance',perf,
            '--r8-smoke',r8.parent/'extended-smoke/summary.json',
            '--output',root/(name.lower()+'-quality-authorized-v1'),'--user-authorized-quality-screen')
    state.update(status='COMPLETE',qualification='FORMAL_PERFORMANCE_STILL_PENDING',
        quality={n:json.loads((root/(n.lower()+'-quality-authorized-v1/summary.json')).read_text()) for n in ('R16','R8')})
except BaseException as exc:
    state.update(status='BLOCKED',error=repr(exc));raise
finally:
    state['updated']=time.time();write(progress,state)
