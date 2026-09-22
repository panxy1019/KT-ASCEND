"""Resume only failed short timing, then both frozen quality screens."""
import fcntl,json,subprocess,time
from pathlib import Path
from common import write
root=Path('/workspace/c64-dev');tools=Path(__file__).parent
lock=(root/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(status='RUNNING',started=time.time(),steps=[],admission_policy='CGROUP_ZERO_SWAP_HOST_NO_ACTIVE_SWAP_V2')
def run(name,script,*args):
    state['phase']=name;write(root/'admission-recovery-progress.json',state)
    with (root/(name+'.log')).open('x') as log:
        subprocess.run(['bash',str(tools/'with_c64_env.sh'),str(tools/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    state['steps'].append(name)
try:
    r8=root/'r8-canary-authorized-v1/summary.json'
    run('recovered-r8-timing','r8_smoke_extension.py','--canary',r8,'--resume-local')
    for n,c in [('R16',root/'r16-canary-v3/summary.json'),('R8',r8)]:
        run('recovered-quality-'+n,'run_quality.py','--canary',c,'--performance',root/'performance-first-abba-v1/summary.json',
            '--r8-smoke',r8.parent/'extended-smoke/summary.json','--output',root/(n.lower()+'-quality-authorized-v1'),'--user-authorized-quality-screen')
    state['status']='COMPLETE'
    state['quality']={n:json.loads((root/(n.lower()+'-quality-authorized-v1/summary.json')).read_text()) for n in ('R16','R8')}
    state['final_review']='PENDING_FULL_COMPARISON_AND_PERFORMANCE_INTERPRETATION'
except BaseException as exc:
    state.update(status='BLOCKED',error=repr(exc));raise
finally:
    state['updated']=time.time();write(root/'admission-recovery-progress.json',state)
