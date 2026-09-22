"""Gate-ordered owned workflow. Existing product processes are never stopped."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
from common import write

root=Path('/workspace/c64-dev');tools=Path(__file__).parent
lock=(root/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(status='RUNNING',phase='WAIT_OPERATOR_MATRIX',started=time.time(),steps=[])
progress=root/'development-progress.json'
write(progress,state)
def run(name,*args):
    with (root/(name+'.log')).open('x') as log:
        subprocess.run(['bash',str(tools/'with_c64_env.sh'),str(tools/(name+'.py')),*map(str,args)],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=43200)
    state['steps'].append(name);write(progress,state)
def wait_json(path,predicate,seconds=86400):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if path.exists():
            value=json.loads(path.read_text())
            if str(value.get('status','')).startswith('BLOCKED_'):
                raise RuntimeError('PREDECESSOR_BLOCKED:'+json.dumps(value))
            if predicate(value):return value
        time.sleep(10)
    raise TimeoutError('WAITING_FOR_'+str(path))
try:
    matrix=root/'operators-v5/summary.json'
    op=wait_json(matrix,lambda v:True,14400)
    assert op['status']=='PASS','OPERATOR_MATRIX_FAILED'
    state['phase']='OPERATOR_INTEGRITY_AUDIT';write(progress,state)
    run('audit_operator_matrix','--matrix',matrix.parent)
    state['phase']='R16_CANARY';write(progress,state)
    canary=root/'r16-canary-v1/summary.json'
    run('run_canary','--output',canary.parent,'--operator-summary',matrix)
    state['phase']='WAITING_FOR_EXCLUSIVE_HOST_NO_PRODUCT_INTERRUPTION';write(progress,state)
    lease=root/'host-performance-lease.json'
    wait_json(lease,lambda v:v.get('status')=='EXCLUSIVE' and time.time()-v['time_unix']<30)
    state['phase']='ISOLATED_ABBA';write(progress,state)
    perf=root/'performance-v1/summary.json'
    run('run_performance','--output',perf.parent,'--canary',canary,'--lease',lease)
    value=json.loads(perf.read_text())
    assert value['status']=='PASS','PERFORMANCE_FAILED'
    state['phase']='MEMORY_COST_AUDIT';write(progress,state)
    run('audit_memory','--performance',perf)
    if value['verdict']!='BENEFICIAL':
        state.update(status='COMPLETE',classification='C64_RUNTIME_QUALIFIED_NO_MEANINGFUL_SPEEDUP',
                     performance_verdict=value['verdict'],long_quality='NOT_RUN_PERFORMANCE_GATE')
    else:
        state['phase']='R8_C64_SMOKE';write(progress,state)
        r8=root/'r8-canary-v1/summary.json'
        with (root/'r8-canary.log').open('x') as log:
            subprocess.run(['bash',str(tools/'with_c64_env.sh'),str(tools/'run_canary.py'),
                '--output',str(r8.parent),'--operator-summary',str(matrix),'--profile','R8',
                '--performance-summary',str(perf)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=43200)
        run('r8_smoke_extension','--canary',r8)
        smoke=r8.parent/'extended-smoke/summary.json'
        state['phase']='R16_C64_LONG_QUALITY';write(progress,state)
        run('run_quality','--canary',canary,'--performance',perf,'--r8-smoke',smoke,'--output',root/'r16-quality-v1')
        q=json.loads((root/'r16-quality-v1/summary.json').read_text())
        state['phase']='R8_C64_LONG_QUALITY';write(progress,state)
        with (root/'r8-quality.log').open('x') as log:
            subprocess.run(['bash',str(tools/'with_c64_env.sh'),str(tools/'run_quality.py'),
                '--canary',str(r8),'--performance',str(perf),'--r8-smoke',str(smoke),'--output',str(root/'r8-quality-v1')],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=43200)
        state.update(status='COMPLETE',classification='C64_RUNTIME_QUALIFIED_AND_BENEFICIAL',
            r16_quality=q['status'],r8_quality=json.loads((root/'r8-quality-v1/summary.json').read_text())['status'])
except BaseException as exc:
    state.update(status='BLOCKED',error=repr(exc),classification='C64_RUNTIME_NOT_QUALIFIED');raise
finally:
    state['ended']=time.time();write(progress,state);write(root/'development-summary.json',state)
