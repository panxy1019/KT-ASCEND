"""Bounded one-container request-cost orchestration; never controls training."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);p.add_argument('--mode',choices=['OFF','ON'],required=True);p.add_argument('--budget',type=pathlib.Path,required=True);a=p.parse_args()
if a.container!='kt-round9-full-model-v71' or len(a.container_id)!=64:raise ValueError('fixed owned container identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 d=json.loads(command(['docker','inspect',a.container]))[0]
 if d['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return d
report={'status':'WAITING_FOR_SERVER','container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0,'readiness_http_requests':0}
path=a.out/'cost-supervisor.json'
path.touch(exist_ok=False)  # Atomic single-supervisor claim; never overwrite/restart.
def save():path.write_text(json.dumps(report,indent=2)+'\n')
save();child=None
try:
 initial=inspect();start=time.time();deadline=start+21600
 while time.time()<deadline:
  d=inspect()
  if not d['State']['Running']:raise RuntimeError('server exited before request-cost completion')
  log=a.out/'server.log'
  if any(f.stat().st_size>512*1024**2 for f in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  if child is None and log.exists():
   with log.open('rb') as f:
    f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'cost-results.json').exists():raise RuntimeError('request ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/request_driver.py','--mode',a.mode,'--out','/out']
    report.update(status='RUNNING_REQUEST_COST',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'cost-driver.log').open('w') as f:child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode
   result=json.loads((a.out/'cost-results.json').read_text());report['requests_recorded']=len(result['attempts']);report['requests_completed']=sum(x['state']=='PASS' for x in result['attempts']);report['request_status']=result['status']
   report['status']='PASS' if child.returncode==0 and result['status']=='PASS' else 'FAIL';break
  time.sleep(5)
 else:raise TimeoutError('bounded model/request-cost supervisor deadline')
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  d=inspect()
  stop_started=time.time()
  was_running=d['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  d=inspect();report['owned_exit_state']=d['State']
  # Keep the stopped container inspectable until its guard observes terminal
  # state. Removing first caused v64's post-completion telemetry exception.
  report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
 except BaseException as exc:report['cleanup_error']=repr(exc);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 save()
print(json.dumps(report),flush=True)

# Single active model by construction; persistent budget only moves forward.
import fcntl
with a.budget.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 budget=json.loads(a.budget.read_text());results=budget.setdefault('results',{})
 raw=json.loads((a.out/'cost-results.json').read_text()) if (a.out/'cost-results.json').exists() else {}
 results[a.mode]=dict(generation_attempts=len(raw.get('attempts',[])),abort_attempts=raw.get('abort_attempts',0),supervisor_status=report['status'],out=str(a.out))
 budget['generation_attempts_used']=sum(x['generation_attempts'] for x in results.values())
 budget['abort_controls_used']=sum(x['abort_attempts'] for x in results.values())
 temp=a.budget.with_suffix('.tmp');temp.write_text(json.dumps(budget,indent=2)+'\n');temp.replace(a.budget)
