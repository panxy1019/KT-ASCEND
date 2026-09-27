"""Bounded threshold supervisor with guard-ACK-before-remove cleanup."""
import argparse,fcntl,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);p.add_argument('--budget',type=pathlib.Path,required=True);a=p.parse_args()
if a.container!='kt-round9-full-model-v90' or len(a.container_id)!=64:raise ValueError('fixed owned container identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 d=json.loads(command(['docker','inspect',a.container]))[0]
 if d['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return d
report={'status':'WAITING_FOR_SERVER','mode':'CANDIDATE129','container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0,'readiness_http_requests':0}
path=a.out/'threshold-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\n')
save();child=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  d=inspect()
  if not d['State']['Running']:raise RuntimeError('server exited before threshold completion')
  if any(f.stat().st_size>512*1024**2 for f in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'threshold-results.json').exists():raise RuntimeError('threshold ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/request_driver.py',
     '--mode','CANDIDATE129','--out','/out','--fixture',a.driver_root+'/threshold-request-fixtures.json',
     '--v71-fixture',a.driver_root+'/request-fixtures.json',
     '--ack-reference',a.driver_root+'/THRESHOLD_ACK_REFERENCE.json']
    report.update(status='RUNNING_THRESHOLD',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'threshold-driver.log').open('w') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode
   result=json.loads((a.out/'threshold-results.json').read_text())
   if result.get('mode')!='CANDIDATE129' or result.get('generation_attempts') != 7 or result.get('abort_attempts') != 0:raise RuntimeError('threshold result budget/identity mismatch')
   report.update(requests_recorded=len(result['attempts']),requests_completed=sum(x['state']=='PASS' for x in result['attempts']),request_status=result['status'])
   report['status']='PASS' if child.returncode==0 and result['status']=='PASS' else 'FAIL';break
  time.sleep(5)
 else:raise TimeoutError('bounded threshold supervisor deadline')
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  d=inspect();stop_started=time.time();was_running=d['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  report['owned_exit_state']=inspect()['State']
  report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 save()
result=json.loads((a.out/'threshold-results.json').read_text()) if (a.out/'threshold-results.json').exists() else {}
with a.budget.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);budget=json.loads(a.budget.read_text())
 budget['results']['CANDIDATE129']={'generation_attempts':result.get('generation_attempts',0),'abort_attempts':result.get('abort_attempts',0),'supervisor_status':report['status'],'out':str(a.out)}
 budget['generation_attempts_used']=result.get('generation_attempts',0);budget['abort_controls_used']=result.get('abort_attempts',0)
 temp=a.budget.with_suffix('.tmp');temp.write_text(json.dumps(budget,indent=2)+'\n');temp.replace(a.budget)
if report['status']!='PASS':raise SystemExit(1)
