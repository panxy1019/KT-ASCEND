"""Bounded Round10 T1 supervisor with guard-ACK-before-remove cleanup."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--process',choices=('A','B','R','F'),required=True);p.add_argument('--version',type=int,required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
if (a.process,a.version) not in (('A',91),('B',92),('A',93),('B',94),('R',95),('F',107)):raise ValueError('unregistered T1 process/version identity')
expected_name='kt-round10-t6-final-v107' if a.process=='F' else f'kt-round10-t1-{a.process.lower()}-v{a.version}'
if a.container!=expected_name or len(a.container_id)!=64:raise ValueError('fixed owned container identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={'status':'WAITING_FOR_SERVER','process':a.process,'version':a.version,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0,'readiness_http_requests':0}
path=a.out/'round10-t6-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\n')
save();child=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  document=inspect()
  if not document['State']['Running']:raise RuntimeError('server exited before T1 completion')
  if any(file.stat().st_size>512*1024**2 for file in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'round10-t6-results.json').exists():raise RuntimeError('T1 ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round10_t6_driver.py',
     '--process',a.process,'--out','/out','--fixture',a.driver_root+'/round10-t1-fixture.json']
    report.update(status='RUNNING_T1',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round10-t1-driver.log').open('w') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode
   result=json.loads((a.out/'round10-t6-results.json').read_text())
   generation_limit={'A':11,'B':11,'R':19,'F':8}[a.process];abort_limit={'A':1,'B':1,'R':2,'F':1}[a.process]
   if result.get('process')!=a.process or not 0<=result.get('generation_attempts',-1)<=generation_limit or not 0<=result.get('abort_controls',-1)<=abort_limit or result.get('automatic_retries')!=0:raise RuntimeError('T1 result budget/identity mismatch')
   if result.get('status')=='PASS_ROUND10_T6_PROCESS' and (result['generation_attempts']!=generation_limit or result['abort_controls']!=abort_limit):raise RuntimeError('passing T1 result did not complete its frozen schedule')
   report.update(requests_recorded=len(result['results']),request_status=result['status'])
   report['status']='PASS' if child.returncode==0 and result['status']=='PASS_ROUND10_T6_PROCESS' else 'FAIL';break
  time.sleep(5)
 else:raise TimeoutError('bounded T1 supervisor deadline')
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  document=inspect();stop_started=time.time();was_running=document['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  report['owned_exit_state']=inspect()['State']
  report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 save()
if report['status']!='PASS':raise SystemExit(1)
