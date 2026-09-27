"""Launch exactly one Round10 T1 F model process; no retry."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent
base=pathlib.Path('/mnt/data1/kt-round9')
out=pathlib.Path('/mnt/data1/kt-round9/evidence/20260926-round10-t6-v107')
name='kt-round10-t6-final-v107'
claim=root/'controller-F.json'
manifest=json.loads((root/'round10-source-sha256.json').read_text())
excluded={'round10-source-sha256.json','execution-budget.json','execution-budget.lock','controller-A.json','controller-B.json'}
actual={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('controller-')}
if actual!=manifest:raise RuntimeError('Round10 T1 snapshot manifest mismatch')
with claim.open('x') as stream:json.dump(dict(process='F',version=107,started=time.time(),status='LAUNCHING'),stream)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','107',
 '--storage','MXFP4','--placement','static','--resource-mode','shared',
 '--shared-training-container','0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef',
 '--bank-receipt',str(root/'rebound/bank-receipt.json'),
 '--device-gate',str(base/'evidence/20260920-c128-expert-v19/result.json'),
 '--overlay-python',str(root/'overlay-python'),'--framework-profile',str(root/'framework-profile.json'),
 '--context-target','4096','--context-reference-profile',str(base/'evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json'),
 '--streaming-profile',str(root/'streaming-profile.json'),'--cancel-trace','--cancel-closure',
 '--cost-mode','OFF','--cost-manifest',str(root/'cost-manifest.json'),
 '--cost-budget',str(root/'execution-budget.json'),'--cost-admission',a.admission]
report=dict(process='F',version=107,launch_command=args,status='LAUNCHING',started=time.time())
def save():claim.write_text(json.dumps(report,indent=2)+'\n')
def cleanup_failed_launch():
 receipt=out/'container-created.json'
 if not receipt.exists():return {'status':'NO_VERIFIED_OWNED_CONTAINER_RECEIPT'}
 expected=json.loads(receipt.read_text())['Id']
 found=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0]
 if found['Id']!=expected:raise RuntimeError('failed-launch owned container identity changed')
 stop_started=time.time();was_running=found['State']['Running']
 if was_running:subprocess.run(['docker','stop','-t','10',expected],check=True,timeout=30)
 ack=wait_for_guard_exit(out/'monitor.jsonl',not_before=stop_started if was_running else report['started'])
 subprocess.run(['docker','rm',expected],check=True,timeout=30)
 return {'status':'REMOVED_AFTER_GUARD_ACK','guard_exit_ack':ack,'container_id':expected}
try:
 subprocess.run(args,check=True,timeout=300)
 start=json.loads((out/'model-start.json').read_text())
 report.update(status='SUPERVISING',container_id=start['container_id']);save()
 supervisor=['python3',str(root/'round10_t1_supervisor.py'),'--process','F','--version','107',
  '--container',start['name'],'--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round10-t1-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:
 report['ended']=time.time();save()
