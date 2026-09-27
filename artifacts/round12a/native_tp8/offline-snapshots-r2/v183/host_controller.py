"""One threshold launch and one supervisor; never retries the model."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent
base=pathlib.Path('/mnt/data1/kt-round9')
out=base/'evidence'/'20260925-plan15-threshold-v90'
name='kt-round9-full-model-v90'
claim=root/'controller-CANDIDATE129.json'
with claim.open('x') as f:json.dump(dict(mode='CANDIDATE129',started=time.time(),status='LAUNCHING'),f)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','90',
 '--storage','MXFP4','--placement','static','--resource-mode','shared',
 '--shared-training-container','0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef',
 '--bank-receipt',str(root/'rebound/bank-receipt.json'),
 '--device-gate',str(base/'evidence/20260920-c128-expert-v19/result.json'),
 '--overlay-python',str(root/'overlay-python'),'--framework-profile',str(root/'framework-profile.json'),
 '--context-target','4096','--context-reference-profile',str(base/'evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json'),
 '--streaming-profile',str(root/'streaming-profile.json'),'--cancel-trace','--cancel-closure',
 '--cost-mode', 'OFF','--cost-manifest',str(root/'cost-manifest.json'),
 '--cost-budget',str(root/'execution-budget.json'),'--cost-admission',a.admission]
report=dict(mode='CANDIDATE129',launch_command=args,status='LAUNCHING',started=time.time())
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
 supervisor=['python3',str(root/'threshold_supervisor.py'),'--container',start['name'],
  '--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base)),'--budget',str(root/'execution-budget.json')]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'threshold-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:
 report['ended']=time.time();save()
