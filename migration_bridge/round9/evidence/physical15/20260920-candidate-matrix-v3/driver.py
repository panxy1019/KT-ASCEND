"""One-shot sequential candidate acceptance driver; not a scheduled task."""
import json,pathlib,subprocess,time,traceback
root=pathlib.Path('/mnt/data1/kt-round9')
tools=root/'candidate-dev-20260920/migration_bridge/round9/tools'
logroot=root/'evidence/20260920-candidate-matrix-v3'
logroot.mkdir(exist_ok=False)
profiles=[(19,'MXFP4','static',4,17),(20,'MXFP4','request_boundary',4,17),(21,'Q8_0','static',1,16),(22,'Q8_0','request_boundary',2,16),(23,'BF16','request_boundary',3,15)]
required=['first-http','system8','canaries','context4096','cancel-reuse']
summary={'status':'RUNNING','scope':'FUNCTIONAL_RESOURCE_ONLY','trials':[]}
def content_failure(r):
 rows=r.get('rows',[])
 if not rows or not all(x.get('numeric_finite') is True and x.get('response',{}).get('text','').strip() for x in rows):return False
 last=rows[-1];error=r.get('error','')
 return r.get('status')=='FAIL' and last.get('passed') is False and (last.get('parse_error')==error or error.startswith('AssertionError('))
def save(): (logroot/'status.json').write_text(json.dumps(summary,indent=2)+'\n')
def run(args,**kw):return subprocess.run(args,check=True,**kw)
save()
try:
 for version,storage,placement,receipt_version,gate_version in profiles:
  name=f'kt-round9-full-model-v{version}';out=root/f'evidence/20260920-full-model-v{version}'
  row=dict(version=version,storage=storage,placement=placement,status='STARTING');summary['trials'].append(row);save()
  run(['python3',str(tools/'launch_candidate_model.py'),'--version',str(version),'--storage',storage,'--placement',placement,'--overlay-python',str(root/'candidate-sglang-v8/python'),'--bank-receipt',str(root/f'evidence/20260920-bank-verify-v{receipt_version}/{storage}-{placement}.json'),'--device-gate',str(root/f'evidence/20260920-c128-expert-v{gate_version}/result.json')])
  commands=[f'python /out/source/tests/full_model_http_probe.py --wait 1200 --output /out/first-http.json > /out/first-http.log 2>&1',f'python /out/source/tests/candidate_model_acceptance.py --placement {placement} --output /out/system8.json > /out/system8.log 2>&1', 'python /out/source/tests/full_model_canaries.py --output /out/canaries.json > /out/canaries.log 2>&1 || true','python /out/source/tests/full_model_context_probe.py --output /out/context4096.json > /out/context4096.log 2>&1',f'python /out/source/tests/candidate_cancel_probe.py --placement {placement} --system8 /out/system8.json --output /out/cancel-reuse.json > /out/cancel-reuse.log 2>&1']
  run(['docker','exec','-d',name,'bash','-c','set -e; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '+'; '.join(commands)])
  row['status']='RUNNING';save();deadline=time.monotonic()+3500;previous=[]
  try:
   while True:
    reports={n:json.loads((out/(n+'.json')).read_text()) for n in required if (out/(n+'.json')).exists()}
    if any(not r['status'].startswith('PASS_') and not (n=='canaries' and content_failure(r)) for n,r in reports.items()):raise RuntimeError('driver failed: '+str({n:r.get('error',r['status']) for n,r in reports.items()}))
    if list(reports)!=previous:print(name,{n:r['status'] for n,r in reports.items()},flush=True);previous=list(reports)
    if len(reports)==len(required):break
    state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
    if not state['Running']:raise RuntimeError('owned server exited before acceptance: '+str(state))
    if time.monotonic()>deadline:raise TimeoutError('bounded candidate acceptance deadline')
    time.sleep(10)
   (out/'controlled-stop.json').write_text(json.dumps(dict(time=time.time(),reason='all resource drivers passed; content canary separately recorded without overriding any failure'))+'\n')
   run(['docker','stop','--time','20',name],timeout=40)
   for _ in range(30):
    rows=[json.loads(s) for s in (out/'monitor.jsonl').read_text().splitlines()]
    if rows[-1].get('action') in ('OWNED_EXITED','OWNED_EXITED_DURING_SAMPLE'):break
    time.sleep(1)
   finalized=subprocess.run(['python3',str(tools/'finalize_candidate_trial.py'),'--container',name,'--output',str(out)]); result=json.loads((out/'trial-summary.json').read_text()); assert result['resource_system_status']=='PASS_CANDIDATE_RESOURCE_SYSTEM'; assert finalized.returncode==(0 if result['status']=='PASS_CANDIDATE_SYSTEM_RESOURCE_ONLY' else 1)
   row['status']=json.loads((out/'trial-summary.json').read_text())['status'];save()
   time.sleep(5)
  except BaseException:
   if not (out/'controlled-stop.json').exists():(out/'controlled-stop.json').write_text(json.dumps(dict(time=time.time(),reason='candidate acceptance failure; stop only this owned trial'))+'\n')
   subprocess.run(['docker','stop','--time','20',name],timeout=40,check=False)
   raise
 summary['status']='COMPLETED_WITH_CANDIDATE_FAILURES' if any(r['status'].startswith('FAIL_') for r in summary['trials']) else 'PASS_FIVE_CANDIDATE_FUNCTIONAL_RESOURCE_TRIALS';save()
except BaseException as exc:
 summary.update(status='STOPPED_ON_FAILURE',error=repr(exc),traceback=traceback.format_exc());save();raise

if summary["status"]=="COMPLETED_WITH_CANDIDATE_FAILURES":raise SystemExit(1)
