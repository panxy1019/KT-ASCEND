"""Serial shared-physical15 component admission; at most one OFF and one ON start.

No full model, HTTP, cache eviction, training control, or automatic retries.
"""
import argparse,hashlib,json,pathlib,subprocess,sys,time

def validate_pair_identity(before, after):
 for key in ('manifest_sha256','capture_sha256','capture_meta_sha256'):
  if before.get(key) is None or before.get(key)!=after.get(key):raise ValueError('OFF/ON identity mismatch: '+key)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['off','on'],required=True);p.add_argument('--version',type=int,required=True);a=p.parse_args()
 root=pathlib.Path('/mnt/data1/kt-round9');source=pathlib.Path(__file__).resolve().parents[1]
 if not source.is_relative_to(root/'shadow-decomposition-20260922-v1'):raise ValueError('requires unique frozen snapshot')
 agg=root/'evidence/20260922-shadow-decomposition-v1';agg.mkdir(exist_ok=True)
 def command(args):return subprocess.check_output(args,text=True,timeout=40)
 # Binding to current SHA manifest happens before any container creation.
 manifest=source.parent/'source-sha256.json';mh=hashlib.sha256(manifest.read_bytes()).hexdigest()
 for name,h in json.loads(manifest.read_text()).items():
  path=(source/name).resolve();path.relative_to(source.resolve())
  if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError('frozen source mismatch '+name)
 training='0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef'
 train=json.loads(command(['docker','inspect',training]))[0]
 if train['Id']!=training or not train['State']['Running']:raise RuntimeError('training identity/state changed')
 from candidate_resources import sample
 try:admission=sample('MXFP4','shared',training)
 except Exception as exc:
  (agg/('denied-'+a.mode+'.json')).write_text(json.dumps(dict(status='BLOCKED_WITH_EVIDENCE',time=time.time(),reason=str(exc),component_starts=0),indent=2));raise
 if any(n.startswith('kt-round9-') for n in command(['docker','ps','--format','{{.Names}}']).splitlines()):raise RuntimeError('another Round9 container active')
 ledger_path=agg/'run-ledger.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'starts':[],'full_model_starts':0,'http_requests':0}
 if len(ledger['starts'])>=2 or any(x['mode']==a.mode for x in ledger['starts']):raise RuntimeError('component budget already used')
 if a.mode=='on':
  if len(ledger['starts'])!=1 or ledger['starts'][0]['mode']!='off':raise RuntimeError('OFF first')
  off=json.loads(pathlib.Path(ledger['starts'][0]['out'],'result.json').read_text())
  if off['status']!='COMPONENT_CASES_VALIDATED':raise RuntimeError('OFF did not validate')
 pair_identity=dict(manifest_sha256=mh,capture_sha256=hashlib.sha256((root/'evidence/20260920-full-model-v63/expert-replay-decode.pt').read_bytes()).hexdigest(),capture_meta_sha256=hashlib.sha256((root/'evidence/20260920-full-model-v63/expert-replay-decode.json').read_bytes()).hexdigest())
 if pair_identity['capture_sha256']!='de82a710a2fd56c7587a91dbc5c37b8989ff0265a502391d797588c77c6e8fc6' or pair_identity['capture_meta_sha256']!='6724418fa2e5def0b27cb97be2708da13a5471c8ad02cd82d85d4d14dea6d2cc':raise ValueError('capture identity changed')
 if a.mode=='on':validate_pair_identity(ledger['starts'][0],pair_identity)
 name=f'kt-round9-layerwise-v{a.version}';out=root/'evidence'/f'20260922-shadow-{a.mode}-v{a.version}';out.mkdir(exist_ok=False)
 (out/'admission.json').write_text(json.dumps(admission,indent=2));(out/'component-budget.json').write_text(json.dumps(dict(owned_limit_mib=4096,allocator_fraction=.04,allocator_max_bytes=2631841873,host_gib=64,workspace_mib=512,wall_seconds=1200,trace_events=2048,trace_bytes=8388608),indent=2))
 relative='/round9/'+str(source.relative_to(root));manifest_in='/round9/'+str(manifest.relative_to(root))
 import shlex
 probe=['python',relative+'/tests/shadow_decomposition_probe.py','--mode',a.mode,'--capture','/round9/evidence/20260920-full-model-v63/expert-replay-decode.pt','--source-manifest',manifest_in,'--source-manifest-sha256',mh,'--output','/out/result.json']
 if a.mode=='on':probe+=['--off-result','/round9/'+str(pathlib.Path(ledger['starts'][0]['out'],'result.json').relative_to(root))]
 script='source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; exec timeout -k 10 1200 '+shlex.join(probe)+' > /out/probe.log 2>&1'
 args=['docker','create','--name',name,'--network','none','--cpuset-cpus',','.join(map(str,admission['cpus'])),'--cpuset-mems','2,3,6,7','--cpus','136','--memory','64g','--memory-swap','64g','--pids-limit','1024','--shm-size','1g','--cap-add','SYS_NICE']
 for d in ['davinci15','davinci_manager','devmm_svm','hisi_hdc']:args+=['--device','/dev/'+d]
 for m in [str(root)+':/round9:ro','/home/admin/models/Qwen3.8-Flash-Next:/models:ro',str(out)+':/out','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']:args+=['-v',m]
 for env in ['PYTHONPATH=/round9/framework-dev-20260921-p1-timeline-v6/overlay-python:/round9/candidate-runtime-python','ASCEND_RT_VISIBLE_DEVICES=0','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2']:args+=['-e',env]
 args+=['--entrypoint','bash','601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a','-c',script]
 (out/'launch-command.json').write_text(json.dumps(args,indent=2));cid=command(args).strip();(out/'container-created.json').write_text(command(['docker','inspect',cid]))
 with (out/'guard.log').open('w') as log:
  guard=subprocess.Popen(['python3',str(source/'tools/monitor_owned_server.py'),'--container',name,'--output',str(out/'monitor.jsonl'),'--ready-file',str(out/'guard-ready.json'),'--resource-mode','shared','--training-container-id',training,'--seconds','1250','--owned-limit-mib','4096','--numa-nodes','2','3','6','7'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  deadline=time.monotonic()+40
  while not (out/'guard-ready.json').exists():
   if guard.poll() is not None or time.monotonic()>deadline:raise RuntimeError('guard not ready')
   time.sleep(.2)
  if time.time()-admission['sampled_at']>=60:raise RuntimeError('admission stale')
  if json.loads(command(['docker','inspect',name]))[0]['Id']!=cid:raise RuntimeError('container identity changed')
  ledger['starts'].append(dict(mode=a.mode,id=cid,out=str(out),time=time.time(),**pair_identity));ledger_path.write_text(json.dumps(ledger,indent=2))
  print(command(['docker','start',cid]),flush=True)
 except BaseException:
  subprocess.run(['docker','stop','-t','10',cid],timeout=30,check=False);raise
if __name__=='__main__':main()
