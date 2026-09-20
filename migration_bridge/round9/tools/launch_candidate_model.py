"""Bounded exclusive physical15 candidate trial. Preserved baseline is untouched."""
import argparse,hashlib,json,pathlib,shutil,subprocess,time
from candidate_resources import sample
from install_candidate_overlay import tree_files,tree_digest
p=argparse.ArgumentParser()
p.add_argument('--version',type=int,required=True)
p.add_argument('--storage',choices=['BF16','Q8_0','MXFP4'],required=True)
p.add_argument('--placement',choices=['static','request_boundary'],required=True)
p.add_argument('--bank-receipt',type=pathlib.Path,required=True)
p.add_argument('--device-gate',type=pathlib.Path,required=True)
p.add_argument('--overlay-python',type=pathlib.Path,required=True)
a=p.parse_args()
root=pathlib.Path('/mnt/data1/kt-round9');source=root/'candidate-dev-20260920/migration_bridge/round9'
if a.version<10:raise ValueError('candidate full-model versions begin at10')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(args):return subprocess.check_output(args,text=True,timeout=30)
def tree(path):return {str(p.relative_to(path)):digest(p) for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
def container_path(path):return '/round9/'+str(path.resolve().relative_to(root))
name=f'kt-round9-full-model-v{a.version}';out=root/'evidence'/f'20260920-full-model-v{a.version}'
profile=root/'candidate-profiles-v1'/f'{a.storage}-{a.placement}.json'
doc=json.loads(profile.read_text());candidate_sha=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest()
receipt=json.loads(a.bank_receipt.read_text());gate=json.loads(a.device_gate.read_text())
if gate['status']!='PASS_REAL_KTEP_LOCAL_MERGE' or gate['storage']!=a.storage or gate['cpu_workers']!=128:
 raise RuntimeError('real C128 candidate gate required')
if a.placement=='request_boundary' and len(gate.get('dynamic_migrations',[]))<3:raise RuntimeError('Dynamic device gate missing')
if receipt['status']!='VERIFIED48_BANK_BYTES_TYPES_SOURCE_NOT_MODEL_QUALIFIED' or receipt['profile_sha256']!=candidate_sha or receipt['source_identity']!=gate['source_identity']:
 raise RuntimeError('bank/source/candidate gate identity mismatch')
if receipt['banks'][0]['sha256']!=gate['bank_sha256']:raise RuntimeError('gate is from a different layer0 bank')
runtime=root/'candidate-runtime-python'
if json.loads((a.device_gate.parent/'runtime-source-sha256.json').read_text())!=tree(runtime):raise RuntimeError('native runtime differs from device gate')
# Installed scheduler/model payload is separately frozen below; local device gate
# binds the preserved KTEP port, not full-model scheduler qualification.
if json.loads((a.device_gate.parent/'sglang-source-sha256.json').read_text())!={str(p.relative_to(root/'src/sglang-round9-port/python')):digest(p) for p in (root/'src/sglang-round9-port/python').rglob('*.py')}:
 raise RuntimeError('KTEP port differs from device gate')
pre=sample();out.mkdir(exist_ok=False)
for sub in ('runtime','tools','tests'):shutil.copytree(source/sub,out/'source'/sub,ignore=shutil.ignore_patterns('__pycache__'))
values=json.loads((source/'runtime/profiles/q64_bf16_c64_sync_physical15.json').read_text())
values.update(kt_cpuinfer=128,kt_threadpool_count=4,kt_numa_nodes=[2,3,6,7],kt_weight_path=receipt['bank_root'],served_model_name=f'KT-ASCEND-R9-C128-{a.storage}-{a.placement}')
(out/'server-profile.json').write_text(json.dumps(values,indent=2)+'\n')
sglang=a.overlay_python.resolve(strict=True)
overlay_path=sglang/'round9-candidate-overlay.json'
overlay=json.loads(overlay_path.read_text())
if overlay['baseline_tree_sha256']!=tree_digest(tree_files(root/'src/sglang-round9-port/python')) or overlay['candidate_tree_sha256']!=tree_digest(tree_files(sglang)):
 raise RuntimeError('installed overlay differs from recorded baseline/candidate payload')
(out/'payload-sha256.json').write_text(json.dumps(dict(source=tree(out/'source'),runtime=tree(runtime),sglang=tree(sglang),bank_receipt=digest(a.bank_receipt),device_gate=digest(a.device_gate)),indent=2)+'\n')
script='source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; exec timeout -k 10 3600 python -u /out/source/runtime/candidate_server.py --profile /out/server-profile.json --resolved-profile /out/resolved-profile.json --admission /out/admission.json > /out/server.log 2>&1'
args=['docker','create','--name',name,'--network','none','--cpuset-cpus',','.join(map(str,pre['cpus'])),'--cpuset-mems','2,3,6,7','--cpus','136','--memory','640g','--memory-swap','640g','--pids-limit','4096','--shm-size','16g','--cap-add','SYS_NICE']
for dev in ('davinci15','davinci_manager','devmm_svm','hisi_hdc'):args+=['--device','/dev/'+dev]
for mount in (f'{root}:/round9:ro','/home/admin/models/Qwen3.8-Flash-Next:/models:ro',f'{out}:/out','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'):args+=['-v',mount]
for env in ('ASCEND_RT_VISIBLE_DEVICES=0','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','ROUND9_MEMORY_OBSERVER=1',
 f'PYTHONPATH={container_path(sglang)}:{container_path(runtime)}',f'ROUND9_CANDIDATE_PROFILE={container_path(profile)}',
 f'ROUND9_BANK_RECEIPT={container_path(a.bank_receipt)}',f'ROUND9_BANK_RECEIPT_SHA256={digest(a.bank_receipt)}',
 'ROUND9_DYNAMIC_EVENTS=/out/dynamic-events.jsonl',f'ROUND9_DYNAMIC_BOUNDARY_ENABLE={int(a.placement=="request_boundary")}'):args+=['-e',env]
args+=['--entrypoint','bash','601e76d17e9f','-c',script];command(args)
inspect=json.loads(command(['docker','inspect',name]))[0];hc=inspect['HostConfig']
if hc['Memory']!=640*1024**3 or hc['MemorySwap']!=hc['Memory'] or hc['NanoCpus']!=136*10**9:raise RuntimeError('container limits differ')
(out/'container-created.json').write_text(json.dumps(inspect,indent=2)+'\n')
with (out/'guard.log').open('w') as log:
 guard=subprocess.Popen(['python3',str(out/'source/tools/monitor_owned_server.py'),'--container',name,'--resource-mode','exclusive','--output',str(out/'monitor.jsonl'),'--ready-file',str(out/'guard-ready.json'),'--seconds','3600'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
 deadline=time.monotonic()+40
 while not (out/'guard-ready.json').exists():
  if guard.poll() is not None:raise RuntimeError('guard failed')
  if time.monotonic()>deadline:raise RuntimeError('guard readiness timeout')
  time.sleep(.2)
 admission=sample();admission.update(status='CANDIDATE_READY_FOR_MONITORED_FULL_MODEL_RUN',profile_sha256=digest(out/'server-profile.json'),candidate_profile_sha256=candidate_sha,payload_sha256=digest(out/'payload-sha256.json'),overlay_sha256=digest(overlay_path),
  bank_identity=dict(receipt_sha256=digest(a.bank_receipt),catalog_sha256=receipt['catalog_sha256'],source_identity=receipt['source_identity']))
 (out/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
 if guard.poll() is not None:raise RuntimeError('guard died before start')
 if tree(out/'source')!=json.loads((out/'payload-sha256.json').read_text())['source']:raise RuntimeError('frozen candidate source changed')
 command(['docker','start',name]);print('STARTED_MONITORED_CANDIDATE',name,flush=True)
except BaseException:
 subprocess.run(['docker','stop','--time','10',name],timeout=30,check=False);raise
