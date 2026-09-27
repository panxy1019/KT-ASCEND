"""Bounded physical12 candidate trial. Preserved baseline is untouched."""
import argparse,hashlib,json,pathlib,re,shutil,subprocess,time
from candidate_resources import sample
from install_candidate_overlay import tree_files,tree_digest
p=argparse.ArgumentParser()
p.add_argument('--version',type=int,required=True)
p.add_argument('--storage',choices=['BF16','Q8_0','MXFP4'],required=True)
p.add_argument('--placement',choices=['static','request_boundary'],required=True)
p.add_argument('--resource-mode',choices=['shared','exclusive'],default='shared')
p.add_argument('--shared-training-container',
 help='stable Docker name/ID whose physical12 child PIDs may rotate within the shared15GiB budget')
p.add_argument('--bank-receipt',type=pathlib.Path,required=True)
p.add_argument('--device-gate',type=pathlib.Path,required=True)
p.add_argument('--overlay-python',type=pathlib.Path,required=True)
p.add_argument('--framework-profile',type=pathlib.Path)
p.add_argument('--context-target',type=int,choices=[4096,16384,65536,131072],
 help='opt-in pool sizing and actual tensor measurement, not long-context qualification')
p.add_argument('--context-reference-profile',type=pathlib.Path,
 help='opt-in BF16 compact or host-FP8/BF16-stage QSA reference consumer')
p.add_argument('--streaming-profile',type=pathlib.Path,
 help='opt-in bounded three-plane streaming prefill coordinator')
p.add_argument('--cancel-trace',action='store_true',
 help='opt in to bounded P0 cancellation lifecycle JSONL at /out/cancel-events.jsonl')
p.add_argument('--cancel-closure',action='store_true',
 help='opt in to traced chunk-boundary cancel cleanup without another model forward')
p.add_argument('--working-set-timeline',action='store_true',
 help='opt in to one bounded batched scheduler/expert/PLE timeline per request')
p.add_argument('--mainline-attribution',action='store_true',help='default-OFF sampled provider durations and bounded decode capture')
p.add_argument('--timeline-max-events',type=int,default=4096)
p.add_argument('--cost-mode',choices=['OFF','ON'],required=True)
p.add_argument('--cost-manifest',type=pathlib.Path,required=True)
p.add_argument('--cost-budget',type=pathlib.Path,required=True)
p.add_argument('--cost-admission',type=pathlib.Path,required=True)
a=p.parse_args()
# Frozen continuous admission is separate from the launcher's fresh final sample.
continuous=json.loads(a.cost_admission.read_text())
if continuous.get('status')!='PASS' or time.time()-continuous['ended_at']>600:
 raise RuntimeError('continuous admission missing/stale')
if a.version != 196 or a.cost_mode != 'OFF':
 raise RuntimeError('fixed cost trial identities required')
if a.shared_training_container!='0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef':
 raise RuntimeError('training identity changed')
root=pathlib.Path('/mnt/data1/kt-round9');source=pathlib.Path(__file__).resolve().parents[1]
if a.version<10:raise ValueError('candidate full-model versions begin at10')
if a.shared_training_container and a.resource_mode!='shared':
 raise ValueError('training container binding is valid only in shared mode')
if a.mainline_attribution and not a.working_set_timeline:
 raise ValueError('mainline attribution requires bounded timeline')
if a.working_set_timeline and a.streaming_profile is None:
 raise ValueError('working-set timeline requires the bounded three-plane streaming profile')
if not 1<=a.timeline_max_events<=65536:
 raise ValueError('timeline max events must be within1..65536')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(args):return subprocess.check_output(args,text=True,timeout=30)
def tree(path):return {str(p.relative_to(path)):digest(p) for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
def container_path(path):return '/round9/'+str(path.resolve().relative_to(root))
name='kt-round13-a-uniform-a-v196';out=root/'evidence'/'20260927-round13-a-uniform-a-v196'
profile=source.parent/'rebound/candidate-profile.json'
doc=json.loads(profile.read_text());candidate_sha=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest()
framework=None
if a.framework_profile:
 import sys
 sys.path.insert(0,str(source))
 from runtime.framework_config import parse_framework
 framework=parse_framework(json.loads(a.framework_profile.read_text()))
resident_capacity=framework['resident_capacity'] if framework else 64
receipt=json.loads(a.bank_receipt.read_text());gate=json.loads(a.device_gate.read_text())
if resident_capacity==64:
 if gate['status']!='PASS_REAL_KTEP_LOCAL_MERGE' or gate['storage']!=a.storage or gate['cpu_workers']!=128:
  raise RuntimeError('real C128 Q64 candidate gate required')
 if a.placement=='request_boundary' and (len(gate.get('dynamic_migrations',[]))!=3 or gate.get('npu_weight_formats')!={'w13_weight':29,'w2_weight':29}):raise RuntimeError('Dynamic actual NZ29 device gate missing')
if receipt['status']!='VERIFIED48_BANK_BYTES_TYPES_SOURCE_NOT_MODEL_QUALIFIED' or receipt['profile_sha256']!=candidate_sha or receipt['source_identity']!=gate['source_identity']:
 raise RuntimeError('bank/source/candidate gate identity mismatch')
if receipt['banks'][0]['sha256']!=gate['bank_sha256']:raise RuntimeError('gate is from a different layer0 bank')
runtime=root/'candidate-runtime-python'
if json.loads((a.device_gate.parent/'runtime-source-sha256.json').read_text())!=tree(runtime):raise RuntimeError('native runtime differs from device gate')
training_container_id=None
if a.shared_training_container:
 inspected=json.loads(command(['docker','inspect',a.shared_training_container]))
 if len(inspected)!=1 or not inspected[0]['State']['Running']:
  raise RuntimeError('shared training container must resolve to one running container')
 training_container_id=inspected[0]['Id']
 if re.fullmatch(r'[0-9a-f]{64}',training_container_id) is None:
  raise RuntimeError('shared training container has malformed full ID')
if resident_capacity==32:
 from runtime.q32_gate import require_q32_device_gate
 native_sha256=gate.get('native_sha256')
 if sum(digest(path)==native_sha256 for path in runtime.rglob('*') if path.is_file())!=1:
  raise RuntimeError('Q32 native extension is not uniquely present in admitted runtime')
 require_q32_device_gate(a.device_gate,digest(a.device_gate),storage=a.storage,workers=128,
  source_identity=receipt['source_identity'],layer0_sha256=receipt['banks'][0]['sha256'],native_sha256=native_sha256)
# Installed scheduler/model payload is separately frozen below; local device gate
# binds the preserved KTEP port, not full-model scheduler qualification.
if json.loads((a.device_gate.parent/'sglang-source-sha256.json').read_text())!={str(p.relative_to(root/'src/sglang-round9-port/python')):digest(p) for p in (root/'src/sglang-round9-port/python').rglob('*.py')}:
 raise RuntimeError('KTEP port differs from device gate')
pre=sample(a.storage,a.resource_mode,training_container_id);out.mkdir(exist_ok=False)
for sub in ('runtime','tools','tests'):shutil.copytree(source/sub,out/'source'/sub,ignore=shutil.ignore_patterns('__pycache__'))
values=json.loads((source/'runtime/profiles/q64_bf16_c64_sync_physical15.json').read_text())
served_name=(f'KT-ASCEND-R9-C128-{a.storage}-{a.placement}' if resident_capacity==64 else
             f'KT-ASCEND-R9-C128-Q32-{a.storage}-{a.placement}')
values.update(skip_server_warmup=True,watchdog_timeout=1800,kt_cpuinfer=128,kt_threadpool_count=4,kt_numa_nodes=[2,3,4,7],
 kt_num_gpu_experts=resident_capacity,kt_weight_path=receipt['bank_root'],
 served_model_name=served_name)
if a.context_target is not None:
 if framework is None:raise RuntimeError('context measurement requires explicit framework profile')
 values.update(max_total_tokens=a.context_target+4,context_length=a.context_target+2)
 model_config=pathlib.Path('/home/admin/models/Qwen3.8-Flash-Next')/pathlib.Path(values['model_path']).name/'config.json'
 context_probe=dict(schema=1,context_tokens=a.context_target,model_config_sha256=digest(model_config))
 (out/'context-probe.json').write_text(json.dumps(context_probe,indent=2)+'\n')
context_reference=None
if a.context_reference_profile is not None:
 from runtime.context_reference_runtime import parse_context_reference_profile
 context_reference=parse_context_reference_profile(json.loads(a.context_reference_profile.read_text()))
 (out/'context-reference-profile.json').write_bytes(a.context_reference_profile.read_bytes())
streaming=None
if a.streaming_profile is not None:
 from runtime.streaming_binding import validate_streaming_stack
 streaming=validate_streaming_stack(json.loads(a.streaming_profile.read_text()),framework,context_reference)
 (out/'streaming-profile.json').write_bytes(a.streaming_profile.read_bytes())
(out/'server-profile.json').write_text(json.dumps(values,indent=2)+'\n')
sglang=a.overlay_python.resolve(strict=True)
overlay_path=sglang/'round9-candidate-overlay.json'
overlay=json.loads(overlay_path.read_text())
if a.placement=='request_boundary':
 gate_source=json.loads((a.device_gate.parent/'source-sha256.json').read_text())
 for module in ('dynamic_adapter.py','dynamic.py','expert_source.py'):
  if digest(sglang/'sglang/srt/layers/round9'/module)!=gate_source['runtime/'+module]:raise RuntimeError('Dynamic math adapter differs from qualified device gate')
if overlay['baseline_tree_sha256']!=tree_digest(tree_files(root/'src/sglang-round9-port/python')) or overlay['candidate_tree_sha256']!=tree_digest(tree_files(sglang)):
 raise RuntimeError('installed overlay differs from recorded baseline/candidate payload')
if framework:
 if not overlay.get('framework_status'):
  raise RuntimeError('framework profile requires installed framework runtime')
 if resident_capacity==32 and framework['frequency']:
  raise RuntimeError('first Q32 qualification requires uniform placement')
 if framework['frequency'] and a.placement=='request_boundary':
  raise RuntimeError('frequency plus Dynamic requires separate interaction gate')
 (out/'framework-profile.json').write_bytes(a.framework_profile.read_bytes())
 if framework['frequency']:
  frequency=framework['frequency'];path=root/pathlib.Path(frequency['path']).relative_to('/round9')
  if digest(path)!=frequency['sha256']:raise RuntimeError('frequency payload changed before launch')
  (out/'frequency-manifest.json').write_bytes(path.read_bytes())
 if framework.get('prefill') is not None:
  expected_placement='request_boundary' if framework['prefill'].get('update_residents',False) else 'static'
  if a.placement != expected_placement or resident_capacity != 32:
   raise RuntimeError('initial layerwise system gate requires Q32 and matching boundary mode')
  from runtime.layerwise_gate import require_layerwise_device_gate
  spec=framework['prefill']['device_gate']
  layerwise_path=root/pathlib.Path(spec['path']).relative_to('/round9')
  layerwise_gate=require_layerwise_device_gate(layerwise_path,spec['sha256'],
   source_identity=receipt['source_identity'],native_sha256=gate['native_sha256'],
   source_sha256={module:digest(sglang/'sglang/srt/layers/round9'/module)
                 for module in ('layerwise.py','dynamic_adapter.py','expert_source.py')})
  backend_source=json.loads((layerwise_path.parent/'sglang-source-sha256.json').read_text())
  if digest(layerwise_path.parent/'sglang-source-sha256.json')!=layerwise_gate['sglang_tree_identity']['manifest_sha256']:
   raise RuntimeError('layerwise backend source manifest changed')
  for rel,sha in backend_source.items():
   if rel.startswith(('sglang/srt/layers/moe/','sglang/srt/layers/quantization/')):
    if digest(sglang/rel)!=sha:raise RuntimeError('layerwise MoE backend differs from device gate')
  (out/'layerwise-device-gate.json').write_bytes(layerwise_path.read_bytes())
 if framework.get('ple') is not None:
  from runtime.ple_gate import require_ple_device_gate,SOURCE_FILES
  spec=framework['ple']['device_gate'];ple_path=root/pathlib.Path(spec['path']).relative_to('/round9')
  ple_receipt=json.loads(ple_path.read_text())
  # Version strings are checked again against the actual imported Torch stack
  # before buffers are allocated. Host preflight binds source and receipt bytes.
  require_ple_device_gate(ple_path,spec['sha256'],source_identity=receipt['source_identity'],
   source_sha256={module:digest(sglang/'sglang/srt/layers/round9'/module) for module in SOURCE_FILES},
   torch_version=ple_receipt.get('torch_version'),torch_npu_version=ple_receipt.get('torch_npu_version'))
  (out/'ple-device-gate.json').write_bytes(ple_path.read_bytes())
 if framework.get('joint_host_budget',False):
  from runtime.joint_budget_gate import require_joint_budget_device_gate,SOURCE_FILES
  if a.placement!='request_boundary':raise RuntimeError('joint budget requires actual Dynamic mode')
  spec=framework['joint_device_gate'];joint_path=root/pathlib.Path(spec['path']).relative_to('/round9')
  joint_receipt=json.loads(joint_path.read_text())
  require_joint_budget_device_gate(joint_path,spec['sha256'],source_identity=receipt['source_identity'],
   source_sha256={module:digest(sglang/'sglang/srt/layers/round9'/module) for module in SOURCE_FILES},
   ple_gate_sha256=framework['ple']['device_gate']['sha256'],
   torch_version=joint_receipt.get('torch_version'),torch_npu_version=joint_receipt.get('torch_npu_version'))
  (out/'joint-device-gate.json').write_bytes(joint_path.read_bytes())
if context_reference is not None:
 from runtime.context_reference_gate import require_context_reference_gate,SOURCE_FILES
 spec=context_reference['device_gate']
 context_path=root/pathlib.Path(spec['path']).relative_to('/round9')
 context_receipt=json.loads(context_path.read_text())
 qsa_binding_path=source.parent/'ROUND12_QSA_SOURCE_BINDING.json'
 qsa_binding={}
 if qsa_binding_path.exists():
  qsa_binding=dict(qsa_source_binding_path=qsa_binding_path,
   qsa_source_binding_sha256=digest(qsa_binding_path),
   qsa_consumer_sha256=digest(sglang/'sglang/srt/models/qwen4_exp.py'),
   qsa_runtime_source_sha256=digest(sglang/'sglang/srt/layers/round9/context_reference_runtime.py'),
   qsa_binding_source_sha256=digest(sglang/'sglang/srt/layers/round9/context_reference_binding.py'))
 require_context_reference_gate(context_path,spec['sha256'],source_identity=receipt['source_identity'],
  source_sha256={module:digest(sglang/'sglang/srt/layers/round9'/module) for module in SOURCE_FILES},
  torch_version=context_receipt.get('torch_version'),torch_npu_version=context_receipt.get('torch_npu_version'),
  **qsa_binding)
 if 'context_reference_ready(self)' not in (sglang/'sglang/srt/model_executor/model_runner.py').read_text():
  raise RuntimeError('context reference model hook is not installed')
 (out/'context-reference-device-gate.json').write_bytes(context_path.read_bytes())
if streaming is not None:
 if 'streaming_model_ready(self)' not in (sglang/'sglang/srt/model_executor/model_runner.py').read_text():
  raise RuntimeError('streaming model hook is not installed')
payload=dict(source=tree(out/'source'),runtime=tree(runtime),sglang=tree(sglang),
 bank_receipt=digest(a.bank_receipt),device_gate=digest(a.device_gate))
if framework:payload['framework_profile']=digest(out/'framework-profile.json')
if a.context_target is not None:
 if 'context_pools_ready(self)' not in (sglang/'sglang/srt/model_executor/model_runner.py').read_text():
  raise RuntimeError('context pool measurement hook is not installed')
 payload['context_probe']=digest(out/'context-probe.json')
if context_reference is not None:
 payload['context_reference_profile']=digest(out/'context-reference-profile.json')
if streaming is not None:payload['streaming_profile']=digest(out/'streaming-profile.json')
(out/'payload-sha256.json').write_text(json.dumps(payload,indent=2)+'\n')
script='source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; exec timeout -k 10 21600 python -u /out/source/runtime/candidate_server.py --profile /out/server-profile.json --resolved-profile /out/resolved-profile.json --admission /out/admission.json > /out/server.log 2>&1'
args=['docker','create','--name',name,'--network','none','--cpuset-cpus',','.join(map(str,pre['cpus'])),'--cpuset-mems','2,3,4,7','--cpus','136','--memory','640g','--memory-swap','640g','--pids-limit','4096','--shm-size','16g','--cap-add','SYS_NICE']
for dev in ('davinci12','davinci_manager','devmm_svm','hisi_hdc'):args+=['--device','/dev/'+dev]
for mount in (f'{root}:/round9:ro','/home/admin/models/Qwen3.8-Flash-Next:/models:ro',f'{out}:/out','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'):args+=['-v',mount]
for env in ('ASCEND_RT_VISIBLE_DEVICES=0','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','ROUND9_MEMORY_OBSERVER=1',
 f'PYTHONPATH={container_path(sglang)}:{container_path(runtime)}',f'ROUND9_CANDIDATE_PROFILE={container_path(profile)}',
 f'ROUND9_BANK_RECEIPT={container_path(a.bank_receipt)}',f'ROUND9_BANK_RECEIPT_SHA256={digest(a.bank_receipt)}',
 f'ROUND9_DYNAMIC_DEVICE_GATE={container_path(a.device_gate)}',f'ROUND9_DYNAMIC_DEVICE_GATE_SHA256={digest(a.device_gate)}',
 'ROUND9_DYNAMIC_EVENTS=/out/dynamic-events.jsonl',f'ROUND9_DYNAMIC_BOUNDARY_ENABLE={int(a.placement=="request_boundary")}'):args+=['-e',env]
if framework:
 for env in ('ROUND9_FRAMEWORK_PROFILE=/out/framework-profile.json',
             f'ROUND9_FRAMEWORK_PROFILE_SHA256={digest(out/"framework-profile.json")}',
             'ROUND9_ROUTE_STATS=/out/route-stats.json'):args+=['-e',env]
if a.context_target is not None:
 for env in ('ROUND9_CONTEXT_PROBE_PROFILE=/out/context-probe.json',
             f'ROUND9_CONTEXT_PROBE_SHA256={digest(out/"context-probe.json")}'):args+=['-e',env]
if context_reference is not None:
 for env in ('ROUND9_CONTEXT_REFERENCE_PROFILE=/out/context-reference-profile.json',
             f'ROUND9_CONTEXT_REFERENCE_SHA256={digest(out/"context-reference-profile.json")}'):args+=['-e',env]
 if qsa_binding_path.exists():
  for env in (f'ROUND9_QSA_SOURCE_BINDING={container_path(qsa_binding_path)}',
              f'ROUND9_QSA_SOURCE_BINDING_SHA256={digest(qsa_binding_path)}'):args+=['-e',env]
if streaming is not None:
 for env in ('ROUND9_STREAMING_PROFILE=/out/streaming-profile.json',
             f'ROUND9_STREAMING_SHA256={digest(out/"streaming-profile.json")}'):args+=['-e',env]
if a.cancel_trace or a.cancel_closure:
 args+=['-e','ROUND9_CANCEL_TRACE_PATH=/out/cancel-events.jsonl']
if a.cancel_closure:args+=['-e','ROUND9_CANCEL_CLOSURE_ENABLE=1']
if a.mainline_attribution:args+=['-e','ROUND9_MAINLINE_ENABLE=1']
if a.working_set_timeline:
 for env in ('ROUND9_TIMELINE_PATH=/out/working-set-timeline.jsonl',
             f'ROUND9_TIMELINE_MAX_EVENTS={a.timeline_max_events}'):args+=['-e',env]
args+=['-e',f'ROUND9_REQUEST_COST_ENABLE={int(a.cost_mode=="ON")}',
 '-e',f'ROUND9_REQUEST_COST_MANIFEST={container_path(a.cost_manifest)}',
 '-e',f'ROUND9_REQUEST_COST_MANIFEST_SHA256={digest(a.cost_manifest)}']
args+=['-e','ROUND9_HYBRID_PROBE=1']
args+=['-e','ROUND13_PLACEMENT_PROFILE=/round9/round13-a-uniform-a-v196/PLACEMENT_PROFILES.json']
args+=['-e','ROUND13_PLACEMENT_PROFILE_SHA256='+digest(root/'round13-a-uniform-a-v196/PLACEMENT_PROFILES.json')]
args+=['-e','ROUND13_PLACEMENT_STRATEGY=uniform']
args+=['-e','ROUND13_PLACEMENT_COMPONENT_GATE=/round9/round13-a-uniform-a-v196/A_COMPONENT_GATE.json']
args+=['-e','ROUND13_PLACEMENT_COMPONENT_GATE_SHA256='+digest(root/'round13-a-uniform-a-v196/A_COMPONENT_GATE.json')]
args+=['--entrypoint','bash','601e76d17e9f','-c',script];command(args)
inspect=json.loads(command(['docker','inspect',name]))[0];hc=inspect['HostConfig']
if hc['Memory']!=640*1024**3 or hc['MemorySwap']!=hc['Memory'] or hc['NanoCpus']!=136*10**9:raise RuntimeError('container limits differ')
(out/'container-created.json').write_text(json.dumps(inspect,indent=2)+'\n')
with (out/'guard.log').open('w') as log:
 guard_args=['python3',str(out/'source/tools/monitor_owned_server.py'),'--container',name,
  '--resource-mode',a.resource_mode,'--output',str(out/'monitor.jsonl'),
  '--ready-file',str(out/'guard-ready.json'),'--seconds','21600',
  '--owned-limit-mib',str(pre['owned_limit_mib']),'--numa-nodes','2','3','4','7']
 if pre['training_pids']:
  guard_args+=['--training-pids',*map(str,pre['training_pids'])]
 if training_container_id:
  guard_args+=['--training-container-id',training_container_id]
 guard=subprocess.Popen(guard_args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
 deadline=time.monotonic()+40
 while not (out/'guard-ready.json').exists():
  if guard.poll() is not None:raise RuntimeError('guard failed')
  if time.monotonic()>deadline:raise RuntimeError('guard readiness timeout')
  time.sleep(.2)
 admission=sample(a.storage,a.resource_mode,training_container_id)
 if (training_container_id is None and
         admission['training_pids']!=pre['training_pids']):
  raise RuntimeError('physical12 training identity changed during admission')
 admission.update(status='CANDIDATE_READY_FOR_MONITORED_FULL_MODEL_RUN',profile_sha256=digest(out/'server-profile.json'),candidate_profile_sha256=candidate_sha,payload_sha256=digest(out/'payload-sha256.json'),overlay_sha256=digest(overlay_path),
  bank_identity=dict(receipt_sha256=digest(a.bank_receipt),catalog_sha256=receipt['catalog_sha256'],source_identity=receipt['source_identity'],layer0_sha256=receipt['banks'][0]['sha256']))
 if framework:admission['framework_profile_sha256']=digest(out/'framework-profile.json')
 if a.context_target is not None:admission['context_probe_sha256']=digest(out/'context-probe.json')
 if context_reference is not None:admission['context_reference_profile_sha256']=digest(out/'context-reference-profile.json')
 if streaming is not None:admission['streaming_profile_sha256']=digest(out/'streaming-profile.json')
 (out/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
 if guard.poll() is not None:raise RuntimeError('guard died before start')
 if tree(out/'source')!=json.loads((out/'payload-sha256.json').read_text())['source']:raise RuntimeError('frozen candidate source changed')
 import fcntl
 with a.cost_budget.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  budget=json.loads(a.cost_budget.read_text())
  if budget.get('stage')!='ROUND13_A_STATIC_PLACEMENT':raise RuntimeError('wrong Round13 A budget')
  starts=budget.setdefault('starts',[])
  if len(starts)>=1 or any(x['mode']==a.cost_mode for x in starts):raise RuntimeError('start budget spent; no retries')
  starts.append(dict(mode=a.cost_mode,name=name,container_id=inspect['Id'],time=time.time(),output=str(out)))
  budget['model_starts_used']=len(starts)
  temp=a.cost_budget.with_suffix('.tmp');temp.write_text(json.dumps(budget,indent=2)+'\n');temp.replace(a.cost_budget)
  (out/'model-start.json').write_text(json.dumps(starts[-1],indent=2)+'\n')
 command(['docker','start',name]);print('STARTED_MONITORED_CANDIDATE',name,flush=True)
except BaseException:
 subprocess.run(['docker','stop','--time','10',name],timeout=30,check=False);raise
