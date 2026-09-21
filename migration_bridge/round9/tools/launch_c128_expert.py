"""Fresh bounded C128 admission and real layer-0 expert probe on physical15.

No performance qualification. Never stops unrelated containers or processes.
"""
import argparse, hashlib, json, os, pathlib, subprocess, time, shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--version',type=int,required=True)
p.add_argument('--probe',choices=['wrapper','ktep','dynamic'],default='wrapper')
p.add_argument('--storage',choices=['BF16','Q8_0','MXFP4'],default='BF16')
p.add_argument('--candidate-native',action='store_true')
p.add_argument('--npu-format',type=int,choices=[2,29],default=2)
p.add_argument('--resource-mode',choices=['shared','exclusive'],default='shared')
p.add_argument('--resident-capacity',type=int,choices=[32,64],default=64)
a=p.parse_args()
if a.resident_capacity==32 and (a.probe!='dynamic' or a.npu_format!=29 or not a.candidate_native):
 raise ValueError('Q32 gate requires real NZ29 Dynamic and explicit candidate native')
if a.probe=='wrapper' and a.storage!='BF16':raise ValueError('lowbit uses complete KTEP probe')
assert a.version>0
root=pathlib.Path('/mnt/data1/kt-round9')
source=pathlib.Path(__file__).resolve().parents[1]
name=f'kt-round9-c128-expert-v{a.version}'
out=root/'evidence'/f'20260920-c128-expert-v{a.version}'
out.mkdir(exist_ok=False)
def command(args):return subprocess.check_output(args,text=True,timeout=30)
assert '110.123.0.3' in command(['hostname','-I']).split()
nodes=[2,3,6,7]
cpus=[]; physical=set(); topology=[]
for node in nodes:
 for cpu in range(node*80,(node+1)*80,2):
  path=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}')
  assert path.joinpath(f'node{node}').exists()
  core=(path/'topology/core_id').read_text().strip()
  socket=(path/'topology/physical_package_id').read_text().strip()
  assert (socket,core) not in physical
  physical.add((socket,core)); cpus.append(cpu)
  topology.append(dict(cpu=cpu,core=f'{socket}:{core}',numa=node))
assert len(physical)==160 and set(cpus)<=os.sched_getaffinity(0)
def stats():
 return {int(v[0][3:]):list(map(int,v[1:])) for line in pathlib.Path('/proc/stat').read_text().splitlines() if (v:=line.split())[0].startswith('cpu') and v[0]!='cpu'}
before=stats();time.sleep(1);after=stats()
idle={c:(after[c][3]-before[c][3])/max(1,sum(after[c][:8])-sum(before[c][:8])) for c in cpus}
assert sum(idle.values())/len(idle)>.8,'selected CPUs busy; retry fresh admission later'
mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0])>128*1024**2
node_mem={n:pathlib.Path(f'/sys/devices/system/node/node{n}/meminfo').read_text() for n in nodes}
for raw in node_mem.values():
 free=next(int(line.split()[-2]) for line in raw.splitlines() if 'MemFree:' in line)
 assert free>8*1024**2,'require8GiB genuinely free per selected node for bounded probe'
npu=command(['npu-smi','info'])
# This launch admits an idle board with the full future15GiB training allowance.
# If training starts/restarts, the live guard stops our probe on its unknown PID.
import re
assert not re.search(r'\|\s*7\s+1\s*\|\s*\d+\s*\|',npu),'refresh training PID admission before sharing active process'
row=next(x for x in npu.splitlines() if '0000:83:00.0' in x)
used,capacity=map(int,re.findall(r'(\d+)\s*/\s*(\d+)',row)[-1])
assert capacity==65536 and used+(15360 if a.resource_mode=='shared' else 0)+4096+8192<=capacity
receipt=dict(status='ADMITTED_BOUNDED_C128_EXPERT_ONLY',time=time.time(),host='110.123.0.3',physical=15,
 numa_nodes=nodes,topology=topology,logical_cpus=cpus,cpu_quota=136,idle_fraction=idle,
 host_meminfo=mem,node_meminfo=node_mem,npu_smi=npu,resource_mode=a.resource_mode,training_allowance_mib=15360 if a.resource_mode=='shared' else 0,
 owned_hbm_budget_mib=4096,board_reserve_mib=8192,host_memory_limit_gib=64,
 probe=a.probe,storage=a.storage,resident_capacity=a.resident_capacity,scope='CURRENT_CAPACITY_NOT_EXCLUSIVE_RESERVATION_NO_FULL_MODEL_ADMISSION')
(out/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
probe_file={'wrapper':'c128_expert_probe.py','ktep':'c128_ktep_probe.py','dynamic':'c128_ktep_probe.py'}[a.probe]
if a.resident_capacity==32:probe_file='q32_ktep_probe.py'
frozen=out/'source'
for subdir in ('tests','tools','runtime'):
 shutil.copytree(source/subdir,frozen/subdir,ignore=shutil.ignore_patterns('__pycache__'))
(out/'source-sha256.json').write_text(json.dumps({str(path.relative_to(frozen)):hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen.rglob('*') if path.is_file()},indent=2)+'\n')
extra=' --dynamic' if a.probe=='dynamic' else ''
if a.probe!='wrapper':extra+=f' --npu-format {a.npu_format}'
bank='/round9/banks/layer0-v1/layer-00-bf16.gguf'
runtime='/round9/candidate-runtime-python' if a.candidate_native else '/round9/runtime-python'
if a.storage!='BF16':
 bank=f'/round9/banks/candidate-layer0-{a.storage.lower()}-v1/layer-00-{a.storage.lower()}.gguf'
 extra+=f' --storage {a.storage} --bank-manifest /round9/banks/candidate-layer0-{a.storage.lower()}-v1/manifest.json'
 runtime='/round9/candidate-runtime-python' 
script=f'source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; exec timeout -k 10 600 python /out/source/tests/{probe_file}{extra} --model /models/{rev} --identity-receipt /models/{rev}.round9-support/destination-verification/verification-1789886190714568317.json --bank {bank} --output /out/result.json > /out/probe.log 2>&1'
runtime_root=root/runtime.removeprefix('/round9/')
(out/'runtime-source-sha256.json').write_text(json.dumps({str(p.relative_to(runtime_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
sglang_root=root/'src/sglang-round9-port/python'
(out/'sglang-source-sha256.json').write_text(json.dumps({str(p.relative_to(sglang_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sglang_root.rglob('*.py')},indent=2)+'\n')
args=['docker','create','--name',name,'--network','none','--cpuset-cpus',','.join(map(str,cpus)),
 '--cpuset-mems',','.join(map(str,nodes)),'--cpus','136','--memory','64g','--memory-swap','64g',
 '--pids-limit','1024','--shm-size','1g','--cap-add','SYS_NICE']
for device in ('davinci15','davinci_manager','devmm_svm','hisi_hdc'):args+=['--device','/dev/'+device]
for mount in (f'{root}:/round9:ro','/home/admin/models/Qwen3.8-Flash-Next:/models:ro',f'{out}:/out','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'):args+=['-v',mount]
for env in (f'PYTHONPATH=/round9/src/sglang-round9-port/python:{runtime}','ASCEND_RT_VISIBLE_DEVICES=0','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2'):args+=['-e',env]
args+=['--entrypoint','bash','601e76d17e9f','-c',script]
command(args)
inspect=json.loads(command(['docker','inspect',name]))[0]
(out/'container-created.json').write_text(json.dumps(inspect,indent=2)+'\n')
assert inspect['HostConfig']['NanoCpus']==136*10**9 and inspect['HostConfig']['Memory']==64*1024**3
with (out/'guard.log').open('w') as log:
 guard=subprocess.Popen(['python3',str(source/'tools/monitor_owned_server.py'),'--container',name,
 '--output',str(out/'monitor.jsonl'),'--ready-file',str(out/'guard-ready.json'),
 '--resource-mode',a.resource_mode,'--seconds','650','--owned-limit-mib','4096'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
 deadline=time.monotonic()+40
 while not (out/'guard-ready.json').exists():
  assert guard.poll() is None,'guard failed'
  assert time.monotonic()<deadline,'guard readiness timeout'
  time.sleep(.2)
 assert time.time()-receipt['time']<60,'admission expired'
 command(['docker','start',name])
 print('STARTED',name,flush=True)
except BaseException:
 subprocess.run(['docker','stop','--time','10',name],check=False,timeout=30)
 raise
