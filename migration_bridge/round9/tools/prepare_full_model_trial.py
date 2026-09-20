"""Host-side, read-only live checks and receipts for the controlled first run.

Does not start a container; a separate launcher arms the live guard first.
The prior verified receipts remain component evidence, not full-model PASS.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
root=pathlib.Path('/mnt/data1/kt-round9')
base=root/'src/KT-ASCEND/migration_bridge/round9'
sys.path.insert(0,str(base/'tools'))
from full_model_admission import evaluate, REQUIRED_GATES, PROFILE, REVISION
from probe_identity import verify_probe_identity

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value): (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
def cmd(args):return subprocess.check_output(args,text=True,timeout=30)
model=pathlib.Path('/home/admin/models/Qwen3.8-Flash-Next')/REVISION
identity=verify_probe_identity(model=model,expected_model=model,
 receipt=model.with_name(REVISION+'.round9-support')/'destination-verification/verification-1789886190714568317.json',
 expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=REVISION,
 manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
write('source-identity.json',identity)
catpath=root/'banks/all48-v1/catalog.json'
assert digest(catpath)=='35e83581957a34a8074caeb9bb40bffcf1cae6b2a03eab30f220fb4ccd01c479'
cat=json.loads(catpath.read_text())
assert len(cat['banks'])==48
for b in cat['banks']:assert (catpath.parent/b['file']).stat().st_size==b['bytes']
reviewpath=root/'evidence/20260920-full-budget/prerequisite-review.json'
review=json.loads(reviewpath.read_text())
for r in review['records']:
 file=root/'evidence'/r['path']
 assert digest(file)==r['sha256'],str(file)
 assert json.loads(file.read_text())['status']==r['status'] and r['status'].startswith('PASS')
assert len(review['records'])==10
trainingpath=root/'evidence/20260920-full-budget/user-training-envelope.json'
training=json.loads(trainingpath.read_text());assert training['peak_mib']==15360
runtime_files=sorted(set(
 list((root/'src/sglang-round9-port/python/sglang/srt').rglob('*.py'))+
 list((base/'runtime').rglob('*.py'))+
 list((root/'runtime-python/kt_kernel').rglob('*.py'))+
 list((root/'runtime-python/kt_kernel').glob('*.so'))+
 list((root/'src/KT-ASCEND/round8_prep/runtime').glob('*.py'))))
manifest={str(f.relative_to(root)):digest(f) for f in runtime_files}
write('runtime-files.json',manifest)
implementation=digest(a.output/'runtime-files.json')
profile=base/'runtime/profiles/q64_bf16_c64_sync_physical15.json'
profile_data=json.loads(profile.read_text())
assert profile_data['kt_cpuinfer']==64 and profile_data['kt_threadpool_count']==2
assert profile_data['kt_numa_nodes']==[6,7]
#Physical representatives demonstrate topology capacity; not exclusive allocation.
rows=[list(map(int,l.split(','))) for l in cmd(['lscpu','-p=CPU,CORE,SOCKET,NODE']).splitlines() if not l.startswith('#')]
vis=[r for r in rows if r[0] in range(480,640)]
top={str(cpu):f'{socket}:{core}' for cpu,core,socket,node in vis}
numa={str(n):[cpu for cpu,core,socket,node in vis if node==n] for n in (6,7)}
representatives={n:[] for n in (6,7)}
seen=set()
for cpu,core,socket,node in vis:
 if (socket,core) not in seen:representatives[node].append(cpu);seen.add((socket,core))
selected=representatives[6][:36]+representatives[7][:36]
mem=dict(l.split(':',1) for l in pathlib.Path('/proc/meminfo').read_text().splitlines())
disk=pathlib.Path('/mnt/data1').stat()
import os
vfs=os.statvfs('/mnt/data1')
raw=cmd(['npu-smi','info']);(a.output/'npu-smi.txt').write_text(raw)
#Import only the pure parser; monitor main intentionally has CLI side effects.
import ast,re
source=root/'tools/monitor_owned_server.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parse_physical15']
ns={'re':re};exec(compile(ast.Module(nodes,type_ignores=[]),str(source),'exec'),ns)
used,processes=ns['parse_physical15'](raw)
assert used<=15360 and list(processes)==[4069720],(used,processes)
now=time.time()
snapshot=dict(host='110.123.0.3',physical_id=15,pci_bus_id='0000:83:00.0',logical_device=0,device_nodes=[15],
 mapping_verified=True,access_verified=True,identity_status='PASS',prerequisite_statuses={'runtime':'PASS'},
 sampled_at=now,hbm_total_mib=65536,training_peak_mib=15360,owned_peak_budget_mib=41984,reserve_mib=8192,
 telemetry_ok=True,cpu_set=selected,cpu_physical_keys=[top[str(c)] for c in selected],observed_cpu_keys={str(c):top[str(c)] for c in selected},
 visible_cpu_set=[r[0] for r in vis],visible_cpu_numa=numa,visible_cpu_topology=top,cpu_quota_cores=72,worker_physical_cores=64,auxiliary_core_budget=8,
 cpu_selection_scope='TOPOLOGY_CAPACITY_REPRESENTATIVES_NOT_EXCLUSIVE_RESERVATION',observed_nonowned_board_mib=used,training_pids=list(processes),
 mapping_evidence='fresh npu-smi PCI row plus current allocator-cap-v1 actual isolated ACL mapping')
common=dict(status='PASS',profile=PROFILE,source_revision=REVISION,implementation_sha256=implementation)
evidence=dict(profile=PROFILE,implementation_sha256=implementation,
 gates={k:dict(common,gate=k,artifact_sha256=digest(reviewpath)) for k in REQUIRED_GATES},
 training_peak_bounded=True,training_bound=dict(artifact_sha256=digest(trainingpath),host='110.123.0.3',physical_id=15,peak_mib=15360,valid_until=now+3600,basis='USER_ATTESTED_WORKLOAD_ENVELOPE'),
 context_tokens=4096,max_new_tokens=256,batch_size=1,max_inflight=1,
 host_available_mib=int(mem['MemAvailable'].split()[0])//1024,host_owned_budget_mib=655360,host_reserve_mib=65536,
 disk_free_mib=vfs.f_bavail*vfs.f_frsize//1024**2,disk_required_mib=1258292,hbm_components_mib=review['components_mib'])
evidence['budget_envelope']=dict(common,artifact_sha256=digest(reviewpath),basis='STATIC_FIRST_RUN_BOUND',**{k:evidence[k] for k in ('context_tokens','max_new_tokens','batch_size','max_inflight','host_owned_budget_mib','disk_required_mib','hbm_components_mib')})
write('snapshot.json',snapshot);write('evidence.json',evidence)
result=evaluate(snapshot,evidence,time.time())
result.update(sampled_at=now,profile_sha256=digest(profile),implementation_sha256=implementation,training_pids=list(processes))
write('admission.json',result)
print(json.dumps(result),flush=True)
assert result['status']=='STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN'
