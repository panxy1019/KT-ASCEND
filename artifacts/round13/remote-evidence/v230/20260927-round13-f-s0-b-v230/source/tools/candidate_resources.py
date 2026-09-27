"""Fresh physical12/C128 capacity check. No reservation claim."""
import os,pathlib,re,subprocess,time

def resource_contract(resource_mode):
 if resource_mode=='exclusive':
  return dict(scope='CANDIDATE_EXCLUSIVE_FULL_MODEL_PREFLIGHT',training_allowance_mib=0,
              model_hbm_budget_mib=46080,allocator_limit_mib=43008,owned_limit_mib=45056)
 if resource_mode=='shared':
  # This leaves at least the live guard's 8.5GiB board reserve after the
  # user-attested 15GiB training peak and the observed system baseline.
  return dict(scope='CANDIDATE_SHARED_FULL_MODEL_PREFLIGHT',training_allowance_mib=15360,
              model_hbm_budget_mib=37888,allocator_limit_mib=36352,owned_limit_mib=37376)
 raise ValueError('resource_mode must be shared or exclusive')

def physical12_processes(raw):
 return {int(match.group(1)):int(match.group(2)) for line in raw.splitlines()
         if (match:=re.match(r'\|\s*6\s+0\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|',line))}

def cgroup_matches_container(raw,container_id):
 if re.fullmatch(r'[0-9a-f]{64}',container_id) is None:
  raise ValueError('training container ID must be a full Docker ID')
 return any(line.split(':',2)[-1]==f'/docker/{container_id}'
            for line in raw.splitlines())

def sample(storage='BF16',resource_mode='exclusive',training_container_id=None):
 def command(args):return subprocess.check_output(args,text=True,timeout=30)
 if '110.123.0.3' not in command(['hostname','-I']).split():raise RuntimeError('wrong host')
 nodes=[2,3,4,7];cpus=[];cores=set();topology=[]
 for node in nodes:
  for cpu in range(node*80,(node+1)*80,2):
   path=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}')
   if not (path/f'node{node}').exists():raise RuntimeError('NUMA topology changed')
   core=((path/'topology/physical_package_id').read_text().strip(),(path/'topology/core_id').read_text().strip())
   if core in cores:raise RuntimeError('SMT sibling counted twice')
   cores.add(core);cpus.append(cpu);topology.append(dict(cpu=cpu,core=core,numa=node))
 if len(cores)!=160 or not set(cpus)<=os.sched_getaffinity(0):raise RuntimeError('physical CPU capacity unavailable')
 def stats():
  return {int(v[0][3:]):list(map(int,v[1:])) for line in pathlib.Path('/proc/stat').read_text().splitlines() if (v:=line.split())[0].startswith('cpu') and v[0]!='cpu'}
 before=stats();time.sleep(1);after=stats()
 idle={c:(after[c][3]-before[c][3])/max(1,sum(after[c][:8])-sum(before[c][:8])) for c in cpus}
 if sum(idle.values())/len(idle)<=.8:raise RuntimeError('selected CPUs busy')
 mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
 if int(mem['MemAvailable'].split()[0])<704*1024**2:raise RuntimeError('less than640GiB model+64GiB host reserve available')
 numa={n:pathlib.Path(f'/sys/devices/system/node/node{n}/meminfo').read_text() for n in nodes}
 from numa_policy import evaluate_nodes
 numa_policy=evaluate_nodes(numa,storage)
 npu=command(['npu-smi','info']);row=next(x for x in npu.splitlines() if '0000:85:00.0' in x)
 used,total=map(int,re.findall(r'(\d+)\s*/\s*(\d+)',row)[-1])
 contract=resource_contract(resource_mode);processes=physical12_processes(npu)
 if resource_mode=='exclusive' and training_container_id is not None:
  raise RuntimeError('exclusive admission cannot bind a training container')
 if training_container_id is not None:
  for pid in processes:
   try:raw=pathlib.Path(f'/proc/{pid}/cgroup').read_text()
   except OSError as error:raise RuntimeError('training process changed during cgroup admission') from error
   if not cgroup_matches_container(raw,training_container_id):
    raise RuntimeError(f'physical12 process {pid} is outside admitted training container')
 training=sum(processes.values())
 if resource_mode=='exclusive' and processes:raise RuntimeError('physical12 not idle; exclusive admission denied')
 if training>contract['training_allowance_mib']:raise RuntimeError('physical12 training exceeds admitted15GiB peak')
 baseline=used-training
 if not 0<=baseline<15360:raise RuntimeError('invalid physical12 system HBM baseline')
 if (total!=65536 or baseline+contract['training_allowance_mib']
         +contract['owned_limit_mib']>56832):
  raise RuntimeError('insufficient HBM for admitted mode, training peak and board reserve')
 return dict(sampled_at=time.time(),host='110.123.0.3',physical=12,resource_mode=resource_mode,cpu_quota=136,
  cpus=cpus,numa_nodes=nodes,numa_policy=numa_policy,topology=topology,idle_fraction=idle,host_meminfo=mem,numa_meminfo=numa,
  npu_smi=npu,baseline_hbm_mib=baseline,training_pids=sorted(processes),
  training_container_id=training_container_id,
  reserve_hbm_mib=8704,host_limit_gib=640,**contract)
