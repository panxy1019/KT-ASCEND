"""Fresh exclusive physical15/C128 capacity check. No reservation claim."""
import os,pathlib,re,subprocess,time

def sample():
 def command(args):return subprocess.check_output(args,text=True,timeout=30)
 if '110.123.0.3' not in command(['hostname','-I']).split():raise RuntimeError('wrong host')
 nodes=[2,3,6,7];cpus=[];cores=set();topology=[]
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
 for raw in numa.values():
  free=next(int(line.split()[-2]) for line in raw.splitlines() if 'MemFree:' in line)
  if free<80*1024**2:raise RuntimeError('less than80GiB free on selected NUMA node')
 npu=command(['npu-smi','info']);row=next(x for x in npu.splitlines() if '0000:83:00.0' in x)
 used,total=map(int,re.findall(r'(\d+)\s*/\s*(\d+)',row)[-1])
 if re.search(r'\|\s*7\s+1\s*\|\s*\d+\s*\|',npu):raise RuntimeError('physical15 not idle; exclusive admission denied')
 if total!=65536 or used+46080+8192>total:raise RuntimeError('insufficient exclusive HBM model+reserve')
 return dict(sampled_at=time.time(),host='110.123.0.3',physical=15,resource_mode='exclusive',cpu_quota=136,
  cpus=cpus,numa_nodes=nodes,topology=topology,idle_fraction=idle,host_meminfo=mem,numa_meminfo=numa,
  npu_smi=npu,baseline_hbm_mib=used,model_hbm_budget_mib=46080,allocator_limit_mib=43008,owned_limit_mib=45056,reserve_hbm_mib=8192,
  training_allowance_mib=0,host_limit_gib=640,scope='CANDIDATE_EXCLUSIVE_FULL_MODEL_PREFLIGHT')
