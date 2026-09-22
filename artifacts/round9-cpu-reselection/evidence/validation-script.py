import pathlib,json,subprocess,time,os,itertools,sys
root=pathlib.Path('/mnt/data1/kt-round9/evidence/20260922-shadow-decomposition-v1');out=root/'cpu-reselection';out.mkdir(exist_ok=True)
def cpulist(s):
 result=[]
 for item in s.strip().split(','):
  if not item:continue
  a,_,b=item.partition('-');result+=range(int(a),int(b or a)+1)
 return result
allowed=os.sched_getaffinity(0);cores={};nodes={}
for nodepath in sorted(pathlib.Path('/sys/devices/system/node').glob('node[0-9]*')):
 node=int(nodepath.name[4:]);nodes[node]=dict(distance=nodepath.joinpath('distance').read_text().strip())
 for cpu in cpulist(nodepath.joinpath('cpulist').read_text()):
  if cpu not in allowed:continue
  c=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');key=(int(c.joinpath('physical_package_id').read_text()),int(c.joinpath('core_id').read_text()))
  row=cores.setdefault(key,dict(node=node,socket=key[0],core=key[1],siblings=[]));assert row['node']==node;row['siblings'].append(cpu)
containers=[];reserved=set()
for cid in subprocess.check_output(['docker','ps','-q'],text=True).split():
 d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];s=d['HostConfig']['CpusetCpus'];containers.append(dict(id=d['Id'],name=d['Name'],cpuset=s,mems=d['HostConfig']['CpusetMems']))
 reserved.update(cpulist(s))
for row in cores.values():row['excluded_explicit_container_affinity']=bool(set(row['siblings'])&reserved)
corelist=sorted(cores.values(),key=lambda x:(x['node'],min(x['siblings'])))
def stats():
 return {int(v[0][3:]):list(map(int,v[1:])) for l in pathlib.Path('/proc/stat').read_text().splitlines() if (v:=l.split())[0].startswith('cpu') and v[0]!='cpu'}
def memory():
 return {str(n):{v[2].rstrip(':'):int(v[3]) for l in pathlib.Path(f'/sys/devices/system/node/node{n}/meminfo').read_text().splitlines() if len(v:=l.split())>=4} for n in nodes}
record=dict(started=time.time(),host=subprocess.check_output(['hostname','-I'],text=True).strip(),cores=corelist,nodes=nodes,containers=containers,pci_numa=pathlib.Path('/sys/bus/pci/devices/0000:83:00.0/numa_node').read_text().strip(),policy='6x10s discovery; per-core idle=min(all SMT siblings); exclude explicit container cpus and all siblings; iowait not idle; no allocations',windows=[])
prev=stats()
for i in range(6):
 time.sleep(10);now=stats();idle={str(c):(now[c][3]-prev[c][3])/max(1,sum(now[c][:8])-sum(prev[c][:8])) for c in allowed if c in now and c in prev};prev=now
 row=dict(index=i,time=time.time(),idle=idle,memory=memory());record['windows'].append(row)
 (out/(sys.argv[1] if len(sys.argv)>1 else 'discovery.json')).write_text(json.dumps(record,indent=2))
 print('WINDOW',i+1,{n:round(sum(min(idle[str(c)] for c in x['siblings']) for x in corelist if x['node']==n)/sum(x['node']==n for x in corelist),3) for n in nodes},flush=True)
print('DISCOVERY_DONE',flush=True)
