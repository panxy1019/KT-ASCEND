"""Read-only target-container capacity/source inventory; no model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def command(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    return dict(command=args, returncode=p.returncode, output=p.stdout)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path('/workspace/kt-src')
    model=Path('/workspace/models/DeepSeek-V2-Lite-604d5664')
    config=json.loads((model/'config.json').read_text())
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    layers=[i for i in range(config['first_k_dense_replace'], config['num_hidden_layers'])
            if i % config['moe_layer_freq']==0]
    keys=[f'model.layers.{l}.mlp.experts.{e}.{pr}_proj.weight'
          for l in layers for e in range(config['n_routed_experts']) for pr in ('gate','up','down')]
    files={}
    for path in ['/proc/meminfo','/proc/vmstat','/proc/pressure/memory','/proc/self/status',
                 '/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current',
                 '/sys/fs/cgroup/memory.stat','/sys/fs/cgroup/memory.events',
                 '/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpuset.cpus.effective',
                 '/sys/fs/cgroup/cpuset.mems.effective','/sys/bus/pci/devices/0000:c1:00.0/numa_node']:
        try: files[path]=Path(path).read_text()
        except OSError as exc: files[path]=repr(exc)
    for path in Path('/sys/devices/system/node').glob('node*/meminfo'):
        files[str(path)]=path.read_text()
    inventory=[]
    for path in Path('/workspace/results/gguf').glob('*.manifest.json'):
        doc=json.loads(path.read_text())
        inventory.append(dict(path=str(path),layers=doc.get('layers'),gguf_sha256=doc.get('gguf_sha256'),
                              gguf_size=doc.get('gguf_size')))
    commands=[['lscpu'],['df','-h','/workspace'],['git','-C',str(root),'rev-parse','HEAD'],
              ['git','-C',str(root/'third_party/sglang'),'rev-parse','HEAD'],
              ['git','-C',str(root),'status','--short']]
    report=dict(started_unix=time.time(),commands=[command(c) for c in commands],files=files,
                cpu_affinity=sorted(os.sched_getaffinity(0)),config=config,
                config_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
                expected_routed_tensors=len(keys),missing_tensors=[k for k in keys if k not in index],
                moe_layers=layers,gguf_inventory=inventory,
                shard_sizes={n:(model/n).stat().st_size for n in sorted(set(index.values()))},
                evidence_state='TARGET_READ_ONLY_PREFLIGHT_NOT_RUNTIME_QUALIFICATION')
    report['all_layer_f32_bytes']=len(layers)*config['n_routed_experts']*3*config['hidden_size']*config['moe_intermediate_size']*4
    (a.output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['missing_tensors']: raise ValueError('INCOMPLETE_HF_EXPERT_POOL')
    print(json.dumps({k:report[k] for k in ('moe_layers','expected_routed_tensors','all_layer_f32_bytes','gguf_inventory')}))


if __name__=='__main__': main()
