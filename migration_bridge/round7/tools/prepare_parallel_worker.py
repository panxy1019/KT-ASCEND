"""Prepare isolated results, retaining byte-identical imported runtime artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from worker_topology import topology, output_root


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-set',required=True)
    p.add_argument('--numa-nodes',type=int,nargs=2,required=True)
    p.add_argument('--physical-npu',type=int,required=True)
    p.add_argument('--container',required=True)
    a=p.parse_args()
    output_root(a.output)
    cfg=json.loads((a.source/'uniform-dev-config.json').read_text())
    cfg.update(cpu_set=a.cpu_set,numa_nodes=a.numa_nodes)
    topology(cfg)
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.source/'profiles',a.output/'profiles')
    raw=(a.source/'provenance.json').read_bytes()
    prov=json.loads(raw)
    prov['inherited_source_provenance']=dict(path=str(a.source/'provenance.json'),
        sha256=hashlib.sha256(raw).hexdigest(),scope='IMPORTED_ARTIFACT_IDENTITY_NOT_NEW_WORKER_EXECUTION')
    prov.update(container=a.container,physical_npu=a.physical_npu,logical_npu=0,
                cpu_set=a.cpu_set,numa_nodes=a.numa_nodes)
    cfg['provenance']=str(a.output/'provenance.json')
    for name,value in [('provenance.json',prov),('uniform-dev-config.json',cfg)]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED_NOT_QUALIFIED',container=a.container,
        cpu_set=a.cpu_set,numa_nodes=a.numa_nodes,physical_npu=a.physical_npu)))


if __name__=='__main__':main()
