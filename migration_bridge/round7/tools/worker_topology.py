"""Explicit C64 relocation; no automatic core-count or topology changes."""
import os
from pathlib import Path

A3_CPUS = list(range(0,64,2)) + list(range(80,144,2))
A3_CPUSET = ','.join(map(str,A3_CPUS))


def cpu_ids(value):
    result=set()
    for item in value.split(','):
        bounds=item.split('-')
        result.update(range(int(bounds[0]),int(bounds[-1])+1))
    return result


def topology(cfg, check_host=True):
    allowed = {'128-191': [4, 5], '0-63': [0, 1], '64-127': [2, 3]}
    cpus = cfg['cpu_set']
    nodes = cfg.get('numa_nodes', [4, 5])
    if cfg.get('topology_profile') == 'a3-smt-c64-v1':
        if cfg['workers']!=64 or cpus!=A3_CPUSET or nodes!=[0,1]:
            raise ValueError('FROZEN_A3_C64_TOPOLOGY_REQUIRED')
        if check_host:
            if not set(A3_CPUS)<=os.sched_getaffinity(0):
                raise ValueError('WORKER_CPUSET_UNAVAILABLE')
            identities=[]
            for index,cpu in enumerate(A3_CPUS):
                node=nodes[index//32]
                members=cpu_ids(Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip())
                if cpu not in members:raise ValueError('WORKER_NUMA_TOPOLOGY_DRIFT')
                base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
                identities.append(((base/'physical_package_id').read_text().strip(),
                                   (base/'core_id').read_text().strip()))
            if len(set(identities))!=64:raise ValueError('SMT_SIBLING_DOUBLE_COUNT')
        return cpus,nodes
    if cfg.get('topology_profile') is not None:
        raise ValueError('UNKNOWN_TOPOLOGY_PROFILE')
    if cfg['workers'] != 64 or allowed.get(cpus) != nodes:
        raise ValueError('FROZEN_C64_TOPOLOGY_REQUIRED')
    lo, hi = map(int, cpus.split('-'))
    expected = set(range(lo, hi + 1))
    if check_host:
        if not expected <= os.sched_getaffinity(0):
            raise ValueError('WORKER_CPUSET_UNAVAILABLE')
        for node, start in zip(nodes, (lo, lo + 32)):
            actual = Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip()
            if actual != f'{start}-{start + 31}':
                raise ValueError('WORKER_NUMA_TOPOLOGY_DRIFT')
    return cpus, nodes


def output_root(path):
    path = Path(path)
    if '..' in path.parts or not path.is_absolute() or not any(
        path.is_relative_to(base) for base in
        ('/workspace/c64-dev/round7', '/workspace/parallel-worker')):
        raise ValueError('UNOWNED_OUTPUT_ROOT')
    return path
