"""Explicit C64 relocation; no automatic core-count or topology changes."""
import os
from pathlib import Path


def topology(cfg, check_host=True):
    allowed = {'128-191': [4, 5], '0-63': [0, 1], '64-127': [2, 3]}
    cpus = cfg['cpu_set']
    nodes = cfg.get('numa_nodes', [4, 5])
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
