"""Provisional MXFP4 admission; other banks retain the historical floor."""
def parse_node(raw):
    return {v[2].rstrip(':'): int(v[3]) for line in raw.splitlines()
            if len(v := line.split()) >= 4}

def conservative_available_kib(d):
    reclaim = max(0, d['Inactive(file)'] - d['Dirty'] - d['Writeback']
                  - d['Shmem'] - d['Mapped'])
    return max(0, d['MemFree'] + reclaim - 8 * 1024**2)

def runtime_pressure(nodes, cgroup):
    for key in ('memory.failcnt',):
        if int(cgroup.get(key, '0')) > 0:
            return 'owned cgroup memory allocation failure'
    events = dict(line.split() for line in cgroup.get('memory.events','').splitlines())
    if any(int(events.get(key,'0')) > 0 for key in ('high','oom','oom_kill')):
        return 'owned cgroup memory pressure/OOM event'
    for node,d in nodes.items():
        if d['MemFree'] < 8*1024**2 and conservative_available_kib(d) < 8*1024**2:
            return f'NUMA{node} free and conservative reclaim headroom below reserve'
    return None

def evaluate_nodes(raw_nodes, storage):
    if storage not in ('BF16', 'Q8_0', 'MXFP4'):
        raise ValueError('unknown bank storage')
    floor = 48 if storage == 'MXFP4' else 80
    records = {}
    for node, raw in raw_nodes.items():
        d = parse_node(raw)
        # Diagnostic estimate only: it is not a kernel NUMA MemAvailable API.
        effective = conservative_available_kib(d)
        if d['MemFree'] < floor * 1024**2:
            raise RuntimeError(f'NUMA{node} below {floor}GiB free for {storage}')
        if storage == 'MXFP4' and effective < (59.75 / 4 + 32) * 1024**2:
            raise RuntimeError(f'NUMA{node} conservative reclaim estimate insufficient')
        records[node] = dict(meminfo_kib=d, effective_estimate_kib=effective)
    return dict(version=1, storage=storage, free_floor_gib=floor,
                runtime_free_floor_gib=8, status='PROVISIONAL_MONITORED_VALIDATION',
                nodes=records)
