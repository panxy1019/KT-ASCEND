"""Provisional MXFP4 admission; other banks retain the historical floor."""
def parse_node(raw):
    return {v[2].rstrip(':'): int(v[3]) for line in raw.splitlines()
            if len(v := line.split()) >= 4}

def evaluate_nodes(raw_nodes, storage):
    if storage not in ('BF16', 'Q8_0', 'MXFP4'):
        raise ValueError('unknown bank storage')
    floor = 48 if storage == 'MXFP4' else 80
    records = {}
    for node, raw in raw_nodes.items():
        d = parse_node(raw)
        # Diagnostic estimate only: it is not a kernel NUMA MemAvailable API.
        reclaim = max(0, d['Inactive(file)'] - d['Dirty'] - d['Writeback']
                      - d['Shmem'] - d['Mapped'])
        effective = max(0, d['MemFree'] + reclaim - 8 * 1024**2)
        if d['MemFree'] < floor * 1024**2:
            raise RuntimeError(f'NUMA{node} below {floor}GiB free for {storage}')
        if storage == 'MXFP4' and effective < (59.75 / 4 + 32) * 1024**2:
            raise RuntimeError(f'NUMA{node} conservative reclaim estimate insufficient')
        records[node] = dict(meminfo_kib=d, effective_estimate_kib=effective)
    return dict(version=1, storage=storage, free_floor_gib=floor,
                runtime_free_floor_gib=8, status='PROVISIONAL_MONITORED_VALIDATION',
                nodes=records)
