"""Read container swap across cgroup versions; fail closed if unavailable."""
from pathlib import Path
import time


def swap_bytes(root=Path('/sys/fs/cgroup')):
    v2=root/'memory.swap.current'
    if v2.exists():return int(v2.read_text())
    values={k:int(v) for k,v in (line.split() for line in
        (root/'memory/memory.stat').read_text().splitlines())}
    # v1 hierarchical swap accounting avoids subtracting two racing usage reads.
    if 'total_swap' not in values:raise ValueError('CGROUP_SWAP_ACCOUNTING_UNAVAILABLE')
    return values['total_swap']


def admission():
    fields={line.split(':',1)[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines() if line.endswith('kB')}
    swap=swap_bytes()
    if swap:raise ValueError('CONTAINER_SWAP_USED')
    if fields['SwapTotal']!=fields['SwapFree']:
        def counters():
            return {k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines())
                    if k in ('pswpin','pswpout')}
        before=counters();time.sleep(2);after=counters()
        if before!=after:raise ValueError('ACTIVE_HOST_SWAPPING')
        fields['host_swap_activity_check']=dict(before=before,after=after,seconds=2)
    fields.update(cgroup_swap_bytes=swap,admission_policy='CGROUP_ZERO_SWAP_HOST_NO_ACTIVE_SWAP_V2')
    if fields['MemAvailable']<max(64*2**30,fields['MemTotal']*.2):raise ValueError('HOST_MEMORY_RESERVE')
    return fields
