"""Streaming resource diagnostic window; never calls this a timed Pareto."""
import argparse
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[];errors=0
    with a.trace.open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if not a.start<=r['time_unix']<=a.end:continue
            if r['status']!='SAMPLED':errors+=1;continue
            procs=r['processes'];owned={p['pid'] for p in procs}
            memory=[p['memory'] for p in procs];cg=next((p['cgroup'] for p in procs if p.get('cgroup')),None)
            npu=sum(int(size) for pid,size in re.findall(r'Process id:(\d+).*?Process memory\(MB\):(\d+)',
                r.get('npu0_process_memory','')) if int(pid) in owned)
            row=dict(time_unix=r['time_unix'],rss_sum_bytes=sum(x['VmRSS'] or 0 for x in memory),
                pss_sum_bytes=sum(p['pss_bytes'] or 0 for p in procs),
                anon_sum_bytes=sum(x['RssAnon'] or 0 for x in memory),file_sum_bytes=sum(x['RssFile'] or 0 for x in memory),
                swap_sum_bytes=sum(x['VmSwap'] or 0 for x in memory),
                cgroup_current_bytes=int(cg['memory.current']) if cg else None,
                host_available_bytes=r['host_memory'].get('MemAvailable'),npu_owned_process_mib=npu)
            rows.append(row)
    fields=[k for k in rows[0] if k!='time_unix'] if rows else []
    summary={key:dict(min=min(v),max=max(v),first=v[0],last=v[-1]) for key in fields
        if (v:=[row[key] for row in rows if row[key] is not None])}
    result=dict(status='DIAGNOSTIC_WINDOW_ONLY',samples=len(rows),errors=errors,start=a.start,end=a.end,
        caveats=['RSS sums double-count shared mappings','NPU process accounting is not allocator tensor bytes',
                 'No throughput or bandwidth saturation conclusion','Sampled peak is a lower bound on true peak'],summary=summary)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
