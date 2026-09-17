"""Resource gate, not a quality/numerical threshold. Freeze before timing."""
import argparse
import json
from pathlib import Path
import statistics
from common import write,sha

def fields(text):
    return {r.split(':')[0]:int(r.split()[1])*1024 for r in text.splitlines() if r.endswith('kB')}
def maj(process):
    raw=process['stat'];return int(raw[raw.rfind(')')+2:].split()[9])

p=argparse.ArgumentParser();p.add_argument('--performance',type=Path,required=True);a=p.parse_args()
d=json.loads(a.performance.read_text());assert d['status']=='PASS'
rows=[];storm=0;maxstorm=0;previous_session=None
for r in d['rows']:
    current_session=(r['block'],r['position'])
    if current_session!=previous_session:storm=0
    previous_session=current_session
    before=r['memory_before'];after=r['memory_after'];mem=fields(after['files']['/proc/meminfo'])
    assert mem['SwapTotal']==mem['SwapFree'],'SWAP'
    assert mem['MemAvailable']>=max(64*2**30,.2*mem['MemTotal']),'MEMORY_RESERVE'
    events=dict(x.split() for x in after['files']['/sys/fs/cgroup/memory.events'].splitlines())
    assert int(events.get('oom',0))==0 and int(events.get('oom_kill',0))==0,'CGROUP_OOM'
    old={p['pid']:maj(p) for p in before['processes']}
    faults=sum(max(0,maj(p)-old[p['pid']]) for p in after['processes'] if p['pid'] in old)
    rate=faults/max(after['time_unix']-before['time_unix'],1e-6)
    storm=storm+1 if rate>=100 else 0;maxstorm=max(maxstorm,storm)
    sums={k:sum(fields(p['status']).get(k,0) for p in after['processes']) for k in ('VmRSS','RssAnon','RssFile')}
    rows.append(dict(mode=r['mode'],shape=r['shape'],**sums,major_faults=faults,major_faults_per_second=rate,
        host_available=mem['MemAvailable'],cgroup_current=int(after['files']['/sys/fs/cgroup/memory.current']),
        cgroup_peak=after['files'].get('/sys/fs/cgroup/memory.peak')))
summary={mode:{key:statistics.median(x[key] for x in rows if x['mode']==mode)
    for key in ('VmRSS','RssAnon','RssFile','cgroup_current')} for mode in ('C32','C64')}
report=dict(status='PASS' if maxstorm<3 else 'FAIL',summary=summary,rows=rows,
    delta={k:summary['C64'][k]-summary['C32'][k] for k in summary['C32']},
    rss_scope='SUM_OF_PROCESS_RSS_NOT_PSS',major_fault_storm='>=100/s FOR >=3 CONSECUTIVE MEASURED REQUESTS',
    max_consecutive_storm_intervals=maxstorm,performance_sha256=sha(a.performance),
    dual_c64_running=False,dual_c64_admission='NOT_GRANTED_BY_THIS_SINGLE_CONTAINER_TEST')
write(a.performance.parent/'memory-audit.json',report)
assert report['status']=='PASS','SUSTAINED_MAJOR_FAULT_STORM'
