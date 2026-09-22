"""Read-only evidence accounting. Upper bounds permit reselecting each window,
so failing them proves no fixed136-core subset passes that observed window.
No profile is published unless the requested threshold is supported.
"""
import json,pathlib,itertools,datetime
base=pathlib.Path(__file__).parent
datasets=[json.loads((base/'evidence'/f).read_text()) for f in ['discovery.json','validation.json']]
rows=[];node_summary={};windows=[]
for data in datasets:
 for w in data['windows']:
  allowed_nodes={n for n,m in w['memory'].items() if m['MemFree']>=48*1024**2}
  eligible=[c for c in data['cores'] if not c['excluded_explicit_container_affinity'] and str(c['node']) in allowed_nodes]
  # Optimistic: independently choose the better sibling and best136 cores every
  # window; this is an upper bound, NOT an admissible fixed-affinity profile.
  best=sorted((max(w['idle'][str(c)] for c in core['siblings']) for core in eligible),reverse=True)
  strict=sorted((min(w['idle'][str(c)] for c in core['siblings']) for core in eligible),reverse=True)
  assert all(0<=x<=1 for x in best+strict)
  windows.append(dict(time=w['time'],eligible_physical_cores=len(eligible),optimistic_best136_mean=sum(best[:136])/136 if len(best)>=136 else None,strict_best136_mean=sum(strict[:136])/136 if len(strict)>=136 else None,max_eligible_thread_idle=max(best) if best else None))
  for n,m in w['memory'].items():
   cores=[c for c in data['cores'] if str(c['node'])==n]
   row=node_summary.setdefault(n,dict(free_gib=[],strict_all_core_idle=[],representative_idle=[],eligible_cores=[]))
   row['free_gib'].append(m['MemFree']/1024**2)
   row['strict_all_core_idle'].append(sum(min(w['idle'][str(c)] for c in core['siblings']) for core in cores)/len(cores))
   row['representative_idle'].append(sum(w['idle'][str(min(core['siblings']))] for core in cores)/len(cores))
   row['eligible_cores'].append(sum(not c['excluded_explicit_container_affinity'] for c in cores))
for n,row in node_summary.items():
 node_summary[n]={k:dict(min=min(v),max=max(v)) for k,v in row.items()}
failed=[w for w in windows if w['optimistic_best136_mean'] is None or w['optimistic_best136_mean']<=.8]
report=dict(status='NO_QUALIFYING_CPU_NUMA_SET_OBSERVED' if failed else 'UPPER_BOUND_NOT_A_SELECTION',policy=dict(minimum_physical_cores=136,preferred_shape='4NUMA x (32workers+2aux); legacy160-cpu allowed envelope also considered',minimum_free_gib_per_node=48,average_idle_strictly_greater_than=.8,explicit_container_sibling_exclusion=True,measurement_windows=len(windows),seconds_per_window=10,independent_rounds=2),node_summary=node_summary,window_upper_bounds=windows,failed_upper_bound_windows=len(failed),selected_profile=None,physical_starts=0,scope='Observed intervals only, not a prediction or future reservation',discovery_pci_numa=datasets[0]['pci_numa'])
(base/'selection-result.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status']);print('windows',len(windows),'failed upper bound',len(failed));print('upper bounds',min(w['optimistic_best136_mean'] for w in windows),max(w['optimistic_best136_mean'] for w in windows))
