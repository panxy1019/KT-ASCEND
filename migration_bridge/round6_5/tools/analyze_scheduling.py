"""Fit exactly one DEV frequency policy; project frozen HOLDOUT, no refitting."""
import argparse
import hashlib
import json
from pathlib import Path
from frequency_policy import project,resident_ids
from build_ratio_profiles import digest


def positions(records,requests):
    by={}
    for record in records:
        key=(record['sequence'],record['layer'])
        if key in by:raise ValueError('DUPLICATE_ROUTE_RECORD')
        by[key]=record
    expected=set()
    for request in requests:
        start=request['sequence_start'];budget=request['budget'];prompt=request['prompt']
        for offset in range(budget+1):
            rows=len(prompt['input_ids']) if offset==0 else 1
            for layer in range(1,27):
                key=(start+offset,layer);expected.add(key)
                row=by[key]
                if len(row['ids'])!=rows or len(row['weights'])!=rows:
                    raise ValueError('ROUTE_ROW_COUNT')
            if offset==budget:continue  # scheduler lookahead: never fit or score
            for index in range(rows):
                position={layer:dict(ids=by[start+offset,layer]['ids'][index],
                    weights=by[start+offset,layer]['weights'][index]) for layer in range(1,27)}
                yield prompt,('prefill' if offset==0 else 'decode'),position
    if set(by)!=expected:raise ValueError('UNACCOUNTED_ROUTE_FORWARD')


def analyze(records,requests,budget):
    counts={layer:[0]*64 for layer in range(1,27)}
    for prompt,phase,position in positions(records,requests):
        if prompt['split']!='dev':continue
        for layer,row in position.items():
            for expert in row['ids']:counts[layer][expert]+=1
    frequency={layer:resident_ids(row,budget) for layer,row in counts.items()}
    uniform={layer:list(range(0,64,64//budget)) for layer in counts}
    totals={}
    for prompt,phase,position in positions(records,requests):
        key=prompt['split']+'/'+phase
        if key not in totals:
            totals[key]={name:dict(positions=0,cpu_events=0,cpu_weight_mass=0.,
                total_weight_mass=0.,fully_npu=0) for name in ('uniform','frequency')}
        for name,mask in [('uniform',uniform),('frequency',frequency)]:
            result=project(position,mask);target=totals[key][name]
            target['positions']+=1;target['fully_npu']+=int(not result['touches_cpu'])
            for field in ('cpu_events','cpu_weight_mass','total_weight_mass'):target[field]+=result[field]
    combined={name:{field:sum(totals.get('holdout/'+phase,{}).get(name,{}).get(field,0) for phase in ('prefill','decode'))
        for field in ('positions','cpu_events','fully_npu')} for name in ('uniform','frequency')}
    u,f=combined['uniform'],combined['frequency']
    reduction=(u['cpu_events']-f['cpu_events'])/u['cpu_events'] if u['cpu_events'] else None
    increase=f['fully_npu']/f['positions']-u['fully_npu']/u['positions']
    return dict(policy='DEV_COUNT_DESC_GLOBAL_ID_ASC_TIE',resident=budget,counts=counts,
        frequency=frequency,uniform=uniform,stratified=totals,holdout=combined,
        event_relative_reduction=reduction,fully_npu_absolute_increase=increase,
        locality='SIGNAL' if (reduction is not None and reduction>=.10) or increase>=.05 else 'WEAK',
        scope='OFFLINE_SAME_HISTORY_PROJECTION_NOT_RUNTIME_QUALIFICATION')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--uniform-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=json.loads((a.capture/'report.json').read_text())
    if report['status']!='SCHEDULING_ROUTE_CAPTURE_PASS':raise ValueError('CAPTURE_PREDECESSOR')
    profile=json.loads(a.uniform_profile.read_text())
    if profile['selected_layers']!=list(range(1,27)):raise ValueError('ALL_MOE_REQUIRED')
    records=[json.loads(line) for line in (a.capture/'session/routing.jsonl').read_text().splitlines()]
    result=analyze(records,report['requests'],profile['resident'])
    a.output.mkdir(parents=True,exist_ok=False)
    profile={k:v for k,v in profile.items() if k not in ('canonical_sha256','pt_sha256')}
    profile.update(version='round6.5-dev-frequency-v1',
        scheduling_manifest_sha256=report['manifest_sha256'],
        dev_route_sha256=report['route_sha256'],policy=result['policy'])
    import torch
    counts=torch.zeros((1,27,64),dtype=torch.int64)
    for row in profile['rows']:
        row['npu_ids']=result['frequency'][row['layer']]
        row['cpu_ids']=sorted(set(range(64))-set(row['npu_ids']))
        counts[0,row['layer'],row['npu_ids']]=1
    profile['canonical_sha256']=digest(profile)
    pt=a.output/'frequency.pt';torch.save({'logical_count':counts},pt)
    profile['pt_sha256']=hashlib.sha256(pt.read_bytes()).hexdigest()
    (a.output/'frequency.json').write_text(json.dumps(profile,indent=2)+'\n')
    (a.output/'frequency_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
