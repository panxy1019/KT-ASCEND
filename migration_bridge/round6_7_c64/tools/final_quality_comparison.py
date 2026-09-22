"""Read completed quality artifacts; never infer completion from partial counts."""
import json,time
from pathlib import Path
from common import sha,write

root=Path('/workspace/c64-dev');base=Path('/workspace/results/r65-r67-serial-v5')
state_path=root/'admission-recovery-progress.json'
deadline=time.monotonic()+86400
while time.monotonic()<deadline:
    state=json.loads(state_path.read_text())
    if state['status']=='BLOCKED':raise RuntimeError('PREDECESSOR_BLOCKED:'+str(state))
    if state['status']=='COMPLETE':break
    time.sleep(30)
else:raise TimeoutError('QUALITY_NOT_COMPLETE')
paths={'R64':base/'R64/quality.json','R16-C32':base/'R16/quality.json',
       'R16-C64':root/'r16-quality-authorized-v1/quality.json',
       'R8-C64':root/'r8-quality-authorized-v1/quality.json'}
data={k:json.loads(p.read_text()) for k,p in paths.items()}
manifest='bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04'
for d in data.values():
    assert d['cmmlu_sha256']==manifest and d['count']==256 and len(d['rows'])==256
    assert len({r['id'] for r in d['rows']})==256
    assert sum(r['correct'] for r in d['rows'])==d['correct']
pairs={}
for a,b in [('R16-C64','R16-C32'),('R8-C64','R64'),('R8-C64','R16-C64')]:
    x={r['id']:r for r in data[a]['rows']};y={r['id']:r for r in data[b]['rows']}
    assert x.keys()==y.keys(),'ITEM_IDENTITY'
    pairs[a+' vs '+b]=dict(delta_pp=(data[a]['correct']-data[b]['correct'])*100/256,
        regressed=sum(y[k]['correct'] and not x[k]['correct'] for k in x),
        improved=sum(x[k]['correct'] and not y[k]['correct'] for k in x))
accepted={k:data[k]['invalid']==0 and data[k]['correct']>=data['R64']['correct']-12 for k in ('R16-C64','R8-C64')}
candidate='R8-C64' if accepted['R8-C64'] else ('R16-C64' if accepted['R16-C64'] else 'R32_C64_OR_R48_C64_NEW_QUALIFICATION_REQUIRED')
report=dict(status='COMPLETE_QUALITY_COMPARISON',artifacts={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()},
    scores={k:{f:d[f] for f in ('correct','count','invalid')} for k,d in data.items()},pairs=pairs,
    quality_accepted=accepted,quality_candidate=candidate,
    final_baseline='PENDING_PERFORMANCE_USABILITY_REVIEW',formal_performance=False)
write(root/'FINAL_QUALITY_COMPARISON.json',report)
lines=['# C64 final quality comparison','', '| Profile | Correct / 256 | Invalid |','|---|---:|---:|']
lines += [f'| {k} | {d["correct"]} | {d["invalid"]} |' for k,d in data.items()]
lines += ['', 'Quality-based candidate: '+candidate,
          '', 'Final baseline awaits performance-usability review; first ABBA is exploratory, not formal decode qualification.',
          '', '## Paired comparisons','', '```json',json.dumps(pairs,indent=2),'```']
(root/'FINAL_QUALITY_COMPARISON.md').write_text('\n'.join(lines)+'\n')
