"""First-divergence diagnostics and observer common-history parity, not token fitting."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--diagnostics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    n=json.loads((a.capture/'npu/capture-result.json').read_text())
    h=json.loads((a.capture/'hybrid/capture-result.json').read_text())
    g=json.loads(a.generation.read_text())
    assert n['status']==h['status']=='CAPTURE_INTEGRITY_PASS' and g['status']=='GENERATION_PASS'
    assert n['inputs_sha256']==h['inputs_sha256']==g['input_sha256']
    natural={(r['rid'],r['step']):r['natural_token'] for r in
             map(json.loads,(a.capture/'hybrid/forcing.jsonl').read_text().splitlines())}
    baseline={r['rid']:r['baseline_output_ids'] for r in n['rows']}
    diagnostics={(r['rid'],r['token_index']):r for r in json.loads(a.diagnostics.read_text())['rows']}
    rows=[]
    for r in g['rows']:
        rid=r['rid'];tokens=r['reference64'];reference=baseline[rid]
        assert len(tokens)==len(reference)==64
        first=next((i for i,(x,y) in enumerate(zip(reference,tokens)) if x!=y),None)
        length=64 if first is None else first+1
        parity=all(tokens[i]==natural[rid,i] for i in range(length))
        if not parity:raise ValueError('OBSERVER_COMMON_HISTORY_MISMATCH:'+rid)
        rows.append(dict(rid=rid,first_divergence=first,common_history_observer_parity=True,
                         first_divergence_diagnostics=None if first is None else diagnostics[rid,first]))
    result=dict(status='FREE_GENERATION_COMMON_HISTORY_AUDIT_PASS',rows=rows,
                generation_sha256=hashlib.sha256(a.generation.read_bytes()).hexdigest(),
                role='First-divergence diagnostics under the frozen heavy-static study; not downstream quality')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
