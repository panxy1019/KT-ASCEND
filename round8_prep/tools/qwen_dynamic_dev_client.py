"""DEV-only threshold probe for one already running Q32 Dynamic candidate."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT=Path('/workspace/round8')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def request(item,label):
    payload=dict(input_ids=item['input_ids'],sampling_params=dict(
        temperature=0,max_new_tokens=16,ignore_eos=True))
    started=time.perf_counter()
    response=requests.post('http://127.0.0.1:18400/generate',json=payload,timeout=3600)
    response.raise_for_status();data=response.json();tokens=data['output_ids']
    assert len(tokens)==16
    return dict(label=label,id=item['id'],input_tokens=len(item['input_ids']),
                seconds=time.perf_counter()-started,output_ids=tokens)


def validate_events(events):
    per_layer={layer:[] for layer in range(40)}
    for event in events:
        assert event['weights']==event['mapping']==event['mask']==event['wrapper_mask']
        assert event['weights']==event['prior_generation']+1
        assert event['weight_identity_exact'] and event['mask_mapping_wrapper_exact']
        assert len(event['selected'])==len(set(event['selected']))==32
        per_layer[event['layer']].append(event)
    assert all(per_layer.values())
    for layer,values in per_layer.items():
        assert [x['weights'] for x in values]==list(range(1,len(values)+1)),layer
    return per_layer


def main():
    p=argparse.ArgumentParser();p.add_argument('--threshold',type=int,choices=(256,512,1024),required=True)
    p.add_argument('--dev',type=Path,required=True);p.add_argument('--audit-root',type=Path,required=True)
    a=p.parse_args();corpus=[json.loads(x) for x in a.dev.read_text().splitlines()]
    short=next(x for x in corpus if len(x['input_ids'])<a.threshold)
    triggered=[x for x in corpus if len(x['input_ids'])>=a.threshold]
    first=next(x for x in triggered if x['family']=='code')
    second=next(x for x in triggered if x['family']=='Chinese')
    protocol=dict(threshold=a.threshold,dev_sha256=hashlib.sha256(a.dev.read_bytes()).hexdigest(),
                  sequence=[short['id'],first['id'],first['id'],second['id'],short['id']],
                  heldout_used=False,cmmlu_used=False)
    (a.audit_root/'dynamic-dev-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    results.append(request(short,'no-trigger-initial'))
    assert not rows(a.audit_root/'generation-audit.jsonl'),'short request triggered update'
    results.append(request(first,'trigger-a-first'));after_first=rows(a.audit_root/'generation-audit.jsonl')
    first_layers=validate_events(after_first);assert all(len(x)==1 for x in first_layers.values())
    results.append(request(first,'trigger-a-repeat'));after_repeat=rows(a.audit_root/'generation-audit.jsonl')
    repeat_layers=validate_events(after_repeat);assert all(len(x)==2 for x in repeat_layers.values())
    assert results[-1]['output_ids']==results[-2]['output_ids'],'same-path output mismatch'
    assert all(v[0]['selected']==v[1]['selected'] for v in repeat_layers.values()),'selection mismatch'
    results.append(request(second,'trigger-b-shift'));after_shift=rows(a.audit_root/'generation-audit.jsonl')
    shift_layers=validate_events(after_shift);assert all(len(x)==3 for x in shift_layers.values())
    changed=[layer for layer,v in shift_layers.items() if v[1]['selected']!=v[2]['selected']]
    before_short=len(after_shift);results.append(request(short,'no-trigger-after-update'))
    assert len(rows(a.audit_root/'generation-audit.jsonl'))==before_short,'no-trigger mutated state'
    consumption=rows(a.audit_root/'generation-consumption.jsonl')
    assert consumption and all(x['weights']==x['mapping']==x['mask']==x['wrapper_mask']
                               for x in consumption)
    summary=dict(status='PASS' if changed else 'WORKLOAD_SHIFT_NOT_OBSERVED',threshold=a.threshold,
        no_trigger=True,trigger=True,weight_identity_exact=True,generation_consistency=True,
        deterministic_selection=True,changed_layers=changed,update_events=len(after_shift),
        migration_bytes=sum(x['migration_bytes'] for x in after_shift),
        diagnostic_update_seconds=sum(x['diagnostic_seconds'] for x in after_shift),
        requests=results,boundary='DEV_ONLY_SERIAL_TP1_BATCH1')
    (a.audit_root/'dynamic-dev-summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
