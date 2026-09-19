"""Independent complete-input/score/update audit; prints a release proposal only."""
import argparse
import hashlib
import json
from pathlib import Path
from qwen_dynamic_evaluation import quality_payload, heldout_payload, validate_entry, CMMLU_SHA, HOLDOUT_SHA
from qwen_cmmlu_reference import score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]

def check_raw_rows(actual,expected):
    assert len(actual)==len(expected)
    assert [r['id'] for r in actual]==[r['id'] for r in expected]
    assert len({r['id'] for r in actual})==len(actual)
    for record,item in zip(actual,expected):
        assert record['http_status']==200
        assert record['payload']['input_ids']==item['input_ids']

def audit(root,mode):
    out=root/f'qwen-dynamic-{mode}-v2'
    assert json.loads((out/'state.json').read_text())['status']=='COLLECTED_PENDING_INDEPENDENT_AUDIT'
    summary=json.loads((out/'summary.json').read_text());validate_entry(summary['entry'])
    assert summary['entry']['candidate_sha256']==sha(root/'qwen-dynamic-candidate-v2/candidate.json')
    manifest=root/('qwen-cmmlu-reference-v1/manifest.json' if mode=='cmmlu' else 'qwen-scheduler-v1/SCHED_HOLDOUT.jsonl')
    assert sha(manifest)==(CMMLU_SHA if mode=='cmmlu' else HOLDOUT_SHA)
    expected=json.loads(manifest.read_text())['rows'] if mode=='cmmlu' else rows(manifest)
    actual=rows(out/'raw.jsonl');check_raw_rows(actual,expected)
    correct=0
    for record,item in zip(actual,expected):
        assert record['payload']==(quality_payload(item) if mode=='cmmlu' else heldout_payload(item))
        data=record['response'];assert data['meta_info']['id']==record['payload']['rid']
        if mode=='cmmlu':
            positions=data['meta_info']['output_token_ids_logprobs'];assert len(positions)==1
            correct+=score(positions[0],item['candidate_ids'])==item['answer']
        else:assert len(data['output_ids'])==16
    log=(out/'server.log').read_text();assert 'Check input storage_offset' not in log
    assert 'Traceback (most recent call last)' not in log
    resources=rows(out/'resources.jsonl');assert resources
    assert all(r['swap']==0 and not r['state']['OOMKilled'] for r in resources)
    assert all(not int(line.split()[1]) for r in resources for line in r['memory_events'].splitlines()
               if line.split()[0] in ('oom','oom_kill'))
    assert all(r['observed_layer_updates']>=0 and r['observed_layer_updates']%40==0 for r in actual)
    assert sum(r['observed_layer_updates']>0 for r in actual)==summary['triggered_requests']
    report=dict(status='PASS',count=len(actual),summary_sha256=sha(out/'summary.json'),
        raw_sha256=sha(out/'raw.jsonl'),resources_sha256=sha(out/'resources.jsonl'))
    if mode=='cmmlu':
        assert correct==summary['correct'] and summary['invalid']==0
        report.update(correct=correct,observed_delta_pp=(correct-222)/256*100,status='PASS' if correct>=210 else 'FAIL')
    else:
        events=rows(out/'generation-audit.jsonl');assert events
        groups={layer:[e for e in events if e['layer']==layer] for layer in range(40)}
        assert len({len(v) for v in groups.values()})==1 and all(groups.values())
        for layer,values in groups.items():
            for generation,event in enumerate(values,1):
                assert event['weights']==event['mapping']==event['mask']==event['wrapper_mask']==generation
                assert event['weight_identity_exact'] and event['mask_mapping_wrapper_exact']
        consumption=rows(out/'generation-consumption.jsonl');assert consumption
        assert all(e['weights']==e['mapping']==e['mask']==e['wrapper_mask'] for e in consumption)
        report.update(update_events=len(events),generations_per_layer=len(groups[0]))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result={mode:audit(a.root,mode) for mode in ('heldout','cmmlu')}
    result['release_proposal']={mode:result[mode]['status'] for mode in result}
    for mode in ('heldout','cmmlu'):result['release_proposal'][mode+'_summary_sha256']=result[mode]['summary_sha256']
    print(json.dumps(result,indent=2))
