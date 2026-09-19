"""Frozen Dynamic held-out/CMMLU client. Never selects thresholds or samples."""
import argparse
import hashlib
import json
from pathlib import Path
import time

CMMLU_SHA='365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d'
HOLDOUT_SHA='67432eca56ee9638dfe72e1d670ef1e30ce8833fe6b457dead6e12a806bac2d7'
ROOT=Path('/workspace/round8')


def quality_payload(row):
    return dict(rid='qwen-dynamic-'+row['id'],input_ids=row['input_ids'],
        sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=False,stop=row['stop']),
        return_logprob=True,logprob_start_len=-1,top_logprobs_num=0,
        token_ids_logprob=row['candidate_ids'])


def heldout_payload(row):
    return dict(rid='qwen-dynamic-'+row['id'],input_ids=row['input_ids'],
        sampling_params=dict(temperature=0,max_new_tokens=16,ignore_eos=True))


def validate_entry(entry):
    assert entry.get('system_gate')=='PASS'
    commit=entry.get('local_commit','');digest=entry.get('candidate_sha256','')
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)


def main():
    import requests
    from qwen_cmmlu_reference import score
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['heldout','cmmlu'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    entry=json.loads((a.output/'entry.json').read_text());validate_entry(entry)
    candidate=ROOT/'qwen-dynamic-candidate-v2/candidate.json'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==entry['candidate_sha256']
    contract=json.loads(candidate.read_text())
    assert contract['threshold']==256 and contract['chunked_prefill_size']==512
    assert hashlib.sha256((ROOT/'q32_profile.json').read_bytes()).hexdigest()==contract['q32_profile_sha256']
    manifest=ROOT/('qwen-cmmlu-reference-v1/manifest.json' if a.mode=='cmmlu' else 'qwen-scheduler-v1/SCHED_HOLDOUT.jsonl')
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest==(CMMLU_SHA if a.mode=='cmmlu' else HOLDOUT_SHA)
    if a.mode=='cmmlu':
        frozen=json.loads(manifest.read_text());rows=frozen['rows']
        for name in ('tokenizer','tokenizer_config'):
            assert hashlib.sha256((ROOT/f'model-sample/{name}.json').read_bytes()).hexdigest()==frozen[name+'_sha256']
    else:rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    assert len(rows)==(256 if a.mode=='cmmlu' else 128)
    assert len({r['id'] for r in rows})==len(rows)
    correct=0;results=[];triggered_requests=0
    with (a.output/'raw.jsonl').open('x') as raw:
        for index,row in enumerate(rows):
            payload=quality_payload(row) if a.mode=='cmmlu' else heldout_payload(row)
            server_log=a.output/'server.log'
            before_updates=server_log.read_text().count('expert update =')
            started=time.time()
            response=requests.post('http://127.0.0.1:18400/generate',json=payload,timeout=7200)
            data=response.json()
            updates=server_log.read_text().count('expert update =')-before_updates
            assert updates>=0 and updates%40==0,'INCOMPLETE_LAYER_UPDATE_TELEMETRY'
            triggered_requests+=int(updates>0)
            raw.write(json.dumps(dict(id=row['id'],payload=payload,http_status=response.status_code,
                response=data,start_unix=started,end_unix=time.time(),observed_layer_updates=updates),ensure_ascii=False)+'\n');raw.flush()
            response.raise_for_status();assert data['meta_info']['id']==payload['rid']
            if a.mode=='cmmlu':
                positions=data['meta_info']['output_token_ids_logprobs'];assert len(positions)==1
                prediction=score(positions[0],row['candidate_ids'])
                correct+=int(prediction==row['answer']);results.append(prediction)
            else:assert len(data['output_ids'])==16
            (a.output/'progress.json').write_text(json.dumps(dict(completed=index+1,total=len(rows),
                correct=correct if a.mode=='cmmlu' else None,invalid=0,updated=time.time())))
            print(f'{a.mode} {index+1}/{len(rows)}',flush=True)
    summary=dict(status='COLLECTED_PENDING_INDEPENDENT_AUDIT',mode=a.mode,count=len(rows),
        invalid=0,manifest_sha256=digest,entry=entry,formal_performance=False,
        triggered_requests=triggered_requests,trigger_rate=triggered_requests/len(rows))
    if a.mode=='cmmlu':summary.update(correct=correct,reference_correct=222,
        observed_delta_pp=(correct-222)/256*100,quality_gate='PASS' if correct>=210 else 'FAIL',
        answer_distribution={c:results.count(c) for c in 'ABCD'})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
