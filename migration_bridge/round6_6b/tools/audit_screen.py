"""Independent raw-response rescoring; unchanged predeclared pass rule."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert hashlib.sha256(a.inputs.read_bytes()).hexdigest()=='885ccf76386dcfed2867142417300ac69c4401754360f92e6b6e4f179adffd97'
    inputs=json.loads(a.inputs.read_text());sequence=json.loads((a.root/'progress.json').read_text())
    assert sequence['status'] in ('SCREEN_PASS_AWAIT_AUDIT','SCREEN_FAILED_NO_RESEED')
    sys.path.insert(0,'/workspace/r66-v6/tools')
    from quality_answers_v2 import score
    from paired_quality_interval import bounds
    audited={};hashes={}
    for mode in ('npu','hybrid'):
        report=json.loads((a.root/mode/'quality-report.json').read_text())
        requests=[json.loads(line) for line in (a.root/mode/'requests.jsonl').read_text().splitlines()]
        assert len(requests)==len(report['rows'])==len(inputs['rows'])==288
        assert json.loads((a.root/mode/'lifecycle.json').read_text())['status']=='COMPLETE'
        rows=[]
        for source,raw,claimed in zip(inputs['rows'],requests,report['rows']):
            assert source['id']==raw['id']==claimed['id']
            payload=raw['payload'];answer=raw['response'];meta=answer['meta_info']
            assert payload['input_ids']==source['input_ids']
            assert payload['sampling_params']==dict(temperature=0,max_new_tokens=source['max_new_tokens'],ignore_eos=False,stop=source['stop'])
            assert meta['prompt_tokens']==len(source['input_ids'])
            lp=meta['output_token_logprobs'];assert [x[1] for x in lp]==answer['output_ids']
            assert all(math.isfinite(x[0]) for x in lp)
            if source['benchmark']=='cmmlu':
                assert payload['token_ids_logprob']==source['candidate_ids']
                scores=meta['output_token_ids_logprobs'][0]
                table={entry[1]:entry[0] for entry in scores}
                assert len(scores)==len(table)==4 and set(table)==set(source['candidate_ids'])
                assert all(math.isfinite(v) for v in table.values())
                maximum=max(table.values())
                # Independently select first label whose score attains the max.
                prediction=next(c for c,t in zip('ABCD',source['candidate_ids']) if table[t]==maximum)
                result=dict(correct=prediction==source['answer'],invalid=False,truncated=False)
            else:result=score('gsm8k',answer['text'],source['answer'],truncated=meta['finish_reason']['type']=='length')
            assert all(result[k]==claimed[k] for k in ('correct','invalid','truncated'))
            rows.append(dict(id=source['id'],benchmark=source['benchmark'],**result))
        audited[mode]=rows
        hashes[mode]={name:hashlib.sha256((a.root/mode/name).read_bytes()).hexdigest()
                      for name in ('requests.jsonl','quality-report.json','lifecycle.json','resolved_command.json')
                      if (a.root/mode/name).exists()}
    reports=[]
    for r in sequence['benchmarks']:
        pairs=[(n,h) for n,h in zip(audited['npu'],audited['hybrid']) if n['benchmark']==r['benchmark']]
        gains=sum(not n['correct'] and h['correct'] for n,h in pairs)
        losses=sum(n['correct'] and not h['correct'] for n,h in pairs)
        assert (gains,losses)==(r['gains'],r['losses'])
        reports.append(dict(**r,paired_marginal_interval=bounds(len(pairs),gains,losses)))
    output=dict(status='RAW_PAIRED_AUDIT_PASS',sequence_status=sequence['status'],benchmarks=reports,
                raw_sha256=hashes,pairs=288,population_noninferiority_claim=False,
                historical_v1='BLOCKED_UNCHANGED',all_moe_qualification=False)
    with a.output.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
