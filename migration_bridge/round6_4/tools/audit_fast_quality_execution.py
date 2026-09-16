"""Recheck raw requests, fixed inputs, scoring and provenance without model work."""
import argparse
import hashlib
import json
from pathlib import Path
from quality_answers_v2 import score
from run_compact_controls import validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','inputs','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_text())
    assert protocol['version']=='round6.4-fast-quality-v1'
    assert sha(a.inputs)==protocol['primary_input_sha256']
    rows=json.loads(a.inputs.read_text())['rows']
    report=json.loads((a.root/'quality-report.json').read_text())
    lifecycle=json.loads((a.root/'lifecycle.json').read_text())
    assert report['status']=='PRIMARY_EXECUTION_COMPLETE' and lifecycle['status']=='PLANNED_STOP'
    assert report['binding']==dict(protocol_sha256=sha(a.protocol),primary_sha256=sha(a.inputs))
    raw=[json.loads(s) for s in (a.root/'requests.jsonl').read_text().splitlines()]
    assert len(raw)==len(rows)==len(report['rows'])==288
    assert len({x['id'] for x in raw})==288
    for source,item,graded in zip(rows,raw,report['rows']):
        assert source['id']==item['id']==graded['id']
        payload,response=item['payload'],item['response']
        rid=f"r64-quality-{report['profile']}-{source['id']}"
        assert payload['rid']==rid and payload['input_ids']==source['input_ids']
        assert payload['sampling_params']==dict(temperature=0,max_new_tokens=source['max_new_tokens'],ignore_eos=False,stop=source['stop'])
        tokens=validate(response,rid,len(response['output_ids']))
        assert len(tokens)<=source['max_new_tokens']
        assert response['meta_info']['prompt_tokens']==len(source['input_ids'])
        finish=response['meta_info']['finish_reason']['type']
        assert finish in ('stop','length') and graded['finish_reason']==finish
        again=score(source['benchmark'],response['text'],source['answer'],truncated=finish=='length')
        assert all(graded[k]==v for k,v in again.items())
    actual=json.loads((a.root/'provenance.json').read_text())['file_sha256']
    assert all(actual.get(k)==v for k,v in protocol['actual_source_sha256'].items())
    result=dict(status='RAW_QUALITY_EXECUTION_AUDIT_PASS',profile=report['profile'],requests=288,
        input_sha256=sha(a.inputs),protocol_sha256=sha(a.protocol),raw_sha256=sha(a.root/'requests.jsonl'),
        report_sha256=sha(a.root/'quality-report.json'),quality_acceptance='separate_paired_screen_required')
    with a.output.open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
