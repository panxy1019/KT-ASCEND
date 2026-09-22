"""Continuous decode capture with audited history and explicit lookahead.

Uses existing debug-only whole-forward capture, without D1-specific route gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
from static_session import session


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--force',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    inputs=json.loads(a.inputs.read_text())
    forced=json.loads(a.force.read_text()) if a.force else None
    if forced and [r['rid'] for r in forced['requests']]!=[r['rid'] for r in inputs['requests']]:
        raise ValueError('FORCE_INPUT_ORDER')
    refs={r['rid']:r for r in forced['requests']} if forced else {}
    report=dict(status='RUNNING',rows=[],forward_count=0,
                inputs_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),forced=bool(forced))
    try:
        with session(a.output,a.profile,capture=True,force=a.force) as root:
            with (root/'index.jsonl').open() as stream:
                for r in inputs['requests']:
                    rid=r['rid']; prompt=r['input_ids'];budget=inputs['token_budget']
                    payload=dict(rid=rid,input_ids=prompt,sampling_params=dict(
                        temperature=0,max_new_tokens=budget,ignore_eos=True),return_logprob=True,
                        logprob_start_len=-1,top_logprobs_num=0)
                    req=urllib.request.Request('http://127.0.0.1:31000/generate',data=json.dumps(payload).encode(),
                                               headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=900) as response: answer=json.load(response)
                    (root/(rid+'.response.json')).write_text(json.dumps(dict(payload=payload,response=answer))+'\n')
                    tokens=answer['output_ids']
                    if answer['meta_info']['id']!=rid or len(tokens)!=budget:raise ValueError('RESPONSE_ID_BUDGET')
                    if forced and (tokens!=refs[rid]['baseline_output_ids'] or prompt!=refs[rid]['input_ids']):
                        raise ValueError('FORCED_HISTORY_DRIFT')
                    vectors=[]; steps=[]
                    for step in range(budget+1):
                        deadline=time.monotonic()+30
                        while True:
                            offset=stream.tell(); line=stream.readline()
                            if line.endswith('\n'):break
                            stream.seek(offset)
                            if time.monotonic()>deadline:raise TimeoutError('CAPTURE_COMMIT')
                            time.sleep(.05)
                        entry=json.loads(line)
                        if entry['sequence']!=report['forward_count'] or entry['batch']['request_ids']!=[rid]:
                            raise ValueError('CAPTURE_SEQUENCE_RID')
                        data=torch.load(entry['file'],map_location='cpu',weights_only=False,mmap=True)
                        tx=data['capture.transaction']
                        if tx['forward_sequence_id']!=entry['sequence'] or tx['pid']!=entry['pid']:
                            raise ValueError('TRANSACTION_JOIN')
                        for key,sha in tx['tensor_byte_sha256'].items():
                            if hashlib.sha256(data[key].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()!=sha:
                                raise ValueError('CAPTURE_BYTES')
                        ids=prompt if step==0 else [tokens[step-1]]
                        pos=list(range(len(prompt))) if step==0 else [len(prompt)+step-1]
                        if data['capture.forward_batch_info.input_ids'].tolist()!=ids or data['capture.forward_batch_info.positions'].tolist()!=pos:
                            raise ValueError('ACTUAL_HISTORY')
                        if data['capture.forward_batch_info.seq_lens'].tolist()!=[len(prompt)+step]:raise ValueError('KV_LENGTH')
                        z=data['capture.next_token_logits']
                        if tuple(z.shape)!=(1,102400) or not torch.isfinite(z).all():raise ValueError('LOGIT_FINITE_SHAPE')
                        if step<budget and not forced and z[0,tokens[step]]!=z.max():raise ValueError('NATURAL_TOP1')
                        if step<budget:vectors.append(z.clone())
                        steps.append(dict(step=step,target=step<budget,file=entry['file'],
                                          logits_sha256=tx['tensor_byte_sha256']['capture.next_token_logits']))
                        report['forward_count']+=1
                    torch.save(torch.cat(vectors),root/(rid+'.logits.pt'))
                    report['rows'].append(dict(rid=rid,input_ids=prompt,baseline_output_ids=tokens,steps=steps))
                    (root/'capture-progress.json').write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(dict(rid=rid,forwards=len(steps),integrity='PASS')),flush=True)
            report['status']='CAPTURE_INTEGRITY_PASS'
        # Audit after owned server stop so late/extra forwards cannot escape.
        entries=(a.output/'index.jsonl').read_text().splitlines()
        physical=list((a.output/'tensors').rglob('*.pt'))
        if len(entries)!=report['forward_count'] or len(physical)!=len(entries):raise ValueError('EXTRA_FORWARD')
        if list((a.output/'tensors').rglob('*.incomplete')):raise ValueError('INCOMPLETE_CAPTURE')
        if forced:
            trace=[json.loads(x) for x in (a.output/'forcing.jsonl').read_text().splitlines()]
            if len(trace)!=len(entries):raise ValueError('FORCING_TRACE_COUNT')
            expected={(r['rid'],s['step']) for r in report['rows'] for s in r['steps']}
            if {(r['rid'],r['step']) for r in trace}!=expected:raise ValueError('FORCING_TRACE_COVERAGE')
            for row in trace:
                r=next(r for r in report['rows'] if r['rid']==row['rid'])
                if row['sampled_logits_sha256']!=r['steps'][row['step']]['logits_sha256']:
                    raise ValueError('SAMPLED_LOGIT_IDENTITY')
        else:
            manifest=dict(version='r63-forced-history-v1',session_id=a.output.name,
                          requests=[{k:r[k] for k in ('rid','input_ids','baseline_output_ids')} for r in report['rows']])
            (a.output/'force.json').write_text(json.dumps(manifest,indent=2)+'\n')
    except BaseException as exc:
        report.update(status='FAILED',exception=repr(exc));raise
    finally:
        if a.output.exists():(a.output/'capture-result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
