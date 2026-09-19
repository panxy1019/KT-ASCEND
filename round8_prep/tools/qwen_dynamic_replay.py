"""Hook-free fresh-process DEV replay against audited transfer-v2 outputs."""
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path('/workspace/round8')
source=ROOT/'q32-dynamic-transfer-v2'
out=ROOT/'q32-dynamic-restart-v2'
protocol=json.loads((source/'dynamic-dev-protocol.json').read_text())
assert protocol['heldout_used'] is False and protocol['cmmlu_used'] is False
corpus=ROOT/'qwen-scheduler-v1/SCHED_DEV.jsonl'
assert hashlib.sha256(corpus.read_bytes()).hexdigest()==protocol['dev_sha256']
by_id={r['id']:r for r in map(json.loads,corpus.read_text().splitlines())}
reference=json.loads((source/'dynamic-dev-summary.json').read_text())
assert reference['status']=='PASS'
with (out/'raw.jsonl').open('x') as raw:
    for index,identity in enumerate(protocol['sequence']):
        item=by_id[identity]
        payload=dict(input_ids=item['input_ids'],sampling_params=dict(
            temperature=0,max_new_tokens=16,ignore_eos=True))
        response=requests.post('http://127.0.0.1:18400/generate',json=payload,timeout=3600)
        data=response.json();raw.write(json.dumps(dict(index=index,id=identity,payload=payload,
            http_status=response.status_code,response=data))+'\n');raw.flush()
        response.raise_for_status()
        assert data['output_ids']==reference['requests'][index]['output_ids'],f'RESTART_OBSERVER_PARITY_{index}'
        print(f'{index+1}/5 exact',flush=True)
(out/'summary.json').write_text(json.dumps(dict(status='PASS',requests=5,
    clean_restart_token_exact=True,observer_on_off_token_exact=True,
    selection_replay_independently_observed=False,heldout_used=False,
    source_summary_sha256=hashlib.sha256((source/'dynamic-dev-summary.json').read_bytes()).hexdigest()),indent=2))
