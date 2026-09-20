"""Fresh Round9 compact32 and repeat/restart trajectories, no speed claim."""
import argparse,hashlib,json,math,pathlib,time,urllib.request
from transformers import AutoTokenizer
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--baseline',type=pathlib.Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
revision='de4b8e4d43b917e7706784d8bb445c9af86a3540'
model=pathlib.Path('/models')/revision
prompts={'zh':'用两句话解释为什么天空通常呈蓝色。','en':'Explain the difference between a list and a set in two sentences.','math':'Compute 17 times 23 and briefly show the calculation.','json':'Return only a JSON object with keys name and count, values apple and 3.'}
t=AutoTokenizer.from_pretrained(model,local_files_only=True)
rows={k:t.encode(t.apply_chat_template([{'role':'user','content':v}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for k,v in prompts.items()}
protocol={'revision':revision,'prompts':prompts,'input_ids':rows,'temperature':0,'thinking':False,'budgets':[32,16,16,16,16,16,16,8],'scope':'SYSTEM_COMPACT_NOT_QUALITY_OR_PERFORMANCE','tokenizer_sha256':hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest(),'tokenizer_config_sha256':hashlib.sha256((model/'tokenizer_config.json').read_bytes()).hexdigest()}
encoded=json.dumps(protocol,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
(a.output/'protocol.json').write_text(encoded)
base={};budgets=protocol['budgets']
if a.baseline:
 assert (a.baseline/'protocol.json').read_text()==encoded
 assert json.loads((a.baseline/'summary.json').read_text())['status']=='PASS_COMPACT32'
 previous=[json.loads(l) for l in (a.baseline/'raw.jsonl').read_text().splitlines()]
 base={x['prompt_id']:x['output_ids'] for x in previous if x['budget']==32};budgets=[32]
count=0
summary={'status':'RUNNING','scope':protocol['scope'],'protocol_sha256':hashlib.sha256(encoded.encode()).hexdigest()}
try:
 with (a.output/'raw.jsonl').open('x') as f:
  for key,ids in rows.items():
   for budget in budgets:
    body={'rid':f'r9-{a.output.name}-{key}-{count}','input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':budget},'return_logprob':True,'logprob_start_len':-1}
    req=urllib.request.Request('http://127.0.0.1:30091/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    started=time.time()
    with urllib.request.urlopen(req,timeout=900) as response:data=json.load(response)
    logs=data['meta_info']['output_token_logprobs'];output_ids=data.get('output_ids',[v[1] for v in logs])
    row={'prompt_id':key,'budget':budget,'output_ids':output_ids,'request':body,'response':data,'started_at':started,'ended_at':time.time()}
    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    assert output_ids and data['text'].strip()
    assert len(logs)==len(output_ids) and all(math.isfinite(v[0]) for v in logs)
    if key not in base:base[key]=output_ids
    else:assert output_ids==base[key][:budget],f'TRAJECTORY_MISMATCH:{key}:{budget}'
    count+=1;print(f'{count} {key} {budget} PASS',flush=True)
 summary.update(status='PASS_RESTART4' if a.baseline else 'PASS_COMPACT32',requests=count)
except BaseException as exc:
 summary.update(status='FAIL',requests=count,error=repr(exc));raise
finally:(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
