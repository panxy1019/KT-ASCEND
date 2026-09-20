"""Fresh original-weight Round9 reference on the unchanged frozen256 items."""
import argparse,hashlib,json,math,pathlib,time,urllib.request
REVISION='de4b8e4d43b917e7706784d8bb445c9af86a3540'
SOURCE_SHA='bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run']);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--source',type=pathlib.Path);p.add_argument('--compact',type=pathlib.Path);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=256);a=p.parse_args()
model=pathlib.Path('/models')/REVISION
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
if a.stage=='freeze':
 from transformers import AutoTokenizer
 assert sha(a.source)==SOURCE_SHA
 source=json.loads(a.source.read_text())['rows'];assert len(source)==256 and len({r['id'] for r in source})==256
 t=AutoTokenizer.from_pretrained(model,local_files_only=True)
 candidates=[t.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(v)==1 for v in candidates);candidates=[v[0] for v in candidates];assert len(set(candidates))==4
 rows=[]
 for r in source:
  ids=t.encode(t.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
  assert 0<len(ids)<4096
  rows.append(dict(id=r['id'],subject=r['subject'],answer=r['answer'],prompt=r['prompt'],input_ids=ids,candidate_ids=candidates,stop=r['stop']))
 a.output.mkdir(parents=True,exist_ok=False)
 write(a.output/'manifest.json',dict(revision=REVISION,source_sha256=SOURCE_SHA,tokenizer_sha256=sha(model/'tokenizer.json'),tokenizer_config_sha256=sha(model/'tokenizer_config.json'),rows=rows,protocol=dict(item_selection='UNCHANGED256_ORDERED_ITEMS_AND_FEWSHOT_TEXT',wrapper='ORIGINAL_REVISION_CHAT_TEMPLATE',thinking=False,temperature=0,max_new_tokens=1,scoring='ARGMAX_ABCD_FIRST_OUTPUT_LOGPROBS',tie='ABCD_ORDER',reference='FRESH_Q64_BF16_STORAGE_F32_CPU_C64_NOT_ALL_NPU')))
 print(json.dumps(dict(status='FROZEN_NOT_SCORED',manifest_sha256=sha(a.output/'manifest.json'),count=256,min_tokens=min(len(r['input_ids']) for r in rows),max_tokens=max(len(r['input_ids']) for r in rows))),flush=True)
else:
 assert json.loads((a.compact/'summary.json').read_text())['status']=='PASS_COMPACT32'
 ack=json.loads((a.output/'freeze-ack.json').read_text());assert ack['manifest_sha256']==sha(a.output/'manifest.json') and len(ack['local_commit'])==40
 manifest=json.loads((a.output/'manifest.json').read_text());assert manifest['revision']==REVISION and manifest['source_sha256']==SOURCE_SHA
 assert sha(model/'tokenizer.json')==manifest['tokenizer_sha256'] and sha(model/'tokenizer_config.json')==manifest['tokenizer_config_sha256']
 assert 0 <= a.start < 256 and 0 < a.count <= 256-a.start
 admission=json.loads((a.output.parent/'preflight/admission.json').read_text())
 container=json.loads((a.output.parent/'container-created.json').read_text())
 assert 'ROUND9_MEMORY_OBSERVER=0' in container['Config']['Env']
 runtime_identity=dict(implementation_sha256=admission['implementation_sha256'],profile_sha256=admission['profile_sha256'],image=container['Image'],observer=False)
 correct=0;predictions=[];summary={'status':'RUNNING','manifest_sha256':sha(a.output/'manifest.json'),'freeze_ack':ack,'start':a.start,'count':a.count,'driver_sha256':sha(pathlib.Path(__file__)),'runtime_identity':runtime_identity}
 try:
  with (a.output/'raw.jsonl').open('x') as f:
   for i,r in enumerate(manifest['rows'][a.start:a.start+a.count]):
    body=dict(rid='r9-cmmlu-'+r['id'],input_ids=r['input_ids'],sampling_params=dict(temperature=0,max_new_tokens=1,stop=r['stop']),return_logprob=True,logprob_start_len=-1,top_logprobs_num=0,token_ids_logprob=r['candidate_ids'])
    req=urllib.request.Request('http://127.0.0.1:30091/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as response:data=json.load(response)
    f.write(json.dumps(dict(id=r['id'],request=body,response=data),ensure_ascii=False)+'\n');f.flush()
    assert data['meta_info']['id']==body['rid'];positions=data['meta_info']['output_token_ids_logprobs'];assert len(positions)==1
    values={v[1]:v[0] for v in positions[0]};assert set(values)==set(r['candidate_ids']) and all(math.isfinite(v) for v in values.values())
    predicted='ABCD'[max(range(4),key=lambda j:values[r['candidate_ids'][j]])]
    predictions.append(predicted);correct+=int(predicted==r['answer'])
    write(a.output/'progress.json',dict(completed=i+1,total=a.count,start=a.start,correct=correct,invalid=0));print(f'{a.start+i+1}/256 correct_in_slice={correct}',flush=True)
  summary.update(status='REFERENCE_SLICE_SCORED_PENDING_AUDIT',correct=correct,count=a.count,invalid=0,answer_distribution={c:predictions.count(c) for c in 'ABCD'},all_npu_reference=False)
 except BaseException as exc:
  summary.update(status='FAIL',completed=len(predictions),error=repr(exc));raise
 finally:write(a.output/'summary.json',summary)
