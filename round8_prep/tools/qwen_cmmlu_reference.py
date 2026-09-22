"""Freeze then evaluate the unchanged256 items with the frozen Qwen tokenizer.
Q64 is a new engineering reference, never the previous DSV2 score/oracle.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCE_SHA='bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04'
ROOT=Path('/workspace/round8')
OUT=ROOT/'qwen-cmmlu-reference-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def score(candidates,ids):
 assert len(candidates)==4 and len(ids)==4 and len(set(ids))==4
 by_id={v[1]:v[0] for v in candidates};assert set(by_id)==set(ids)
 values=[by_id[k] for k in ids];assert all(math.isfinite(v) for v in values)
 return 'ABCD'[max(range(4),key=lambda j:values[j])]

def freeze():
 from transformers import AutoTokenizer
 source=Path('/workspace/results/r65-r67-serial-v5/cmmlu.json')
 assert sha(source)==SOURCE_SHA
 rows=json.loads(source.read_text())['rows'];assert len(rows)==256 and len({r['id'] for r in rows})==256
 t=AutoTokenizer.from_pretrained(ROOT/'model-sample',local_files_only=True)
 choices=[t.encode(s,add_special_tokens=False) for s in 'ABCD']
 assert all(len(c)==1 for c in choices);candidates=[c[0] for c in choices];assert len(set(candidates))==4
 frozen=[]
 for r in rows:
  rendered=t.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=False,
      add_generation_prompt=True,enable_thinking=False)
  ids=t.encode(rendered,add_special_tokens=False);assert 0<len(ids)<4096
  frozen.append(dict(id=r['id'],subject=r['subject'],answer=r['answer'],prompt=r['prompt'],
    rendered_prompt=rendered,input_ids=ids,candidate_ids=candidates,stop=r['stop']))
 OUT.mkdir(exist_ok=False)
 manifest=dict(version='qwen35-cmmlu-reference-v1',source_sha256=SOURCE_SHA,
  model_revision='59d61f3ce65a6d9863b86d2e96597125219dc754',rows=frozen,
  tokenizer_sha256=sha(ROOT/'model-sample/tokenizer.json'),
  tokenizer_config_sha256=sha(ROOT/'model-sample/tokenizer_config.json'),
  protocol=dict(item_selection='UNCHANGED_256_ORDERED_ITEMS_AND_FEWSHOT_TEXT',
   wrapper='FROZEN_QWEN_CHAT_TEMPLATE',thinking=False,temperature=0,max_new_tokens=1,
   scoring='ARGMAX_A_B_C_D_FIRST_OUTPUT_POSITION_LOGPROBS',tie='A_B_C_D_ORDER',
   reference='FRESH_Q64_C64_NOT_ALL_NPU',q32_observed_delta_min_pp=-5,invalid_required=0))
 write(OUT/'manifest.json',manifest)
 print(json.dumps(dict(manifest_sha256=sha(OUT/'manifest.json'),count=256,
   min_tokens=min(len(r['input_ids']) for r in frozen),max_tokens=max(len(r['input_ids']) for r in frozen))),flush=True)

def run():
 import requests
 assert json.loads((ROOT/'q64-generation-screen-v1/summary.json').read_text())['status']=='PASS'
 for layer in (0,20,39):
  assert json.loads((ROOT/f'three-way-grouped-L{layer}-v2/result.json').read_text())['status']=='PASS'
 ack=json.loads((OUT/'freeze-ack.json').read_text())
 assert ack['manifest_sha256']==sha(OUT/'manifest.json') and len(ack['local_commit'])==40
 manifest=json.loads((OUT/'manifest.json').read_text())
 assert manifest['source_sha256']==SOURCE_SHA
 assert sha(ROOT/'model-sample/tokenizer.json')==manifest['tokenizer_sha256']
 raw=OUT/'raw.jsonl';assert not raw.exists(),'DO_NOT_OVERWRITE_QUALITY'
 correct=0;predictions=[]
 with raw.open('x') as f:
  for i,r in enumerate(manifest['rows']):
   payload=dict(rid='qwen-q64-'+r['id'],input_ids=r['input_ids'],sampling_params=dict(
    temperature=0,max_new_tokens=1,ignore_eos=False,stop=r['stop']),return_logprob=True,
    logprob_start_len=-1,top_logprobs_num=0,token_ids_logprob=r['candidate_ids'])
   response=requests.post('http://127.0.0.1:18400/generate',json=payload,timeout=3600)
   data=response.json();record=dict(id=r['id'],payload=payload,http_status=response.status_code,response=data)
   f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();response.raise_for_status()
   assert data['meta_info']['id']==payload['rid']
   positions=data['meta_info']['output_token_ids_logprobs'];assert len(positions)==1
   predicted=score(positions[0],r['candidate_ids'])
   correct+=int(predicted==r['answer']);predictions.append(predicted)
   write(OUT/'progress.json',dict(completed=i+1,total=256,correct=correct,invalid=0))
   print(f'{i+1}/256 correct={correct}',flush=True)
 write(OUT/'summary.json',dict(status='REFERENCE_SCORED_PENDING_AUDIT',correct=correct,count=256,
  invalid=0,manifest_sha256=sha(OUT/'manifest.json'),freeze_ack=ack,
  answer_distribution={c:predictions.count(c) for c in 'ABCD'},all_npu_reference=False,
  quality_comparison_to_q32='NOT_RUN',formal_performance=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);a=p.parse_args()
 freeze() if a.stage=='freeze' else run()
