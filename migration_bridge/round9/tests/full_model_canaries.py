"""Fixed small known-answer sanity checks before scoring a fresh reference."""
import argparse,json,math,pathlib,urllib.request
from transformers import AutoTokenizer
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
t=AutoTokenizer.from_pretrained('/models/de4b8e4d43b917e7706784d8bb445c9af86a3540',local_files_only=True)
cases=[('multiply','Calculate 17 * 23. Reply with only the integer.','391'),('capital','中国的首都是哪座城市？只回答城市名。','北京'),('echo','Reply with exactly this word and nothing else: ORANGE','ORANGE'),('json','Return only a JSON object with keys name and count, values apple and 3.',{'name':'apple','count':3})]
report={'status':'RUNNING','scope':'FIXED_KNOWN_ANSWER_CANARIES_NOT_BENCHMARK_SCORE','rows':[]}
try:
 for name,prompt,expected in cases:
  ids=t.encode(t.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
  body={'rid':'r9-canary-'+name,'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':32},'return_logprob':True,'logprob_start_len':-1}
  req=urllib.request.Request('http://127.0.0.1:30091/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=300) as response:data=json.load(response)
  text=data['text'].strip();value=json.loads(text) if name=='json' else text
  row=dict(name=name,expected=expected,request=body,response=data,passed=value==expected);report['rows'].append(row)
  assert all(math.isfinite(v[0]) for v in data['meta_info']['output_token_logprobs'])
  assert row['passed'],f'{name}: {text!r} != {expected!r}'
 report['status']='PASS_FOUR_KNOWN_ANSWER_CANARIES'
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],flush=True)
