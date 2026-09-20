"""Candidate full-model system checks; no accuracy or performance qualification."""
import argparse,json,math,pathlib,time,urllib.request
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--placement',choices=['static','request_boundary'],required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
t=AutoTokenizer.from_pretrained('/models/de4b8e4d43b917e7706784d8bb445c9af86a3540',local_files_only=True)
prompts=['用两句话解释为什么天空通常呈蓝色。','Explain the difference between a list and a set.','Compute 17 times 23.','Return only a JSON object with name apple and count 3.']
report=dict(status='RUNNING',scope='FULL_MODEL_SYSTEM_ONLY_NOT_QUALITY_OR_PERFORMANCE',placement=a.placement,rows=[])
try:
 for index,prompt in enumerate(prompts):
  ids=t.encode(t.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
  for repeat in range(2):
   body=dict(rid=f'r9-candidate-{index}-{repeat}',input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=32),return_logprob=True,logprob_start_len=-1)
   req=urllib.request.Request('http://127.0.0.1:30091/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=900) as f:data=json.load(f)
   logs=data['meta_info']['output_token_logprobs'];output=[item[1] for item in logs]
   if not output or not data['text'].strip() or not all(math.isfinite(x[0]) for x in logs):raise RuntimeError('empty or nonfinite generation')
   if repeat and a.placement=='static' and output!=previous:raise RuntimeError('static trajectory changed on repeat')
   previous=output;report['rows'].append(dict(request=body,response=data));print(index,repeat,'PASS',flush=True)
 if a.placement=='request_boundary':
  events=[json.loads(line) for line in pathlib.Path('/out/dynamic-events.jsonl').read_text().splitlines()]
  completions={e.get('request_id'):e for e in events if 'request_id' in e}
  # Scheduler publishes the response before reaching its post-release hook.
  for _ in range(100):
   if all(r['request']['rid'] in completions for r in report['rows']):break
   time.sleep(.1);events=[json.loads(line) for line in pathlib.Path('/out/dynamic-events.jsonl').read_text().splitlines()];completions={e.get('request_id'):e for e in events if 'request_id' in e}
  if not all(r['request']['rid'] in completions for r in report['rows']):raise RuntimeError('missing drained Dynamic request boundary')
  report['dynamic_events']=events
 report['status']='PASS_FULL_MODEL_SYSTEM8'
except BaseException as exc:report.update(status='FAIL',error=repr(exc));raise
finally:a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],flush=True)
