"""Exactly one bounded HTTP request; no retry, no hidden warmup."""
import argparse,json,pathlib,time,urllib.request,hashlib
p=argparse.ArgumentParser();p.add_argument('--rid',required=True);p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--kind',choices=['base','chunk','decode'],required=True);p.add_argument('--max-new-tokens',type=int,default=1)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
if not 1<=a.max_new_tokens<=8:raise ValueError('output budget')
if a.kind=='decode' and a.max_new_tokens<2:raise ValueError('decode requires at least2 requested output tokens; actual EOS is preserved')
body=dict(rid=a.rid,stream=True,sampling_params=dict(temperature=0,max_new_tokens=a.max_new_tokens,ignore_eos=False))
if a.kind=='chunk':
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/models/de4b8e4d43b917e7706784d8bb445c9af86a3540',local_files_only=True)
 ids=tokenizer.encode('Repeat these words in order: '+ ' red blue green yellow'*24,add_special_tokens=False)[:64]
 if len(ids)!=64:raise RuntimeError('fixture did not form64 tokens')
 body['input_ids']=ids
else:body['text']='Answer with exactly OK.' if a.kind=='base' else 'Continue the sequence: one, two, three,'
report=dict(status='RUNNING',body=body,events=[],started_unix=time.time(),clock='CLIENT_MONOTONIC')
started=time.monotonic();first=None;token_times=[];last_count=0
try:
 req=urllib.request.Request('http://127.0.0.1:30091/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=900) as response:
  for line in response:
   if not line.startswith(b'data:'):continue
   raw=line[5:].strip()
   if raw==b'[DONE]':break
   event=json.loads(raw);now=time.monotonic()-started
   count=event.get('meta_info',{}).get('completion_tokens',0)
   if count>last_count:
    if first is None:first=now
    token_times.append(dict(cumulative_tokens=count,client_seconds=now));last_count=count
   report['events'].append(dict(client_seconds=now,response=event))
 final=report['events'][-1]['response']
 meta=final['meta_info'];finish=meta.get('finish_reason')
 if not finish or finish.get('type')=='abort':raise RuntimeError('no normal terminal')
 if a.kind=='chunk' and meta['prompt_tokens']!=64:raise RuntimeError('not exactly64 input tokens')
 if a.kind=='decode' and meta['prompt_tokens']>32:raise RuntimeError('decode input budget exceeded')
 report.update(status='NORMAL_HTTP_COMPLETE_NOT_QUALITY_QUALIFIED',actual_prompt_tokens=meta['prompt_tokens'],actual_output_tokens=meta['completion_tokens'],output_ids=final.get('output_ids'),final_text=final.get('text'),finish_reason=finish)
 if a.kind=='decode':
  capture=pathlib.Path('/out/expert-replay-decode.json')
  if meta['completion_tokens']<2:
   report['status']='NORMAL_HTTP_COMPLETE_NO_DECODE_EVIDENCE'
  elif a.rid=='r9-mainline-capture':
   if not capture.exists() or json.loads(capture.read_text()).get('request_id')!=a.rid:
    raise RuntimeError('real decode replay capture not observed')
   report['decode_capture_manifest']=json.loads(capture.read_text())
except BaseException as e:report.update(status='FAIL',error=repr(e));raise
finally:
 report.update(client_wall_seconds=time.monotonic()-started,client_ttft_seconds=first,token_arrival_samples=token_times)
 a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(report['status'],flush=True)
