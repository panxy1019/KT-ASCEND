"""Abort an active owned HTTP request, then verify clean state-slot reuse."""
import argparse,json,pathlib,urllib.request
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--framework-smoke',type=pathlib.Path,required=True);p.add_argument('--placement',choices=['static','request_boundary'],required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
system=json.loads(a.framework_smoke.read_text())
if system['status']!='PASS_FRAMEWORK_HTTP2':raise RuntimeError('short framework smoke must pass before cancel/reuse')
baseline=system['rows'][0]
baseline_ids=baseline['response'].get('output_ids',[x[1] for x in baseline['response']['meta_info']['output_token_logprobs']])
url='http://127.0.0.1:30091'
def request(path,body):return urllib.request.urlopen(urllib.request.Request(url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300)
rid='r9-cancel-active';report={'status':'RUNNING','scope':'ACTIVE_ABORT_AND_REQUEST_STATE_REUSE','events':[]}
try:
 body={'rid':rid,'input_ids':baseline['request']['input_ids'],'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True}}
 aborted=False
 with request('/generate',body) as response:
  for line in response:
   line=line.decode().strip()
   if not line.startswith('data:'):continue
   payload=line[5:].strip()
   if payload=='[DONE]':break
   data=json.loads(payload);report['events'].append(data)
   if not aborted and data.get('meta_info',{}).get('completion_tokens',0)>0:
    with request('/abort_request',{'rid':rid,'abort_all':False}) as ack:report['abort_http_status']=ack.status
    aborted=True
 assert aborted and report['abort_http_status']==200
 assert any(x.get('meta_info',{}).get('finish_reason',{}).get('type')=='abort' for x in report['events'] if isinstance(x.get('meta_info',{}).get('finish_reason'),dict)),report['events'][-1]
 body=dict(baseline['request']);body['rid']='r9-post-cancel-reuse'
 with request('/generate',body) as response:data=json.load(response)
 report['reuse_response']=data
 if a.placement=='static' and data.get('output_ids',[x[1] for x in data['meta_info']['output_token_logprobs']])!=baseline_ids:raise RuntimeError('post-cancel state leaked into repeated request')
 if a.placement=='request_boundary':
  import math,time
  if not all(math.isfinite(x[0]) for x in data['meta_info']['output_token_logprobs']):raise RuntimeError('nonfinite post-cancel output')
  for attempt in range(1200):
   events=[json.loads(line) for line in pathlib.Path('/out/dynamic-events.jsonl').read_text().splitlines()]
   ids={e.get('request_id') for e in events}
   if {rid,'r9-post-cancel-reuse'}<=ids:break
   time.sleep(.1)
  else:raise RuntimeError('Dynamic abort/reuse boundaries not drained')
  report['dynamic_abort_and_reuse_boundaries']=True
  report['same_placement_static_oracle']='PENDING_NOT_CLAIMED'
 report['status']='PASS_ACTIVE_ABORT_AND_STATE_REUSE'
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],flush=True)
