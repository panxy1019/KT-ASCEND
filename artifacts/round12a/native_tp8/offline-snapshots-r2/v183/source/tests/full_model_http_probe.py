"""Bounded first real SGLang HTTP prefill/decode; no quality claim."""
import argparse,json,math,pathlib,time,urllib.request,urllib.error
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--wait',type=int,default=600);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
base='http://127.0.0.1:30091'
def request(path,body=None,timeout=300):
 data=None if body is None else json.dumps(body).encode()
 r=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=timeout) as response:return response.read().decode()
report={'status':'RUNNING','scope':'FIRST_REAL_HTTP_PREFILL_DECODE_ONLY','started_at':time.time()}
try:
 end=time.monotonic()+a.wait
 while True:
  try:
   request('/health',timeout=2);break
  except (urllib.error.URLError,TimeoutError):
   if time.monotonic()>end:raise TimeoutError('server did not become healthy within readiness window')
   time.sleep(2)
 report['healthy_at']=time.time()
 body={'text':'Hello','sampling_params':{'temperature':0,'max_new_tokens':4,'ignore_eos':True},'return_logprob':True,'logprob_start_len':0,'top_logprobs_num':5}
 report['request']=body
 report['response']=json.loads(request('/generate',body))
 meta=report['response']['meta_info'];assert meta['completion_tokens']==4
 for item in meta.get('output_token_logprobs',[]):assert math.isfinite(item[0])
 assert report['response'].get('text') is not None
 report['status']='PASS_FIRST_REAL_HTTP_PREFILL_DECODE'
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:
 report['ended_at']=time.time();a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
