"""4096-token resource envelope, using synthetic token IDs; not task quality."""
import argparse,json,math,pathlib,time,urllib.request
from transformers import AutoTokenizer
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
t=AutoTokenizer.from_pretrained('/models/de4b8e4d43b917e7706784d8bb445c9af86a3540',local_files_only=True)
unit=t.encode(' The sky is blue.',add_special_tokens=False);assert unit
ids=(unit*(3840//len(unit)+1))[:3840]
body={'rid':'r9-resource-context4096','input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True},'return_logprob':True,'logprob_start_len':-1}
report={'status':'RUNNING','scope':'SYNTHETIC_TOKEN_RESOURCE_ENVELOPE_NOT_QUALITY','prompt_tokens':3840,'max_new_tokens':256,'started_at':time.time(),'request':body}
try:
 r=urllib.request.Request('http://127.0.0.1:30091/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=3000) as response:data=json.load(response)
 report['response']=data
 meta=data['meta_info'];assert meta['prompt_tokens']==3840 and meta['completion_tokens']==256
 logs=meta['output_token_logprobs'];assert len(logs)==256 and all(math.isfinite(x[0]) for x in logs)
 report['status']='PASS_4096_TOKEN_RESOURCE_REQUEST'
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:
 report['ended_at']=time.time();a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],flush=True)
