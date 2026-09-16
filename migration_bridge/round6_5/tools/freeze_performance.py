"""Deterministic fixed-length diagnostic timing inputs, not quality examples."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='257034644e2a3ef516722d354be4787965a5c937205bf73eb629ace57f42c19f':
        raise ValueError('PERF_SOURCE_DRIFT')
    old=json.loads(raw);ids=old['shapes'][0]['input_ids']
    if len(ids)!=192:raise ValueError('PERF_SOURCE_SHAPE')
    obj=dict(version='r65-static-performance-v1',source_sha256=hashlib.sha256(raw).hexdigest(),
        tokenizer_sha256=old['tokenizer_sha256'],batch=1,ignore_eos=True,cache=False,
        observer=False,output_tokens=64,warmups_per_shape=2,timed_per_shape=3,
        shapes=[dict(id='short192',input_ids=ids),dict(id='long1024',input_ids=(ids*6)[:1024])],
        long_construction='repeat frozen192 six times and take first1024, diagnostic not quality',
        timing='client SSE TTFT and decode, no hardware-kernel prefill inference')
    obj['canonical_sha256']=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with a.output.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')


if __name__=='__main__':main()
