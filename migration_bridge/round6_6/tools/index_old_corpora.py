"""Fingerprint historical fixture/corpus prompts, excluding all evidence outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def text_hash(s):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',s).split()).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();text=set();ids=set();files=[]
    def visit(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in ['text','prompt','prompt_text','question'] and isinstance(x,str):text.add(text_hash(x))
                if k=='input_ids' and isinstance(x,list) and all(type(t)==int for t in x):
                    ids.add(hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest())
                visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    for path in sorted(a.root.rglob('*.json')):
        parts=path.relative_to(a.root).parts
        if 'round6_6' in parts or 'evidence' in parts or not ({'corpora','fixtures'}&set(parts)):continue
        raw=path.read_bytes()
        try:d=json.loads(raw)
        except (ValueError,UnicodeDecodeError):continue
        visit(d);files.append(dict(path=str(path.relative_to(a.root)),sha256=hashlib.sha256(raw).hexdigest()))
    with a.output.open('x') as f:json.dump(dict(text_sha256=sorted(text),input_ids_sha256=sorted(ids),sources=files),f,indent=2);f.write('\n')
    print(json.dumps(dict(source_files=len(files),text_fingerprints=len(text),token_fingerprints=len(ids))))


if __name__=='__main__':main()
