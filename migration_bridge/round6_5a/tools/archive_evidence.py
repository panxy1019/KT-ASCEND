"""Hash all diagnostic artifacts; archive compact evidence, retain raw on target."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();base=Path('/workspace/results')
    roots=[base/name for name in ('r65-5a-subset-v1','r65-5a-canonical-v1',
        'r65-5a-bisection-v1','r65-5a-bisection-v2','r65-5a-causal-v1','r65-5a-prefill-v1',
        'r65-5a-runtime-source-manifest-v1.json')]
    rows=[];selected=[]
    for root in roots:
        for path in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if not path.is_file():continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
            compact=(path.suffix in ('.json','.jsonl','.log','.txt','.sha256') or
                path.suffix=='.pt' and ('/probe/' in str(path) or '-interventions/' in str(path)))
            rows.append(dict(path=str(path.relative_to(base)),bytes=path.stat().st_size,
                             sha256=h.hexdigest(),in_compact_archive=compact))
            if compact:selected.append(path)
    manifest=base/'r65-5a-artifact-manifest-v1.json'
    with manifest.open('x') as f:
        json.dump(dict(scope='ALL_DIAGNOSTIC_FILES_SHA_FROZEN_RAW_RETAINED_ON_TARGET',files=rows),f,indent=2);f.write('\n')
    with tarfile.open(a.output,'x:gz') as archive:
        for path in selected+[manifest]:archive.add(path,arcname=str(path.relative_to(base)),recursive=False)
    print(json.dumps(dict(files=len(rows),archived=len(selected)+1,archive=str(a.output),
        sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
