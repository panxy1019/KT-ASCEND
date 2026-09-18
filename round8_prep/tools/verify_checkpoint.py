"""Streaming full checkpoint identity audit against the frozen manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(directory, manifest):
    expected = {s['name']: (s['size'], s['hub_lfs_sha256']) for s in manifest['shards']}
    expected.update({n:(v['size'],v['sha256']) for n,v in manifest['files'].items() if n.endswith('.json')})
    rows=[]
    for name,(size,digest) in sorted(expected.items()):
        path=directory/name
        if not path.is_file() or path.stat().st_size!=size:
            raise ValueError(f'MISSING_OR_SIZE_MISMATCH:{name}')
        h=hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=digest:raise ValueError(f'SHA_MISMATCH:{name}')
        rows.append(dict(file=name,bytes=size,sha256=digest))
    return dict(status='CHECKPOINT_IDENTITY_PASS_NOT_MODEL_QUALIFICATION',repo=manifest['repo'],revision=manifest['revision'],files=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.directory,json.loads(a.manifest.read_text())),indent=2))
