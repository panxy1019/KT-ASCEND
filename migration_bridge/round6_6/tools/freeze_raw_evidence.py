"""Content manifest for already closed study evidence, no mutation of sources."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    base=Path('/workspace/results').resolve()
    roots=[r.resolve(strict=True) for r in a.roots]
    assert len(set(roots))==len(roots)
    assert all(r.parent==base and r.name.startswith('r65-r66-') for r in roots)
    assert all(not a.output.resolve().is_relative_to(r) for r in roots)
    guard=base/'r65-r66-final-screen-guard-v1'
    assert (guard/'result.json').exists(), 'MODEL_SEQUENCE_STILL_ACTIVE'
    assert json.loads((guard/'reaping.json').read_text())['complete']
    count=0;total=0
    with a.output.open('x') as out:
        for root in roots:
            for path in sorted(root.rglob('*')):
                assert not path.is_symlink(), 'UNEXPECTED_EVIDENCE_SYMLINK'
                if not path.is_file():continue
                before=path.stat()
                sha=digest(path)
                after=path.stat()
                assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), 'EVIDENCE_CHANGED_DURING_HASH'
                out.write(json.dumps(dict(path=str(path.relative_to(base)),bytes=before.st_size,sha256=sha),sort_keys=True)+'\n')
                count+=1;total+=before.st_size
    print(json.dumps(dict(files=count,bytes=total,manifest_sha256=digest(a.output))))


if __name__=='__main__':main()
