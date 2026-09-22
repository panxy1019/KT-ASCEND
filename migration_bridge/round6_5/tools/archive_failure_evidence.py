"""Package bounded metadata and first-failure tensors, never model weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=Path('/workspace/results')
    roots=['r65-preflight-v1','r65-scale-matrix-v2/four-R32','r65-four-R32-failure-replay-v1',
        'r65-four-R32-localization-v1','r65-failure-experts-v1','r65-capacity-sweep-v1']
    files=set()
    for root in roots:
        for path in (base/root).rglob('*'):
            if not path.is_file():continue
            if path.suffix in ('.json','.jsonl','.log') and 'resource-monitor' not in path.name:files.add(path)
            elif path.parent.name=='stagewise' and path.name.endswith('_call0055.pt'):files.add(path)
    first=json.loads((base/'r65-scale-matrix-v2/four-R32/first-failure-audit.json').read_text())
    files.update(Path(r['file']) for r in first['captures'].values())
    files.add(base/'r65-scale-matrix-v2/progress.json')
    manifest=[]
    with tarfile.open(a.output,'x:gz') as archive:
        for path in sorted(files):
            name=str(path.relative_to(base));size=path.stat().st_size
            if size>10*1024**2:raise ValueError('UNEXPECTED_LARGE_ARTIFACT:'+name)
            payload=path.read_bytes();manifest.append(dict(path=name,bytes=size,sha256=hashlib.sha256(payload).hexdigest()))
            archive.add(path,arcname=name,recursive=False)
        payload=(json.dumps(dict(scope='FIRST_FAILURE_METADATA_AND_TOKEN_TENSORS',files=manifest),indent=2)+'\n').encode()
        info=tarfile.TarInfo('r65-failure-evidence-manifest.json');info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
    print(json.dumps(dict(files=len(manifest),archive_bytes=a.output.stat().st_size,
        sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
