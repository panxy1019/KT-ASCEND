"""Start only the frozen command after complete conversion and fresh admission."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True)
    a=p.parse_args()
    admission=json.loads(a.admission.read_text())
    assert admission['admitted'] and 0<=time.time()-admission['timestamp']<=60
    root=Path('/workspace/round8')
    manifest=json.loads((root/'cpu-experts-f32-v1/manifest.json').read_text())
    assert manifest['status']=='FULL_40_LAYER_CONVERSION_IDENTITY_PASS_NOT_MODEL_QUALIFICATION'
    assert manifest['exact_projection_instances']==30720
    assert {r['layer'] for r in manifest['layers']}==set(range(40))
    config=json.loads((root/'q64-launch-v2/resolved-command.json').read_text())
    for path,digest in config['code_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, path
    env=os.environ.copy();env.update(config['env'])
    # Owned foreground server; outer docker exec/log identifies this process tree.
    os.execvpe(config['argv'][0],config['argv'],env)


if __name__=='__main__':main()
