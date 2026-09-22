"""New attempt identity; keep failed attempt and expensive verified GGUF hash."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    src,dst=map(Path,sys.argv[1:3]);dst.mkdir(exist_ok=False)
    shutil.copytree(src/'profiles',dst/'profiles')
    provenance=json.loads((src/'provenance.json').read_text())
    provenance['previous_attempt']=str(src)
    provenance['previous_failure']=json.loads((src/'dev-stat-summary.json').read_text())
    provenance['tools_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest()
        for x in sorted(Path(__file__).parent.iterdir()) if x.is_file()}
    (dst/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    cfg=json.loads((src/'uniform-dev-config.json').read_text())
    for k in ('profile','activation_path'):cfg[k]=cfg[k].replace(str(src),str(dst))
    (dst/'uniform-dev-config.json').write_text(json.dumps(cfg,indent=2)+'\n')


if __name__=='__main__':main()
