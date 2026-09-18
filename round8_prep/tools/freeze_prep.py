"""Hash authored artifacts and snapshots; exclude disposable build state and self-manifest."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'FINAL/ROUND8_PREP_SHA256.json'

def main():
    result={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==TARGET: continue
        rel=p.relative_to(ROOT)
        if any(x in rel.parts for x in ('venv','__pycache__','wheels','build')): continue
        result[str(rel)]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    TARGET.write_text(json.dumps(dict(scope='PREP_ARTIFACTS_NOT_MODEL_SHARDS',files=result),indent=2)+'\n')
    print(f'Frozen {len(result)} prep artifacts')

if __name__=='__main__': main()
