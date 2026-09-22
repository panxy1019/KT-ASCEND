"""Metadata only: pinned sources, never downloads checkpoint shards."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def fetch(url, target):
    req = urllib.request.Request(url, headers={'User-Agent': 'KT-ASCEND-prep'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(40 * 1024 * 1024 + 1)
    if len(data) > 40 * 1024 * 1024:
        raise ValueError('METADATA_SIZE_LIMIT')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != data:
        raise ValueError(f'FROZEN_FILE_CHANGED: {target}')
    target.write_bytes(data)
    return data

def main():
    repo = 'Qwen/Qwen3.5-35B-A3B'
    out = ROOT / '01_MODEL'
    info = json.loads(fetch(f'https://huggingface.co/api/models/{repo}?blobs=true', out/'hub-info.json'))
    revision = info['sha']
    inventory = {x['rfilename']: x for x in info['siblings']}
    names = ['config.json', 'tokenizer_config.json', 'tokenizer.json',
             'model.safetensors.index.json', 'generation_config.json', 'README.md']
    files = {}
    for name in names:
        data = fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', out/name)
        files[name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    index = json.loads((out/'model.safetensors.index.json').read_text())
    shards = []
    for name in sorted(set(index['weight_map'].values())):
        item = inventory[name]
        shards.append(dict(name=name, size=item['size'], hub_lfs_sha256=item.get('lfs', {}).get('sha256'),
                           downloaded=False))
    result = dict(repo=repo, revision=revision, files=files, shards=shards,
                  total_shard_bytes=sum(x['size'] for x in shards),
                  shard_identity_scope='HUB_DECLARED_NOT_LOCAL_VERIFICATION')
    (out/'QWEN35_MODEL_MANIFEST.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(revision=revision, shard_count=len(shards), bytes=result['total_shard_bytes'])))

if __name__ == '__main__':
    main()
