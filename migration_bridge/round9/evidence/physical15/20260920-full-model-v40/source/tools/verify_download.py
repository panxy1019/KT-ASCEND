"""Streaming identity audit against an independently frozen official Hub manifest."""
import argparse
import hashlib
import json
import pathlib
import math
import struct
import time


def verify_file(root, row):
    root = root.resolve()
    path = (root / row['path']).resolve()
    if pathlib.Path(row['path']).is_absolute() or not path.is_relative_to(root):
        raise ValueError('path escapes model directory')
    size = path.stat().st_size
    if size != row['size']:
        raise ValueError(f"size mismatch: {row['path']}")
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            blob.update(chunk)
    if row.get('sha256'):
        valid = sha.hexdigest() == row['sha256']
    else:
        valid = blob.hexdigest() == row['git_blob']
    if not valid:
        raise ValueError(f"content identity mismatch: {row['path']}")
    return {'path': row['path'], 'size': size, 'sha256': sha.hexdigest(),
            'status': 'PASS'}


def tensor_census(root, index):
    widths = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8,
              'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    tensors = {}
    total = 0
    for shard in sorted(set(index['weight_map'].values())):
        path = (root / shard).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('index path escape')
        with path.open('rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            if not 0 < header_len <= 64 * 1024 * 1024:
                raise ValueError('unbounded safetensors header')
            header = json.loads(f.read(header_len))
        cursor = 0
        entries = [(name, value) for name, value in header.items() if name != '__metadata__']
        for name, value in sorted(entries, key=lambda item: item[1]['data_offsets'][0]):
            start, end = value['data_offsets']
            if name in tensors or index['weight_map'].get(name) != shard:
                raise ValueError('tensor identity/index mismatch')
            if value['dtype'] not in widths:
                raise ValueError(f"dtype needs explicit audit: {value['dtype']}")
            expected = math.prod(value['shape']) * widths[value['dtype']]
            if start != cursor or end < start or end - start != expected:
                raise ValueError('shape/offset/gap mismatch')
            tensors[name] = dict(value, shard=shard, bytes=expected)
            cursor = end
            total += expected
        if 8 + header_len + cursor != path.stat().st_size:
            raise ValueError('shard payload boundary mismatch')
    if set(tensors) != set(index['weight_map']) or total != int(index['metadata']['total_size']):
        raise ValueError('global tensor coverage/bytes mismatch')
    return {'tensor_count': len(tensors), 'payload_bytes': total, 'tensors': tensors}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=pathlib.Path)
    p.add_argument('model_dir', type=pathlib.Path)
    p.add_argument('evidence_dir', type=pathlib.Path)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Distinct timestamped receipts preserve failed attempts; never reuse a PASS.
    receipt = a.evidence_dir / f'verification-{time.time_ns()}.json'
    try:
        names = [x['path'] for x in manifest['files']]
        if not names or len(names) != len(set(names)):
            raise ValueError('empty or duplicate manifest')
        for row in manifest['files']:
            rows.append(verify_file(a.model_dir, row))
            print(json.dumps(rows[-1]), flush=True)
        index = json.loads((a.model_dir / 'model.safetensors.index.json').read_text())
        required = set(index['weight_map'].values())
        actual = {x['path'] for x in rows if x['path'].endswith('.safetensors')}
        if required != actual:
            raise ValueError('index/shard coverage mismatch')
        census = tensor_census(a.model_dir, index)
        result = {'status': 'PASS', 'revision': manifest['revision'],
                  'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  'files': rows, 'shards': len(required), 'census': census,
                  'model_execution': 'NOT_RUN'}
    except Exception as exc:
        result = {'status': 'FAIL', 'error': str(exc), 'files': rows}
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        raise
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print('DOWNLOAD_IDENTITY_PASS; model execution remains NOT_RUN', flush=True)


if __name__ == '__main__':
    main()
