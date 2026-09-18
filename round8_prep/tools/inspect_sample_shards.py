"""Read-only, streaming whole-shard identity and packed header audit."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def inspect(directory, manifest):
    result = {}
    for number, projection in ((6, 'gate_up_proj'), (12, 'down_proj')):
        name = f'model.safetensors-{number:05d}-of-00014.safetensors'
        expected = next(s for s in manifest['shards'] if s['name'] == name)
        path = directory / name
        if path.stat().st_size != expected['size']:
            raise ValueError(f'{name}: size mismatch')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected['hub_lfs_sha256']:
            raise ValueError(f'{name}: SHA mismatch')
        with path.open('rb') as source:
            header_size = struct.unpack('<Q', source.read(8))[0]
            if not 0 < header_size <= min(64 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError('invalid header length')
            header = json.loads(source.read(header_size))
        key = f'model.language_model.layers.0.mlp.experts.{projection}'
        tensor = header[key]
        shape = tensor['shape']
        count = 256 * 2048 * 512 * (2 if projection == 'gate_up_proj' else 1)
        if tensor['dtype'] != 'BF16' or len(shape) != 3 or shape[0] != 256 or math.prod(shape) != count:
            raise ValueError(f'{key}: unexpected packed shape/dtype: {tensor}')
        start, end = tensor['data_offsets']
        if not (0 <= start < end <= path.stat().st_size - 8 - header_size) or end - start != count * 2:
            raise ValueError(f'{key}: invalid payload extent')
        result[key] = dict(shard=name, sha256=digest.hexdigest(), tensor=tensor,
                           data_start=8 + header_size,
                           status='IDENTITY_AND_HEADER_PASS_NOT_CONVERSION')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.directory, json.loads(args.manifest.read_text())), indent=2))
