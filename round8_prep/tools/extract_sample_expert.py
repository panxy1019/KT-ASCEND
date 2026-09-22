"""Extract packed Layer0/E0 as F32 reference storage, not a production GGUF."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit = json.loads(a.audit.read_text())
    a.output.mkdir(exist_ok=False)
    tensors = {}
    sources = {}
    for projection, shape in [('gate_up_proj', [256, 1024, 2048]), ('down_proj', [256, 2048, 512])]:
        key = f'model.language_model.layers.0.mlp.experts.{projection}'
        entry = audit[key]
        if entry['status'] != 'IDENTITY_AND_HEADER_PASS_NOT_CONVERSION':
            raise ValueError('shard identity gate absent')
        path = a.directory / entry['shard']
        with safe_open(path, framework='pt', device='cpu') as handle:
            view = handle.get_slice(key)
            if view.get_shape() != shape:
                raise ValueError('unexpected tensor shape')
            original = view[0].contiguous()
        if original.dtype != torch.bfloat16 or not torch.isfinite(original).all():
            raise ValueError('unexpected dtype/nonfinite')
        expanded = original.float()
        assert torch.equal(expanded.bfloat16(), original)
        tensors[projection] = expanded
        sources[projection] = original
    dest = a.output / 'layer0-expert0-f32.safetensors'
    save_file(tensors, str(dest))
    reloaded = load_file(str(dest))
    for name, tensor in reloaded.items():
        assert torch.equal(tensor, tensors[name])
        assert torch.equal(tensor.bfloat16(), sources[name])
    result = dict(status='BF16_F32_STORAGE_ROUNDTRIP_PASS', layer=0, expert=0,
                  output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  bytes=dest.stat().st_size, gate_up_split='NOT_YET_QUALIFIED',
                  production_gguf=False, operator_qualified=False)
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
