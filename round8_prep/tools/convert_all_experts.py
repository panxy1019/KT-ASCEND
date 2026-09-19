"""Bounded-layer F32 GGUF conversion, full source and projection identity audit."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe-root', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    identity = json.loads(a.identity.read_text())
    assert identity['status'] == 'CHECKPOINT_IDENTITY_PASS_NOT_MODEL_QUALIFICATION'
    assert identity['revision'] == '59d61f3ce65a6d9863b86d2e96597125219dc754'
    for layer in (0, 20, 39):
        probe = json.loads((a.probe_root/f'multi-expert-layer{layer}-v1/result.json').read_text())
        assert probe['status'] == 'PASS_SINGLE_LAYER_TOP8_NOT_FULL_MODEL'
        assert probe['source_revision'] == identity['revision']
        assert probe['cpu_workers'] == 64 and probe['subpools'] == 2
        assert probe['gguf_roundtrip_exact']
    a.output.mkdir(exist_ok=False)
    for row in identity['files']:
        path = a.directory/row['file']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    index = json.loads((a.directory/'model.safetensors.index.json').read_text())['weight_map']
    report = dict(status='CONVERTING_NOT_FROZEN', source_revision=identity['revision'],
                  source_identity_sha256=sha(a.identity), layers=[], mtp_excluded=True,
                  scope='TEXT_ROUTED_EXPERTS_ONLY', representation='F32_EXACT_EXPANDED_BF16')
    manifest = a.output/'manifest.json'
    for layer in range(40):
        prefix = f'model.language_model.layers.{layer}.mlp.experts.'
        arrays = {}
        for projection, shape in [('gate_up_proj', (256,1024,2048)), ('down_proj',(256,2048,512))]:
            key = prefix+projection
            with safe_open(a.directory/index[key], framework='pt', device='cpu') as f:
                source = f.get_tensor(key)
                assert tuple(source.shape) == shape and source.dtype == torch.bfloat16
                assert torch.isfinite(source).all()
                if projection == 'gate_up_proj':
                    arrays['gate'] = source[:,:512,:].float().contiguous().numpy()
                    arrays['up'] = source[:,512:,:].float().contiguous().numpy()
                else:
                    arrays['down'] = source.float().contiguous().numpy()
                del source
        path = a.output/f'layer-{layer:02d}-f32.gguf'
        w = gguf.GGUFWriter(str(path), 'llama')
        for name, value in arrays.items():
            w.add_tensor(f'blk.{layer}.ffn_{name}_exps.weight', value)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        del w, arrays
        gc.collect()
        # Independent source re-open and per-expert exact comparison of ALL120GiB.
        reader = gguf.GGUFReader(str(path))
        assert len(reader.tensors) == 3
        count = 0
        for tensor in reader.tensors:
            name = tensor.name.split('ffn_')[1].split('_exps')[0]
            key = prefix+('down_proj' if name=='down' else 'gate_up_proj')
            with safe_open(a.directory/index[key], framework='pt', device='cpu') as f:
                view = f.get_slice(key)
                for expert in range(256):
                    expected = view[expert]
                    if name == 'gate': expected = expected[:512,:]
                    elif name == 'up': expected = expected[512:,:]
                    assert np.array_equal(tensor.data[expert], expected.float().numpy())
                    count += 1
        assert count == 768
        report['layers'].append(dict(layer=layer, file=path.name, bytes=path.stat().st_size,
                                     sha256=sha(path), exact_projection_instances=count))
        manifest.write_text(json.dumps(report, indent=2)+'\n')
        print(f'LAYER {layer} CONVERTED_AND_ALL_768_PROJECTIONS_EXACT', flush=True)
        del reader, tensor, view, expected
        gc.collect()
    assert len(report['layers']) == 40
    report['status'] = 'FULL_40_LAYER_CONVERSION_IDENTITY_PASS_NOT_MODEL_QUALIFICATION'
    report['exact_projection_instances'] = sum(r['exact_projection_instances'] for r in report['layers'])
    report['payload_bytes'] = 40*256*3*2048*512*4
    manifest.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
