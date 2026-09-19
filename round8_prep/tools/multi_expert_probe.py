"""Diagnostic real 256-expert layer conversion and top8 CPU execution.

Not full-model/router/shared-expert qualification. One layer per process.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import torch_npu
from kt_kernel import KTMoEWrapper
from safetensors import safe_open


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True, choices=range(40))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    identity = json.loads(a.identity.read_text())
    assert identity['status'] == 'CHECKPOINT_IDENTITY_PASS_NOT_MODEL_QUALIFICATION'
    assert identity['revision'] == '59d61f3ce65a6d9863b86d2e96597125219dc754'
    index_path = a.directory / 'model.safetensors.index.json'
    expected = {row['file']: row for row in identity['files']}
    assert digest(index_path) == expected[index_path.name]['sha256']
    index = json.loads(index_path.read_text())['weight_map']
    a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(8675309)
    weights = {}
    checked = set()
    for proj, shape in [('gate_up_proj', (256, 1024, 2048)), ('down_proj', (256, 2048, 512))]:
        key = f'model.language_model.layers.{a.layer}.mlp.experts.{proj}'
        shard = a.directory / index[key]
        if shard.name not in checked:
            assert digest(shard) == expected[shard.name]['sha256']
            checked.add(shard.name)
        with safe_open(shard, framework='pt', device='cpu') as f:
            source = f.get_tensor(key)
            assert tuple(source.shape) == shape and source.dtype == torch.bfloat16
            assert torch.isfinite(source).all()
            if proj == 'gate_up_proj':
                gate, up = source.chunk(2, dim=1)
                weights['gate'] = gate.float().contiguous()
                weights['up'] = up.float().contiguous()
            else:
                weights['down'] = source.float().contiguous()
    path = a.output / f'layer{a.layer}-256-f32.gguf'
    writer = gguf.GGUFWriter(str(path), 'llama')
    for name, value in weights.items():
        writer.add_tensor(f'blk.{a.layer}.ffn_{name}_exps.weight', value.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(str(path))
    assert len(reader.tensors) == 3
    for tensor in reader.tensors:
        name = tensor.name.split('ffn_')[1].split('_exps')[0]
        assert np.array_equal(tensor.data, weights[name].numpy())
    report = dict(scope='REAL_SINGLE_LAYER_256_EXPERT_TOP8_CPU_ONLY', layer=a.layer,
                  source_revision=identity['revision'], gguf_sha256=digest(path),
                  gguf_roundtrip_exact=True, cpu_workers=64, subpools=2,
                  numa_nodes=[4, 5], rows=[])
    def save():
        (a.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    cpu = KTMoEWrapper(layer_idx=a.layer, num_experts=256, num_experts_per_tok=8,
        hidden_size=2048, moe_intermediate_size=512,
        gpu_experts_mask=torch.zeros(256, dtype=torch.bool),
        cpuinfer_threads=64, threadpool_count=2, numa_nodes=[4, 5],
        weight_path=str(path), chunked_prefill_size=64,
        max_deferred_experts_per_token=0, method='LLAMAFILE')
    cpu.load_weights(torch.arange(256, dtype=torch.int32))
    selected = torch.tensor([0, 1, 31, 63, 127, 191, 254, 255], dtype=torch.int64)
    for qlen in (1, 8, 64):
        x = torch.randn(qlen, 2048).bfloat16()
        ids = selected.expand(qlen, -1).contiguous()
        routes = torch.softmax(torch.randn(qlen, 8), dim=-1).contiguous()
        ref = torch.zeros(qlen, 2048)
        for j, expert in enumerate(selected.tolist()):
            g = (x.float() @ weights['gate'][expert].T).bfloat16().float()
            u = (x.float() @ weights['up'][expert].T).bfloat16().float()
            y = ((torch.nn.functional.silu(g)*u).bfloat16().float()
                 @ weights['down'][expert].T).bfloat16().float()
            ref += y * routes[:, j:j+1].bfloat16().float()
        outputs = [cpu.forward(x, ids, routes).detach().cpu().clone() for _ in range(10)]
        finite = all(torch.isfinite(y).all().item() for y in outputs)
        exact = all(torch.equal(outputs[0], y) for y in outputs[1:])
        rel = ((outputs[0].double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-30)).item()
        report['rows'].append(dict(qlen=qlen, experts=selected.tolist(),
                                  finite=finite, repeat_exact=exact, relative_l2=rel))
        save()
        assert finite and exact and rel <= .01, 'LOCAL_TOP8_GATE_FAILURE'
    report['status'] = 'PASS_SINGLE_LAYER_TOP8_NOT_FULL_MODEL'
    save()


if __name__ == '__main__':
    main()
