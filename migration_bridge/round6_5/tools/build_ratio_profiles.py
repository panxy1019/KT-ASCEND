"""Freeze nested uniform masks without using routing/quality observations."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build(config, layers, resident):
    n = config['n_routed_experts']
    if n != 64 or resident not in (8, 16, 32, 64):
        raise ValueError('UNFROZEN_BUDGET')
    moe = [i for i in range(config['num_hidden_layers'])
           if i >= config['first_k_dense_replace'] and i % config['moe_layer_freq'] == 0]
    if len(layers) != len(set(layers)) or not set(layers) <= set(moe):
        raise ValueError('INVALID_LAYERS')
    rows = []
    for layer in moe:
        gpu = list(range(0, n, n // resident)) if layer in layers else list(range(n))
        cpu = sorted(set(range(n)) - set(gpu))
        rows.append(dict(layer=layer, npu_ids=gpu, cpu_ids=cpu))
    total = sum(len(r['npu_ids']) for r in rows)
    obj = dict(version='round6.5-uniform-v1', resident=resident, selected_layers=layers,
               moe_layers=moe, rows=rows, total_npu_experts=total,
               gpu_experts_ratio=total/(len(moe)*n), deferred=0, dynamic=False)
    obj['canonical_sha256'] = digest(obj)
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    config = json.loads(args.config.read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    profiles = {}
    all_layers = list(range(1, 27))
    for scope, layers in [('single', [17]), ('four', [1, 9, 17, 26]), ('all', all_layers)]:
        for resident in (64, 32, 16, 8):
            name = f'{scope}-R{resident}'
            data = build(config, layers, resident)
            counts = torch.zeros((1, config['num_hidden_layers'], 64), dtype=torch.int64)
            for row in data['rows']:
                counts[0, row['layer'], row['npu_ids']] = 1
            target = args.output / (name + '.pt')
            torch.save({'logical_count': counts}, target)
            data['pt_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
            (args.output / (name + '.json')).write_text(json.dumps(data, indent=2)+'\n')
            profiles[name] = data
    (args.output/'ratio_profiles.json').write_text(json.dumps(profiles, indent=2)+'\n')


if __name__ == '__main__':
    main()
