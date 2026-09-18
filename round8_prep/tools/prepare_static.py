"""Offline, config-derived budgets/masks/index inventory. Not a kernel test."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def save(name, obj):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def profile(config, resident):
    n, layers = config['num_experts'], config['num_hidden_layers']
    if not 0 <= resident <= n:
        raise ValueError('invalid resident budget')
    # A fixed evenly-spaced permutation gives nested masks, independent of quality.
    order = sorted(range(n), key=lambda e: int(f'{e:08b}'[::-1], 2))
    ids = sorted(order[:resident])
    return dict(resident=resident, layers=[dict(layer=l, npu_ids=ids,
        cpu_ids=[e for e in range(n) if e not in ids],
        logical_to_physical=[ids.index(e) if e in ids else -1 for e in range(n)])
        for l in range(layers)], mask_selection='fixed_bit_reversal_uniform',
        ownership='disjoint_complete', placeholder_id=0,
        placeholder_weight=0, mapping_dtype='int32',
        status='PREP_CANDIDATE_NOT_RUNTIME_QUALIFIED')

def main():
    c = json.loads((ROOT/'01_MODEL/config.json').read_text())['text_config']
    expected = dict(hidden_size=2048, num_hidden_layers=40, num_experts=256,
                    num_experts_per_tok=8, moe_intermediate_size=512,
                    shared_expert_intermediate_size=512)
    if any(c.get(k) != v for k,v in expected.items()):
        raise ValueError('CONFIG_MISMATCH')
    h, m, l, e = c['hidden_size'], c['moe_intermediate_size'], c['num_hidden_layers'], c['num_experts']
    parameters = 3*h*m
    rows = [dict(profile=f'Q{r}', routed_bf16_bytes=l*r*parameters*2,
                 routed_bf16_gib=l*r*parameters*2/2**30) for r in (256,64,32,16,8)]
    save('04_MEMORY/QWEN35_RESIDENCY_PLAN.json', dict(
        evidence='ANALYTIC_DERIVED', per_expert_parameters=parameters,
        cpu_f32_full_pool_bytes=l*e*parameters*4,
        cpu_bf16_full_pool_bytes=l*e*parameters*2, profiles=rows,
        excludes=['attention','GDN state','embeddings','shared expert','vision',
                  'KV cache','workspace','allocator','staging','anonymous copies'],
        runtime_ram_and_hbm='NOT_MEASURED'))
    for r in (64,32,16):
        save(f'04_MEMORY/q{r}_profile.json', profile(c,r))
    weights = json.loads((ROOT/'01_MODEL/model.safetensors.index.json').read_text())['weight_map']
    experts = {k:v for k,v in weights.items() if k.startswith('model.language_model.layers.') and '.mlp.experts.' in k}
    layers = sorted({int(re.search(r'layers\.(\d+)', k).group(1)) for k in experts})
    save('01_MODEL/WEIGHT_INVENTORY.json', dict(index_keys=len(weights),
        expert_keys=len(experts), expert_layers=layers, examples=dict(list(experts.items())[:8]),
        excluded_mtp_expert_keys=sum(k.startswith('mtp.') and '.mlp.experts.' in k for k in weights),
        tensor_shapes='UNVERIFIED_INDEX_HAS_NO_SHAPES',
        conversion='REQUIRES_SAFETENSORS_HEADER_AND_REAL_SAMPLE'))
    manifest = json.loads((ROOT/'01_MODEL/QWEN35_MODEL_MANIFEST.json').read_text())
    for name, item in manifest['files'].items():
        if hashlib.sha256((ROOT/'01_MODEL'/name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('METADATA_HASH_MISMATCH')
    print('config, metadata hashes, residency profiles and index inventory: PASS')

if __name__ == '__main__': main()
