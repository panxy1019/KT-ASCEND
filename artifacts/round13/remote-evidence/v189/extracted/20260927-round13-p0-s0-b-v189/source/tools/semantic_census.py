"""Classify a validated Qwen4 tensor census without reading tensor payloads.

This is a structural gate, not a numerical or model execution qualification.
CLI rebuilds the census from live safetensors headers using the identity
verifier's shape/offset/index checks; callers must separately verify identity.
"""
import argparse
import collections
import json
import math
import pathlib
import re

PREFIX = 'model.language_model.layers.'
TABLE = re.compile(re.escape(PREFIX) + r'(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$')
ROUTED = re.compile(re.escape(PREFIX) + r'(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$')


def classify(census, config):
    c = config['text_config']
    tensors = census['tensors']
    if (len(tensors) != census['tensor_count']
            or sum(t['bytes'] for t in tensors.values()) != census['payload_bytes']):
        raise ValueError('census count/bytes mismatch')
    groups = {k: {'tensor_count': 0, 'bytes': 0} for k in
              ('routed', 'ple_tables', 'ple_other', 'mtp', 'vision', 'text_other')}
    ple_layers = [i - 1 for i in c['ple_layer_ids']]
    if len(set(ple_layers)) != len(ple_layers) or any(i < 0 or i >= c['num_hidden_layers'] for i in ple_layers):
        raise ValueError('invalid PLE layer numbering')
    heads = c['heads_per_ngram'] * (c['ngram_size'] - 1)
    if heads <= 0 or c['ple_embed_dim'] % heads:
        raise ValueError('invalid PLE head width')
    table_width = c['ple_embed_dim'] // heads
    expert_keys, table_keys = set(), set()
    table_rows = collections.defaultdict(set)
    for name, t in tensors.items():
        routed, table = ROUTED.fullmatch(name), TABLE.fullmatch(name)
        if routed:
            layer, part = int(routed[1]), routed[2]
            if name != f'{PREFIX}{layer}.mlp.experts.{part}':
                raise ValueError('noncanonical expert layer name: ' + name)
            hidden, intermediate, experts = c['hidden_size'], c['moe_intermediate_size'], c['num_experts']
            expected = ([experts, 2 * intermediate, hidden] if part == 'gate_up_proj'
                        else [experts, hidden, intermediate])
            if t['dtype'] != 'BF16' or t['shape'] != expected or t['bytes'] != math.prod(expected) * 2:
                raise ValueError('routed expert shape/dtype/bytes mismatch: ' + name)
            expert_keys.add((layer, part)); group = 'routed'
        elif table:
            layer, shard = int(table[1]), int(table[2])
            if name != f'{PREFIX}{layer}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight':
                raise ValueError('noncanonical PLE shard name: ' + name)
            shape = t['shape']
            if (t['dtype'] != 'BF16' or len(shape) != 2 or shape[0] <= 0
                    or shape[1] != table_width or t['bytes'] != math.prod(shape) * 2):
                raise ValueError('PLE table shape/dtype/bytes mismatch: ' + name)
            table_keys.add((layer, shard)); table_rows[layer].add(shape[0]); group = 'ple_tables'
        elif name.startswith('mtp.'):
            group = 'mtp'
        elif name.startswith('model.visual.'):
            group = 'vision'
        elif '.ple.' in name:
            group = 'ple_other'
        elif name.startswith(('model.language_model.', 'lm_head.')):
            group = 'text_other'
        else:
            raise ValueError('unclassified tensor: ' + name)
        groups[group]['tensor_count'] += 1
        groups[group]['bytes'] += t['bytes']
    if expert_keys != {(i, p) for i in range(c['num_hidden_layers']) for p in ('gate_up_proj', 'down_proj')}:
        raise ValueError('incomplete or unexpected main routed layers')
    if table_keys != {(i, s) for i in ple_layers for s in range(c['split_ngram_parts'])}:
        raise ValueError('PLE layer/shard coverage mismatch')
    if any(len(rows) != 1 for rows in table_rows.values()):
        raise ValueError('unequal PLE shard rows')
    return {'status': 'PASS', 'scope': 'STRUCTURAL_CENSUS_ONLY', 'groups': groups,
            'tensor_count': len(tensors), 'payload_bytes': census['payload_bytes'],
            'ple_checkpoint_layers': ple_layers, 'ple_table_width': table_width,
            'ple_rows_per_shard': {str(i): next(iter(rows)) for i, rows in table_rows.items()},
            'model_execution': 'NOT_RUN'}


def main():
    from verify_download import tensor_census
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_root', type=pathlib.Path)
    args = parser.parse_args()
    config = json.loads((args.model_root / 'config.json').read_text())
    index = json.loads((args.model_root / 'model.safetensors.index.json').read_text())
    print(json.dumps(classify(tensor_census(args.model_root, index), config), indent=2))


if __name__ == '__main__':
    main()
