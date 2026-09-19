"""Fixed, batch-one client timing for one profile/block; orchestration alternates profiles."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


ROOT = Path('/workspace/round8')
SEEDS = {
    'short': 'Explain in English why the sky is blue, using numbered sentences. ',
    'medium': 'Discuss how a cache works in a computer and give a concrete example. ',
    'long': 'Describe the steps of testing a distributed program and how to record failures. ',
}
TARGETS = {'short': (128, 64), 'medium': (512, 128), 'long': (1024, 256)}


def make_inputs():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model-sample', local_files_only=True)
    out = {}
    for name, text in SEEDS.items():
        target, decode = TARGETS[name]
        rendered = tokenizer.apply_chat_template(
            [dict(role='user', content=text * (target // 12 + 2))],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)[:target]
        assert len(ids) == target
        out[name] = dict(input_ids=ids, decode=decode)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('q64', 'q32'), required=True)
    parser.add_argument('--block', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    out = ROOT / 'qwen-static-perf-v1' / f'{args.profile}-block{args.block}'
    out.mkdir(parents=True, exist_ok=False)
    inputs = make_inputs()
    protocol = dict(version=1, inputs=inputs, max_running_requests=1,
                    temperature=0, ignore_eos=True, graph=False,
                    dynamic=False, deferred=False, diagnostic_hooks=False,
                    order=['short', 'medium', 'long'])
    blob = json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    (out / 'protocol.json').write_text(blob)
    url = 'http://127.0.0.1:18400/generate'
    warm = dict(input_ids=inputs['short']['input_ids'],
                sampling_params=dict(temperature=0, max_new_tokens=8, ignore_eos=True))
    warm_response = requests.post(url, json=warm, timeout=3600)
    warm_response.raise_for_status()
    assert len(warm_response.json()['output_ids']) == 8
    rows = []
    for name in protocol['order']:
        item = inputs[name]
        payload = dict(input_ids=item['input_ids'],
                       sampling_params=dict(temperature=0, max_new_tokens=item['decode'],
                                            ignore_eos=True))
        started = time.perf_counter()
        response = requests.post(url, json=payload, timeout=3600)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        assert len(data['output_ids']) == item['decode']
        row = dict(workload=name, input_tokens=len(item['input_ids']),
                   output_tokens=len(data['output_ids']), client_seconds=elapsed,
                   server_e2e_seconds=data['meta_info'].get('e2e_latency'),
                   output_sha256=hashlib.sha256(json.dumps(data['output_ids']).encode()).hexdigest())
        rows.append(row)
        (out / 'raw.jsonl').open('a').write(json.dumps(row) + '\n')
        print(name, elapsed, flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='BLOCK_COMPLETE',
        profile=args.profile, block=args.block,
        protocol_sha256=hashlib.sha256(blob.encode()).hexdigest(), rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
