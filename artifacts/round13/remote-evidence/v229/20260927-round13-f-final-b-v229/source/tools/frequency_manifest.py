"""Bind real complete route traces to HTTP inputs and the exact admitted runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.frequency import build_frequency, payload_digest
from runtime.candidate_bank import MANIFEST_SHA


def produce(stats, smoke, corpus, payload, admission):
    if (stats.get('schema') != 1 or smoke.get('status') != 'PASS_FRAMEWORK_HTTP2' or
            corpus.get('split') != 'QWEN38_SCHED_DEV' or stats.get('placement_epoch') != 0):
        raise ValueError('complete real Static DEV trace and passing HTTP run required')
    prompts = corpus['prompts']
    requests = stats['requests']
    if not 1 <= len(prompts) <= 64 or len(requests) != len(prompts) or len(smoke['rows']) != len(prompts):
        raise ValueError('calibration corpus coverage mismatch')
    for prompt, trace, row in zip(prompts, requests, smoke['rows'], strict=True):
        request = row['request']
        ids = request['input_ids']
        if (trace['request_id'] != prompt['id'] or request['rid'] != prompt['id'] or
                trace['tokens'] != len(ids) or trace['input_ids_sha256'] != payload_digest(ids) or
                row.get('finite') is not True):
            raise ValueError('trace does not match actual corpus HTTP inputs')
    if stats['statistics']['tokens'] != sum(r['tokens'] for r in requests):
        raise ValueError('route token totals differ from HTTP inputs')
    if stats['statistics']['completed_phases'] != len(requests):
        raise ValueError('route request totals differ from HTTP inputs')
    residents = stats['initial_residents']
    if (len(residents) != 48 or any(len(row) != 64 or len(set(row)) != 64 or
            any(type(i) is not int or not 0 <= i < 512 for i in row) for row in residents)):
        raise ValueError('missing real producer placement')
    identity = dict(model_manifest_sha256=MANIFEST_SHA,
                    tokenizer_sha256=payload_digest(smoke['tokenizer_files']),
                    runtime_sha256=payload_digest(dict(native=payload['runtime'], sglang=payload['sglang'])),
                    corpus_sha256=smoke['corpus_sha256'], placement_sha256=payload_digest(residents))
    if (admission.get('status') != 'CANDIDATE_READY_FOR_MONITORED_FULL_MODEL_RUN' or
            admission.get('bank_identity', {}).get('source_identity', {}).get('manifest_sha256') != MANIFEST_SHA):
        raise ValueError('admitted producer required')
    # Receipt linkage is additionally checked against raw bytes by the CLI.
    return {capacity: build_frequency(stats['statistics'], identity, capacity=capacity) for capacity in (32, 64)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stats', 'smoke', 'corpus', 'payload', 'admission', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    docs = {name: json.loads(getattr(a, name).read_text()) for name in ('stats', 'smoke', 'corpus', 'payload', 'admission')}
    if hashlib.sha256(a.corpus.read_bytes()).hexdigest() != docs['smoke']['corpus_sha256']:
        raise ValueError('corpus bytes changed')
    if hashlib.sha256(a.payload.read_bytes()).hexdigest() != docs['admission']['payload_sha256']:
        raise ValueError('producer payload differs from admission')
    results = produce(**docs)
    a.output.mkdir(exist_ok=False)
    for capacity, document in results.items():
        (a.output / f'frequency-q{capacity}.json').write_text(json.dumps(document, indent=2) + '\n')
    print('GENERATED_Q32_Q64_NOT_DEVICE_QUALIFIED')
