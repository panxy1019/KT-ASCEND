"""Round6.4 D1 S6 repeat/restart; no historical numerical candidate prerequisite.

Only deterministic runtime evidence. This does not qualify quality, oracle,
actual expert-forward lifecycle, or the combined Deferred feature.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal

from run_compact_controls import checked, one_profile, S6_SHA


def validate_entry(corpus, contract):
    if (corpus['corpus'] != 'S6' or corpus['sha256'] != S6_SHA or
            len(corpus['prompts']) != 12 or len({p['id'] for p in corpus['prompts']}) != 12):
        raise ValueError('S6_IDENTITY_MISMATCH')
    if (contract['version'] != 'round6.4-approx-v1' or
            contract['same_path_determinism_required'] is not True or
            contract['quality']['approved'] is not True or
            contract['quality']['absolute_noninferiority_margin'] != 0.02):
        raise ValueError('ROUND64_APPROVAL_MISMATCH')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--contract-sha256', required=True)
    p.add_argument('--provenance', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    contract_sha = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    if contract_sha != args.contract_sha256:
        raise ValueError('APPROVAL_FILE_DRIFT')
    corpus = checked(args.corpus)
    validate_entry(corpus, json.loads(args.contract.read_text()))
    if args.output.parent != Path('/workspace/results') or not args.output.name.startswith('r64-'):
        raise ValueError('DEDICATED_OUTPUT_REQUIRED')
    provenance = json.loads(args.provenance.read_text())
    # Verify all deployed artifacts in supplied actual-checkout metadata now,
    # before the first server. Do not equate a local commit with A3 overlays.
    for filename, expected in provenance['file_sha256'].items():
        with open(filename, 'rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError(f'DEPLOYED_SOURCE_DRIFT:{filename}')
    frozen_extra = {
        '/workspace/kt-src/third_party/sglang/python/sglang/srt/model_executor/model_runner.py':
            '71b1daaf2a78afde91794963eb5c6a0779bb6ed931841e0a56f8b8bf34ce88d9',
        '/workspace/results/placements/placement_d1.pt':
            '23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c',
        '/workspace/results/gguf/deepseek-v2-lite-round6-d3-f32.gguf':
            '4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f',
    }
    for filename, expected in frozen_extra.items():
        if filename in provenance['file_sha256']:
            actual = provenance['file_sha256'][filename]
        else:
            with open(filename, 'rb') as source:
                actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'FROZEN_PROFILE_DRIFT:{filename}')
    binding = dict(contract_sha256=contract_sha,
        provenance_sha256=hashlib.sha256(args.provenance.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lifecycle_runner_sha256=hashlib.sha256(Path(__file__).with_name('run_compact_controls.py').read_bytes()).hexdigest(),
        frozen_extra=frozen_extra)
    selected = {row['id']: row for row in corpus['prompts']}
    args.output.mkdir(exist_ok=False)
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith(('KT_R61_', 'KT_R62_', 'KT_R63_', 'KT_DEBUG_')):
            del env[name]
    env.update(ASCEND_RT_VISIBLE_DEVICES='0', SGLANG_APPLY_CONFIG_BACKUP='none',
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION='false', KT_R6_ENABLE_TENSOR_DUMP='0',
        KT_R6_ENABLE_ROUTE_RECORDER='1')
    def cancelled(signum, frame):
        raise InterruptedError(f'RUNTIME_CANCELLED:{signum}')
    signal.signal(signal.SIGTERM, cancelled)
    state = dict(status='RUNNING', binding=binding, completed=[])
    try:
        prior = one_profile('D1', args.output/'same_process', selected, env,
                            runtime=True, binding=binding)
        state['completed'].append('same_process')
        (args.output/'progress.json').write_text(json.dumps(state, indent=2)+'\n')
        one_profile('D1', args.output/'clean_restart', selected, env,
                    runtime=True, binding=binding, restart=prior)
        state['completed'].append('clean_restart')
        state['status'] = 'RUNTIME_PASS'
    except BaseException as exc:
        state.update(status='FAIL', exception=repr(exc))
        raise
    finally:
        (args.output/'progress.json').write_text(json.dumps(state, indent=2)+'\n')


if __name__ == '__main__':
    main()
