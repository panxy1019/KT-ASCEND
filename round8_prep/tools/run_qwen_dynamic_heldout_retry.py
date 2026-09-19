"""Bounded non-timed heldout timeout retry; original candidate stays immutable."""
import json
import hashlib
from pathlib import Path
import sys


def retry_config(original):
    assert '--watchdog-timeout' not in original['argv']
    result = json.loads(json.dumps(original).replace('qwen-dynamic-heldout-v2', 'qwen-dynamic-heldout-v3'))
    result['argv'] += ['--watchdog-timeout', '1800']
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapted_controller(source):
    substitutions = {
        "name=f'qwen-dynamic-{a.mode}-v2'": "assert a.mode=='heldout'; name='qwen-dynamic-heldout-v3'",
        "source=frozen/(a.mode+'.json');assert sha(source)==contract['configs_sha256'][a.mode]":
        "source=ROOT/'qwen-dynamic-heldout-retry-v3/heldout.json';assert sha(source)==json.loads((source.parent/'retry.json').read_text())['config_sha256']",
        'run_qwen_dynamic_evaluation.py --mode': 'run_qwen_dynamic_heldout_retry.py --mode',
    }
    for before, after in substitutions.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--prepare':
        frozen, target = map(Path, sys.argv[2:])
        target.mkdir(exist_ok=False)
        (target/'heldout.json').write_text(json.dumps(retry_config(json.loads((frozen/'heldout.json').read_text())), indent=2)+'\n')
        metadata = dict(version='heldout-watchdog-retry-v3', candidate_sha256=sha(frozen/'candidate.json'),
            original_config_sha256=sha(frozen/'heldout.json'), config_sha256=sha(target/'heldout.json'),
            runner_sha256=sha(Path(__file__)), watchdog_seconds=1800, timed=False,
            failed_attempt='qwen-dynamic-heldout-v2', replacement_attempt='qwen-dynamic-heldout-v3',
            restart_all_128=True, threshold_changed=False, corpus_changed=False)
        (target/'retry.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps(metadata));return
    root = Path('/workspace/round8' if '--serve' in sys.argv else '/home/admin/kt-round8-qwen35')
    frozen=root/'qwen-dynamic-candidate-v2';retry=root/'qwen-dynamic-heldout-retry-v3'
    metadata=json.loads((retry/'retry.json').read_text())
    ack=json.loads((retry/'retry-ack.json').read_text())
    assert len(ack['local_commit'])==40 and ack['retry_sha256']==sha(retry/'retry.json')
    assert sha(Path(__file__))==metadata['runner_sha256']
    assert sha(frozen/'candidate.json')==metadata['candidate_sha256']
    assert sha(frozen/'heldout.json')==metadata['original_config_sha256']
    assert sha(retry/'heldout.json')==metadata['config_sha256']
    assert json.loads((retry/'heldout.json').read_text())==retry_config(json.loads((frozen/'heldout.json').read_text()))
    original=root/'tools/run_qwen_dynamic_evaluation.py'
    contract=json.loads((frozen/'candidate.json').read_text())
    assert sha(original)==contract['tools_sha256'][original.name]
    # Reuse the frozen admission, lifecycle, client and resource guards unchanged.
    # Three fail-closed substitutions affect only attempt path/config/entrypoint.
    source=adapted_controller(original.read_text())
    exec(compile(source,str(original)+'[heldout-v3]', 'exec'), {'__name__':'__main__'})


if __name__=='__main__':main()
