"""Separate, bounded performance watchdog retry; immutable v2 evidence retained."""
import copy
import json
from pathlib import Path
import sys
from qwen_dynamic_retry_release import audit, frozen_source, replace_once
from run_qwen_dynamic_heldout_retry import sha


def retry_config(original):
    assert not any(a.startswith('--watchdog-timeout') for a in original['argv'])
    result = copy.deepcopy(original)
    result['argv'] += ['--watchdog-timeout', '1800']
    return result


def adapted_controller(source):
    source = replace_once(source, "NAME='qwen-dynamic-perf-v2'", "NAME='qwen-dynamic-perf-v3'")
    source = replace_once(source, "assert '--forward-hooks' not in cfg['argv']",
                          "cfg=retry_config(cfg); assert '--forward-hooks' not in cfg['argv']")
    source = replace_once(source, 'run_qwen_dynamic_perf.py --serve',
                          'run_qwen_dynamic_perf_retry.py --serve')
    source = replace_once(source, "'/workspace/round8/tools/qwen_dynamic_perf.py',",
                          "'/workspace/round8/tools/run_qwen_dynamic_perf_retry.py','--client',")
    return replace_once(source, "sha(ROOT/f'qwen-dynamic-{mode}-v2/summary.json')",
        "sha(ROOT/('qwen-dynamic-heldout-v3' if mode=='heldout' else 'qwen-dynamic-cmmlu-v2')/'summary.json')")


def prepare(root):
    target = root/'qwen-dynamic-perf-retry-v3'
    target.mkdir(exist_ok=False)
    original = root/'qwen-dynamic-candidate-v2'
    configs = {}
    for profile in ('static', 'dynamic'):
        path = target/f'perf-{profile}.json'
        path.write_text(json.dumps(retry_config(json.loads((original/path.name).read_text())), indent=2)+'\n')
        configs[profile] = sha(path)
    metadata = dict(candidate_sha256=sha(original/'candidate.json'),
        runner_sha256=sha(Path(__file__)), configs_sha256=configs,
        watchdog_seconds=1800, failed_attempt='qwen-dynamic-perf-v2',
        replacement_attempt='qwen-dynamic-perf-v3', restart_all_cells=True,
        workload_changed=False, timed_boundary_changed=False)
    (target/'retry.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))


def main():
    root = Path('/workspace/round8' if '--serve' in sys.argv or '--client' in sys.argv
                else '/home/admin/kt-round8-qwen35')
    if sys.argv[1:] == ['--prepare']:
        prepare(root)
        return
    target = root/'qwen-dynamic-perf-retry-v3'
    metadata = json.loads((target/'retry.json').read_text())
    ack = json.loads((target/'retry-ack.json').read_text())
    assert len(ack['local_commit']) == 40 and ack['retry_sha256'] == sha(target/'retry.json')
    assert metadata['runner_sha256'] == sha(Path(__file__))
    assert metadata['candidate_sha256'] == sha(root/'qwen-dynamic-candidate-v2/candidate.json')
    for profile in ('static', 'dynamic'):
        name = f'perf-{profile}.json'
        assert sha(target/name) == metadata['configs_sha256'][profile]
        assert json.loads((target/name).read_text()) == retry_config(
            json.loads((root/'qwen-dynamic-candidate-v2'/name).read_text()))
    if '--client' in sys.argv:
        sys.argv.remove('--client')
        source = replace_once(frozen_source(root, 'qwen_dynamic_perf.py'),
                              "'qwen-dynamic-perf-v2'", "'qwen-dynamic-perf-v3'")
    else:
        if '--serve' not in sys.argv:
            assert json.loads((root/'qwen-dynamic-perf-v2/state.json').read_text())['status'] == 'FAILED'
            expected = audit(root)['release_proposal']
            release = json.loads((root/'qwen-dynamic-candidate-v2/qualification-release.json').read_text())
            assert all(release[k] == v for k, v in expected.items())
            pause = Path('/home/admin/kt-round9-qwen38')
            assert (pause/'PAUSE_DOWNLOAD').exists()
            assert json.loads((pause/'download-state.json').read_text())['status'] == 'PAUSED'
        source = adapted_controller(frozen_source(root, 'run_qwen_dynamic_perf.py'))
    exec(compile(source, 'frozen_performance[watchdog-v3]', 'exec'),
         {'__name__': '__main__', 'retry_config': retry_config})


if __name__ == '__main__':
    main()
