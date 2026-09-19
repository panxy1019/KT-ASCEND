"""Freeze hook-free static launches from already qualified Q64/Q32 commands."""
import hashlib
import json
from pathlib import Path


ROOT = Path('/workspace/round8')
SOURCES = {'q64': 'q64-launch-v6', 'q32': 'q32-launch-restart-v1'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for profile, source_name in SOURCES.items():
        source = ROOT / source_name / 'resolved-command.json'
        config = json.loads(source.read_text())
        assert config['resident'] == int(profile[1:])
        assert config['graph'] is config['dynamic'] is config['deferred'] is False
        assert config['batch'] == config['tp'] == 1
        assert config['profile_sha256'] == sha(ROOT / f'{profile}_profile.json')
        args = config['argv'][:]
        hook_index = args.index('--forward-hooks')
        assert hook_index == len(args) - 2
        del args[hook_index:]
        assert '--forward-hooks' not in args
        output = ROOT / f'{profile}-perf-launch-v1'
        output.mkdir(exist_ok=False)
        config['argv'] = args
        config['purpose'] = 'ISOLATED_STATIC_PERFORMANCE_AND_STABILITY'
        config['quality_authorized'] = False
        config['source_command_sha256'] = sha(source)
        config['diagnostic_hooks_enabled'] = False
        (output / 'resolved-command.json').write_text(json.dumps(config, indent=2) + '\n')
        print(profile, sha(output / 'resolved-command.json'), flush=True)


if __name__ == '__main__':
    main()
