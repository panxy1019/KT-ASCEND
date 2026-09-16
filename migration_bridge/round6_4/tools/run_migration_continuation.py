"""Continue target migration after compact16; stop on the first hard failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compact16', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tools = Path(__file__).parent
    state = dict(status='WAITING_COMPACT16', completed=[], started_unix=time.time())
    def save():
        (args.output/'progress.json').write_text(json.dumps(state, indent=2)+'\n')
    def call(name, command):
        state.update(status='RUNNING', active=name)
        save()
        with (args.output/(name+'.log')).open('x') as log:
            subprocess.run(command, check=True, timeout=18000, stdout=log, stderr=subprocess.STDOUT)
        state['completed'].append(name)
        save()
    try:
        save()
        deadline = time.monotonic()+10800
        while not (args.compact16/'report.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('COMPACT16_NOT_FINISHED')
            time.sleep(15)
        prior = json.loads((args.compact16/'report.json').read_text())
        assert prior['status'] == 'MIGRATION_COMPACT_PASS' and len(prior['rows']) == 32
        state['completed'].append('compact16')
        call('real_expert_safety', ['bash', str(tools/'run_real_expert_safety.sh'),
                                  '/workspace/results/r64-910b-expert-safety-v1'])
        expert = json.loads(Path('/workspace/results/r64-910b-expert-safety-v1/cases/report.json').read_text())
        assert expert['status'] == 'EXPERT_SAFETY_PASS' and len(expert['cases']) == 16
        root = '/workspace/results/r64-910b-compact32-v1'
        call('compact32', [sys.executable, str(tools/'run_migration_compact.py'),
                          '--corpus', '/workspace/results/corpora-v2/s6.json',
                          '--output', root, '--cpu-workers', '32'])
        compact = json.loads((Path(root)/'report.json').read_text())
        assert compact['status'] == 'MIGRATION_COMPACT_PASS' and len(compact['rows']) == 32
        root = '/workspace/results/r64-910b-descriptive32-v1'
        call('descriptive32', [sys.executable, str(tools/'run_descriptive_holdout.py'),
                              '--inputs', '/workspace/results/round6_3-formal-inputs',
                              '--entry-audit', str(tools.parent/'fixtures/descriptive_entry.json'),
                              '--output', root, '--cpu-workers', '32'])
        call('descriptive_analysis', [sys.executable, str(tools/'analyze_descriptive_holdout.py'), '--root', root])
        state['status'] = 'MIGRATION_AND_DESCRIPTIVE_COMPLETE_PENDING_FINAL_REVIEW'
    except BaseException as exc:
        state.update(status='STOPPED_FIRST_FAILURE', exception=repr(exc))
        raise
    finally:
        state['ended_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
