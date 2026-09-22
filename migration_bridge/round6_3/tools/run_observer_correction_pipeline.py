"""Finite OFF/reference correction only. Does not enter pilot or qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/results')
TOOLS = ROOT / 'round6_3-prequalification/tools'
INPUT = ROOT / 'round6_3-prequalification/long-smoke-input.json'


def main():
    output = ROOT / 'r63-observer-correction-retry1-pipeline'
    output.mkdir(exist_ok=False)
    result = dict(status='STARTED', started_unix=time.time())
    def call(name, *args):
        with (output / 'process.log').open('a') as log:
            # Individual sessions own their server guardians. Do not wrap in a
            # short-grace outer process-group guardian that can orphan servers.
            subprocess.run([sys.executable, str(TOOLS / name), *map(str, args)],
                env=dict(os.environ, KT_R63_OBSERVER_PROFILE='full', KT_R63_COMPACT_CAPTURE='0',
                         PYTHONPATH=str(ROOT / 'round6_2-prequalification/tools')),
                check=True, stdout=log, stderr=subprocess.STDOUT)
    try:
        ready = ROOT / 'r63-natural-off0-retry1-control/result.json'
        deadline = time.monotonic() + 1200
        while True:
            if ready.exists():
                state = json.loads(ready.read_text())['status']
                if state == 'NATURAL_OBSERVER_CONTROL_PASS':
                    break
                if state != 'REQUESTS_COMPLETE':
                    raise ValueError(f'OFF0_FAILED:{state}')
            if time.monotonic() > deadline:
                raise TimeoutError('OFF0_WAIT')
            time.sleep(2)
        for mode in (0, 1):
            full = ROOT / ('r63-long-baseline-session' if mode == 0 else 'r63-natural-full1-session')
            profile = 'D1_NODEFER' if mode == 0 else 'D1'
            if mode == 1:
                call('run_stateful_session.py', '--profile', profile, '--server-root',
                     ROOT / 'r63-natural-full1-server', '--output', full, '--requests', INPUT)
            for observer in ('off', 'reference'):
                prefix = ROOT / ('r63-natural-off0-retry1' if mode == 0 and observer == 'off'
                                 else f'r63-natural-{observer}{mode}')
                server = Path(str(prefix) + '-server')
                control = Path(str(prefix) + '-control')
                if not (mode == 0 and observer == 'off'):
                    call('run_natural_observer_control.py', '--profile', profile, '--observer', observer,
                         '--requests', INPUT, '--server-root', server, '--output', control)
                args = ['--natural', control / 'result.json', '--full-session', full,
                        '--output', ROOT / f'r63-natural-{observer}{mode}-parity.json']
                if observer == 'reference':
                    args += ['--reference-server', server]
                call('compare_natural_observers.py', *args)
        result['status'] = 'OBSERVER_CORRECTION_PASS'
    except BaseException as exc:
        result.update(status='FAIL', failure=repr(exc))
        raise
    finally:
        result['ended_unix'] = time.time()
        (output / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
