"""Fresh exposure-only pilot; all observer gates must already be closed."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/results')
FROZEN = ROOT / 'round6_3-prequalification'
TOOLS = FROZEN / 'tools'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-observer', action='store_true')
    args = parser.parse_args()
    if args.wait_observer:
        deadline = time.monotonic() + 2400
        observed = ROOT / 'r63-observer-correction-retry1-pipeline/result.json'
        while not observed.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('OBSERVER_GATE_NOT_FINISHED')
            time.sleep(5)
    gates = [ROOT / 'r63-observer-correction-retry1-pipeline/result.json']
    gates += [ROOT / f'r63-natural-{observer}{mode}-parity.json'
              for mode in (0, 1) for observer in ('off', 'reference')]
    gates += [ROOT / f'r63-long-observer{mode}-parity.json' for mode in (0, 1)]
    for index, gate in enumerate(gates):
        value = json.loads(gate.read_text())
        assert value['status'] == ('OBSERVER_CORRECTION_PASS' if index == 0 else 'PASS'), gate
    pilot = FROZEN / 'pilot-input.json'
    assert hashlib.sha256(pilot.read_bytes()).hexdigest() == '0e309656587683068e515cd34024a360c6c8ca36924e3ee8a6a64d5f5b42e6ef'
    audit = json.loads((FROZEN / 'pilot-token-audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['pilot_sha256'] == hashlib.sha256(pilot.read_bytes()).hexdigest()
    root = ROOT / 'r63-pilot-retry1-pipeline'
    root.mkdir(exist_ok=False)
    report = dict(status='STARTED', started_unix=time.time(),
        gates={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in gates},
        supersedes='premature r63-pilot-*; no old inference data reused')
    (root / 'gate-freeze.json').write_text(json.dumps(report, indent=2))
    def call(name, *args):
        with (root / 'process.log').open('a') as log:
            subprocess.run([sys.executable, str(TOOLS / name), *map(str, args)],
                env=dict(os.environ, KT_R63_OBSERVER_PROFILE='full', KT_R63_COMPACT_CAPTURE='1',
                         PYTHONPATH=str(ROOT / 'round6_2-prequalification/tools')),
                check=True, stdout=log, stderr=subprocess.STDOUT)
    try:
        baseline = ROOT / 'r63-pilot-retry1-baseline-session'
        deferred = ROOT / 'r63-pilot-retry1-defer1-session'
        forcing = FROZEN / 'pilot-retry1-force.json'
        call('run_stateful_session.py', '--profile', 'D1_NODEFER', '--server-root',
             ROOT / 'r63-pilot-retry1-baseline-server', '--output', baseline, '--requests', pilot)
        call('prepare_stateful_smoke.py', '--baseline-result', baseline / 'capture/result.json', '--output', forcing)
        call('run_stateful_session.py', '--profile', 'D1', '--server-root',
             ROOT / 'r63-pilot-retry1-defer1-server', '--output', deferred, '--requests', forcing, '--forced')
        call('assemble_stateful_positions.py', '--session', deferred, '--corpus', pilot,
             '--forcing', forcing, '--output', ROOT / 'r63-pilot-retry1-positions.json')
        call('pilot_sample_size.py', '--pilot', deferred / 'capture/result.json',
             '--output', ROOT / 'r63-pilot-retry1-size-plan.json')
        report['status'] = 'EXPOSURE_PILOT_COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', failure=repr(exc))
        raise
    finally:
        report['ended_unix'] = time.time()
        (root / 'result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
