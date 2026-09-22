"""Frozen QD3 only: captures -> complete audits -> exposure -> derivation.

Never starts D1 runtime/HD3/quality or promotes a derived candidate itself.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/workspace/results')
TOOLS = ROOT/'round6_3-prequalification/tools'
INPUTS = ROOT/'round6_3-formal-inputs'
OLD = ROOT/'round6_2-prequalification/tools'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from evaluate_stateful_contract import frozen
    prereg = frozen(INPUTS/'PREREGISTRATION.json')
    assert prereg['sha256'] == '19685bff9c72337873c466067ba76c579e4e562b25170d84d46e3d5185f83eb0'
    qd3 = frozen(INPUTS/'qd3.json')
    assert qd3['corpus'] == 'QD3' and qd3['sha256'] == prereg['corpus_sha256']['QD3']
    for path, key in [(OLD/'evaluate_contract_v2.py', 'metrics_module_sha256'),
                      (OLD/'tail_statistics.py', 'statistics_module_sha256'),
                      (TOOLS/'evaluate_stateful_contract.py', 'stateful_evaluator_sha256')]:
        assert sha(path) == prereg[key], f'CODE_DRIFT:{path}'
    unit = ROOT/'r63-formal-adapter-unit/result.json'
    assert json.loads(unit.read_text())['success']
    pilot = ROOT/'r63-pilot-retry1-pipeline/result.json'
    assert json.loads(pilot.read_text())['status'] == 'EXPOSURE_PILOT_COMPLETE'
    budget = json.loads((INPUTS/'QD3_EXECUTION_BUDGET.json').read_text())
    assert budget['preregistration_sha256'] == prereg['sha256']
    assert shutil.disk_usage(ROOT).free >= budget['minimum_free_disk_bytes'], 'DISK_BUDGET'
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', 31000)) != 0, 'EXISTING_SERVER_DO_NOT_TOUCH'
    output = ROOT/'r63-qd3-v1-pipeline'
    output.mkdir(exist_ok=False)
    result = dict(status='STARTED', started_unix=time.time(), budget=budget,
                  preregistration_sha256=prereg['sha256'], qd3_sha256=qd3['sha256'],
                  unit_evidence_sha256=sha(unit), pilot_evidence_sha256=sha(pilot),
                  tool_sha256={p.name:sha(p) for p in TOOLS.glob('*.py')})
    (output/'gate-freeze.json').write_text(json.dumps(result, indent=2))
    def call(script, *args, timeout=1800):
        result['current_step'] = script + ' ' + ' '.join(map(str, args))
        (output/'progress.json').write_text(json.dumps(result, indent=2))
        with (output/'process.log').open('a') as log:
            # Model session has its own bounded client and owned server guard;
            # do not kill an outer process group before its cleanup can run.
            subprocess.run([sys.executable, str(TOOLS/script), *map(str, args)],
                env=dict(os.environ, KT_R63_OBSERVER_PROFILE='full', KT_R63_COMPACT_CAPTURE='1',
                         PYTHONPATH=str(TOOLS)+':'+str(OLD)), check=True,
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    try:
        baseline = ROOT/'r63-qd3-v1-natural-session'
        force0 = ROOT/'r63-qd3-v1-force0-session'
        force1 = ROOT/'r63-qd3-v1-force1-session'
        history = INPUTS/'qd3-force.json'
        for mode, profile, session in [('natural', 'D1_NODEFER', baseline),
                                       ('force0', 'D1_NODEFER', force0), ('force1', 'D1', force1)]:
            args = ['--profile', profile, '--server-root', ROOT/f'r63-qd3-v1-{mode}-server',
                    '--output', session, '--requests', INPUTS/'qd3.json' if mode == 'natural' else history,
                    '--client-timeout-seconds', 3000]
            if mode != 'natural':
                args += ['--forced']
            call('run_stateful_session.py', *args, timeout=None)
            assert json.loads((session/'final-integrity.json').read_text())['status'] == 'PASS'
            if mode == 'natural':
                call('prepare_stateful_smoke.py', '--baseline-result', baseline/'capture/result.json', '--output', history)
            elif mode == 'force0':
                call('compare_observers.py', '--full', baseline/'capture/result.json',
                     '--compact', force0/'capture/result.json', '--output', ROOT/'r63-qd3-v1-self-history-parity.json')
        for mode, session in [('defer0', force0), ('defer1', force1)]:
            call('assemble_stateful_positions.py', '--session', session, '--corpus', INPUTS/'qd3.json',
                 '--forcing', history, '--output', ROOT/f'r63-qd3-v1-{mode}-positions.json')
        call('evaluate_stateful_contract.py', 'derive', '--baseline', ROOT/'r63-qd3-v1-defer0-positions.json',
             '--deferred', ROOT/'r63-qd3-v1-defer1-positions.json', '--corpus', INPUTS/'qd3.json',
             '--preregistration', INPUTS/'PREREGISTRATION.json', '--output', ROOT/'r63-qd3-v1-derivation')
        result['status'] = 'QD3_DERIVED_PENDING_CANDIDATE_COMMIT_AND_RUNTIME'
    except BaseException as exc:
        result.update(status='STOPPED', failure=repr(exc))
        raise
    finally:
        result['ended_unix'] = time.time()
        (output/'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
