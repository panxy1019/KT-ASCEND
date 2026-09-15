"""Observer-OFF compact regression; not full D1 qualification or performance.

Run inside the project container with CANN environment initialized. Each
server is owned by the retained process-group guard. No other task is stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

OLD = Path('/workspace/results/round6_2-prequalification/tools')
sys.path.insert(0, str(OLD))
# Same validators locally and on A3, no old numerical candidate gate imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'round6_2/tools'))
from run_runtime_determinism import validate
from verify_reserved_inputs import checked

S6_SHA = 'e51af126389507e03b58ae37e5b39c912ece14d54c1daa0ba4baea634ff8f9c5'
PROMPTS = ('s6_en_01', 's6_zh_01', 's6_math_01', 's6_json_01')
PROFILES = ('P1_B0', 'P2_B0', 'P3_B0', 'D1_NODEFER', 'D1')


def ensure_free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 31000))


def check_corpus(path):
    corpus = checked(path)
    if corpus['sha256'] != S6_SHA:
        raise ValueError('S6_DRIFT')
    selected = {p['id']: p for p in corpus['prompts'] if p['id'] in PROMPTS}
    if set(selected) != set(PROMPTS):
        raise ValueError('S6_SELECTION_MISSING')
    return selected


def one_profile(profile, root, selected, env):
    root.mkdir()
    (root / 'tensors').mkdir()  # empty readiness control, no observer
    py = sys.executable
    ensure_free_port()
    command = [py, str(OLD / 'run_owned_process.py'), '--output', str(root / 'guard'),
               '--timeout-seconds', '1800', '--', 'taskset', '-c', '0-15', 'bash',
               '/workspace/kt-src/migration_bridge/round6/tools/launch_round6_profile.sh',
               profile, str(root), '31000']
    report = dict(profile=profile, status='STARTING', started_unix=time.time(),
                  observer='OFF', corpus_sha256=S6_SHA, command=command)
    with (root / 'launcher.log').open('x') as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['owned_guard_pid'] = server.pid
        try:
            subprocess.run([py, str(OLD / 'wait_capture_ready.py'), '--log', str(root / 'guard/process.log'),
                '--captures', str(root / 'tensors'), '--output', str(root / 'readiness.json')],
                env=env, check=True, timeout=920)
            references = None
            if profile.endswith('_B0'):
                historical = Path('/workspace/results') / f'b0-{profile[:2].lower()}-final/result.json'
                prior = json.loads(historical.read_text())
                if prior['corpus_sha256'] != S6_SHA:
                    raise ValueError('HISTORICAL_CORPUS_DRIFT')
                report['historical_reference_sha256'] = hashlib.sha256(historical.read_bytes()).hexdigest()
                references = {r['prompt_id']: r['reference'] for r in prior['rows']}
            rows = []
            with (root / 'requests.jsonl').open('x') as raw:
                for prompt_id in PROMPTS:
                    baseline = None
                    for index, budget in enumerate((32, 32, 8, 16, 32)):
                        if server.poll() is not None:
                            raise RuntimeError('OWNED_SERVER_EXITED')
                        rid = f'r64-compact-{profile}-{prompt_id}-{index}'
                        payload = dict(rid=rid, input_ids=selected[prompt_id]['input_ids'],
                            sampling_params=dict(temperature=0, max_new_tokens=budget, ignore_eos=True),
                            return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                        request = urllib.request.Request('http://127.0.0.1:31000/generate',
                            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
                        begin = time.monotonic()
                        with urllib.request.urlopen(request, timeout=180) as response:
                            answer = json.load(response)
                        raw.write(json.dumps(dict(payload=payload, response=answer,
                            seconds=time.monotonic()-begin)) + '\n')
                        raw.flush()
                        tokens = validate(answer, rid, budget)
                        if baseline is None:
                            baseline = tokens
                        if tokens != baseline[:budget]:
                            raise ValueError(f'SAME_PATH_NONDETERMINISM:{rid}')
                        if references is not None and tokens != references[prompt_id][:budget]:
                            raise ValueError(f'NONDEFER_REFERENCE_MISMATCH:{rid}')
                        print(json.dumps(dict(profile=profile, rid=rid, exact=True)), flush=True)
                    rows.append(dict(prompt_id=prompt_id, reference32=baseline))
            if list((root / 'tensors').iterdir()):
                raise ValueError('OBSERVER_OFF_CONTAMINATION')
            report.update(status='COMPACT_PASS', rows=rows, request_count=20,
                not_proven='Full runtime, no-hit, expert numerical, quality and speedup are separate gates.')
        except BaseException as exc:
            report.update(status='FAIL', exception=repr(exc))
            raise
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
                server.wait(timeout=60)
            report.update(ended_unix=time.time(), guard_returncode=server.returncode)
            guard = root / 'guard/result.json'
            if guard.exists():
                report['guard'] = json.loads(guard.read_text())
            if report['status'] == 'COMPACT_PASS' and report.get('guard', {}).get('reason') != 'CANCELLED':
                report.update(status='FAIL', exception='UNEXPECTED_SERVER_TERMINATION')
            (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] != 'COMPACT_PASS':
        raise RuntimeError(report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    args = p.parse_args()
    if args.output.parent != Path('/workspace/results') or not args.output.name.startswith('r64-'):
        raise ValueError('DEDICATED_RESULT_ROOT_REQUIRED')
    selected = check_corpus(args.corpus)
    args.output.mkdir(exist_ok=False)
    def cancelled(signum, frame):
        raise InterruptedError(f'COMPACT_CANCELLED:{signum}')
    signal.signal(signal.SIGTERM, cancelled)
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith(('KT_R61_', 'KT_R62_', 'KT_R63_', 'KT_DEBUG_')):
            del env[name]
    env.update(ASCEND_RT_VISIBLE_DEVICES='0', SGLANG_APPLY_CONFIG_BACKUP='none',
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION='false', KT_R6_ENABLE_TENSOR_DUMP='0',
        KT_R6_ENABLE_ROUTE_RECORDER='1')  # matches historical cache-disabled controls
    results = []
    try:
        for profile in PROFILES:
            results.append(one_profile(profile, args.output / profile, selected, env))
    finally:
        (args.output / 'progress.json').write_text(json.dumps(dict(
            completed=[r['profile'] for r in results], expected=list(PROFILES),
            status='COMPACT_PASS' if len(results) == len(PROFILES) else 'INCOMPLETE'), indent=2)+'\n')


if __name__ == '__main__':
    main()
