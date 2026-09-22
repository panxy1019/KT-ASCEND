"""Natural greedy OFF/reference control, never forces history or fits bounds."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['D1', 'D1_NODEFER'], required=True)
    p.add_argument('--observer', choices=['off', 'reference'], required=True)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--server-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.requests.read_text())
    here = Path(__file__).resolve().parent
    old = Path('/workspace/results/round6_2-prequalification/tools')
    env = dict(os.environ, KT_R63_OBSERVER_PROFILE=args.observer,
               PYTHONPATH=str(old))
    rows = []
    result = dict(status='STARTING', observer=args.observer, profile=args.profile,
                  input_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                  started_unix=time.time(), rows=rows)
    def cancelled(signum, frame):
        raise InterruptedError(f'OWNED_CONTROL_CANCELLED:{signum}')
    signal.signal(signal.SIGTERM, cancelled)
    with (args.output / 'launcher.log').open('x') as log:
        server = subprocess.Popen(['bash', str(here / 'launch_stateful_smoke.sh'),
            args.profile, str(args.server_root)], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            subprocess.run([sys.executable, str(old / 'wait_capture_ready.py'),
                '--log', str(args.server_root / 'guard/process.log'), '--captures',
                str(args.server_root / 'tensors'), '--output', str(args.server_root / 'readiness.json')],
                env=env, check=True, timeout=1000)
            for request in frozen['requests']:
                budget = frozen['token_budget']
                payload = dict(rid=request['rid'], input_ids=request['input_ids'],
                    sampling_params=dict(temperature=0, max_new_tokens=budget, ignore_eos=True),
                    return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                http = urllib.request.Request('http://127.0.0.1:31000/generate',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                begin = time.monotonic()
                with urllib.request.urlopen(http, timeout=600) as response:
                    answer = json.load(response)
                (args.output / (request['rid'] + '.response.json')).write_text(json.dumps(answer))
                assert answer['meta_info']['id'] == request['rid']
                assert len(answer['output_ids']) == budget
                probabilities = answer['meta_info']['output_token_logprobs']
                assert len(probabilities) == budget and all(math.isfinite(x[0]) for x in probabilities)
                rows.append(dict(rid=request['rid'], input_ids=request['input_ids'],
                    baseline_output_ids=answer['output_ids'], seconds=time.monotonic()-begin))
                print(json.dumps(dict(rid=request['rid'], observer=args.observer, status='HTTP_PASS')), flush=True)
            result['status'] = 'REQUESTS_COMPLETE'
        except BaseException as exc:
            result.update(status='FAILED', exception=repr(exc))
            raise
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
                server.wait(timeout=90)
            result.update(ended_unix=time.time(), launcher_returncode=server.returncode)
            (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    guard = json.loads((args.server_root / 'guard/result.json').read_text())
    assert guard['reason'] == 'CANCELLED', guard
    files = sorted((args.server_root / 'tensors').rglob('*.pt'))
    assert not list((args.server_root / 'tensors').rglob('*.incomplete'))
    if args.observer == 'off':
        assert not files and not (args.server_root / 'index.jsonl').exists()
    else:
        import torch
        torch.set_num_threads(1)
        assert len(files) == len(rows) * (frozen['token_budget'] + 1)
        assert len({file.parent for file in files}) == 1
        for sequence, file in enumerate(files):
            data = torch.load(file, map_location='cpu', weights_only=False)
            tx = data['capture.transaction']
            row = rows[sequence // (frozen['token_budget'] + 1)]
            step = sequence % (frozen['token_budget'] + 1)
            assert tx['reference_logits_only'] and tx['forward_sequence_id'] == sequence
            assert tx['batch']['request_ids'] == [row['rid']]
            logits = data['capture.next_token_logits']
            assert tuple(logits.shape) == (1, 102400) and torch.isfinite(logits).all()
            if step < frozen['token_budget']:
                assert logits[0, row['baseline_output_ids'][step]] == logits.max()
    result.update(status='NATURAL_OBSERVER_CONTROL_PASS', physical_captures=len(files))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
