"""Require natural token parity; optionally compare reference/full raw logits."""
import argparse
import hashlib
import json
from pathlib import Path


def compare_rows(left, right):
    assert len(left) == len(right), 'REQUEST_COUNT'
    for a, b in zip(left, right):
        assert a['rid'] == b['rid'] and a['input_ids'] == b['input_ids'], 'INPUT_IDENTITY'
        assert a['baseline_output_ids'] == b['baseline_output_ids'], f"TOKEN_PARITY:{a['rid']}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--natural', type=Path, required=True)
    p.add_argument('--full-session', type=Path, required=True)
    p.add_argument('--reference-server', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    control = json.loads(args.natural.read_text())
    assert control['status'] == 'NATURAL_OBSERVER_CONTROL_PASS'
    assert json.loads((args.full_session / 'session.json').read_text())['status'] == 'PREQUALIFICATION_SESSION_PASS'
    assert json.loads((args.full_session / 'final-integrity.json').read_text())['status'] == 'PASS'
    full = json.loads((args.full_session / 'capture/result.json').read_text())
    assert not full['forced'], 'NATURAL_VS_FORCED_FORBIDDEN'
    result = dict(status='FAIL', control_sha256=hashlib.sha256(args.natural.read_bytes()).hexdigest(),
                  full_result_sha256=hashlib.sha256((args.full_session / 'capture/result.json').read_bytes()).hexdigest())
    try:
        compare_rows(control['rows'], full['rows'])
        comparisons = 0
        if args.reference_server:
            assert control['observer'] == 'reference'
            import torch
            torch.set_num_threads(1)
            files = sorted((args.reference_server / 'tensors').rglob('*.pt'))
            steps = [s for row in full['rows'] for s in row['steps']]
            assert len(files) == len(steps)
            for file, step in zip(files, steps):
                data = torch.load(file, map_location='cpu', weights_only=False)
                value = data['capture.next_token_logits'].contiguous()
                assert hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() == step['logits_sha256'], f'LOGITS_PARITY:{file}'
                comparisons += 1
        else:
            assert control['observer'] == 'off'
        result.update(status='PASS', requests=len(control['rows']), full_logits_byte_comparisons=comparisons)
    except BaseException as exc:
        result['failure'] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
