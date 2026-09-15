"""Offline original-tensor/prefix verification; no inference or refitting."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from assemble_capture import byte_sha
    from audit_deferred_transitions import validate_transition
    torch.set_num_threads(1)
    root = args.root
    report_path = root/'r63-qd3-v1-derivation/report.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'REJECTED'
    manifests = [json.loads((root/f'r63-qd3-v1-{mode}-positions.json').read_text())
                 for mode in ('defer0', 'defer1')]
    indices = [{(r['prompt_id'],r['token_index']):r for r in m['rows']} for m in manifests]
    output = []
    for failure in report['failures']:
        rid, t = failure['prompt_id'], failure['token_index']
        metric = next(r for r in report['rows'] if r['prompt_id']==rid and r['token_index']==t)
        checks, logits, total_events = 0, [], [0,0]
        for step in range(t+1):
            rows = [index[(rid,step)] for index in indices]
            for key in ('request_input_hash','history_hash','baseline_history_sha256','actual_input_ids','actual_positions'):
                assert rows[0][key] == rows[1][key], (rid,step,key)
            for mode, row in enumerate(rows):
                data = torch.load(row['capture_file'], map_location='cpu', weights_only=False, mmap=True)
                tx = data['capture.transaction']
                assert tx['batch']['request_ids'] == [rid]
                for key, expected in tx['tensor_byte_sha256'].items():
                    assert byte_sha(data[key]) == expected, (rid,step,key)
                value = data['capture.next_token_logits'][0]
                assert byte_sha(value) == row['logits_sha256']
                assert torch.isfinite(value).all()
                count = 0
                for layer, owned in ((7,(16,38,41,50)),(8,(8,25,33,55))):
                    delayed = data.get(f'r62.cpu.layer{layer}.deferred_ids')
                    if delayed is not None:
                        # Partition tensors can contain non-CPU expert IDs;
                        # only CPU-owned, selected routes submit CPU work.
                        top = data[f'r62.ktep.layer{layer}.topk_ids']
                        n = sum(int(top[r,k]) in owned and int(delayed[r,k]) == int(top[r,k])
                                for r in range(top.shape[0]) for k in range(top.shape[1]))
                        assert layer == 7 or n == 0
                        count += n
                validate_transition(tx['deferred_events'], count)
                total_events[mode] += count
                checks += 1
                if step == t:
                    logits.append(value.double())
        for mode, key in enumerate(('baseline_candidate_logits','deferred_candidate_logits')):
            assert logits[mode][metric['candidate_ids']].tolist() == metric[key]
        pairs = []
        bound = report['reserves']['contraction']['threshold']
        for i in metric['candidate_ids']:
            for j in metric['candidate_ids']:
                b, d = (float(x[i]-x[j]) for x in logits)
                if i != j and b > bound and d <= 0:
                    pairs.append(dict(ids=[i,j], baseline=b, deferred=d))
        assert len(pairs) == metric['stable_pair_flips']
        output.append(dict(prompt_id=rid,token_index=t,capture_checks=checks,
            prefix_deferred_routes_by_mode=total_events,pairs=pairs,
            deferred_history=indices[1][(rid,t)]['deferred_route_records']))
    result = dict(status='FORENSIC_VERIFICATION_PASS_NOT_QUALIFICATION',
        original_status='REJECTED', report_raw_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        checks=output, limitations='Existing trace/hash assertions only; not an independent proof of arithmetic or buffer lifetime')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
