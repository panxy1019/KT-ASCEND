"""Account for every failed-run capture without qualifying or filtering the run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refuse incident report overwrite')
    import torch
    from assemble_capture import byte_sha
    from run_free_generation_captured import inspect_capture
    metadata = json.loads((args.root / 'run_spec.json').read_text())['provenance']
    rows = {r['request_id']: r for r in
            (json.loads(p.read_text()) for p in (args.root / 'positions').glob('*.json'))}
    paths = sorted(args.captures.rglob('Pass*.pt'))
    counts, ids, extras, failures, inventory = Counter(), [], [], [], []
    for index, path in enumerate(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = dict(path=str(path), sha256=digest)
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            tx = data['capture.transaction']
            rid_list = tx['batch']['request_ids']
            ids.append((tx['pid'], tx['forward_sequence_id']))
            entry.update(forward_id=ids[-1], request_ids=rid_list)
            if tx['provenance'] != metadata:
                raise ValueError('PROVENANCE_MISMATCH')
            for key, sha in tx['tensor_byte_sha256'].items():
                if byte_sha(data[key]) != sha:
                    raise ValueError('TENSOR_HASH_MISMATCH')
            logits = data['capture.next_token_logits']
            if logits.shape != (1, 102400) or not torch.isfinite(logits).all():
                raise ValueError('LOGITS_SHAPE_OR_FINITE')
            rid = rid_list[0] if len(rid_list) == 1 else None
            if rid not in rows:
                entry.update(kind='OUTSIDE_QUALIFICATION',
                    input_ids=data['capture.forward_batch_info.input_ids'].reshape(-1).tolist(),
                    positions=data['capture.forward_batch_info.positions'].reshape(-1).tolist(),
                    seq_lens=data['capture.forward_batch_info.seq_lens'].reshape(-1).tolist(),
                    deferred_events=tx['deferred_events'])
                extras.append(entry.copy())
            else:
                row = rows[rid]
                step = 0 if str(path) == row['capture_file'] else 1
                # Exact input is persisted separately from the derived position metadata.
                target = torch.load(row['capture_file'], map_location='cpu', weights_only=False)
                prompt = {'input_ids': target['capture.forward_batch_info.input_ids'].reshape(-1).tolist()}
                inspect_capture(path, prompt, [row['serving_token']], step, rid, metadata, metadata['mode'])
                counts[(rid, step)] += 1
                entry['kind'] = 'TARGET' if step == 0 else 'SERVING_LOOKAHEAD'
        except Exception as exc:
            failures.append(dict(path=str(path), error=str(exc)))
        inventory.append(entry)
        if (index + 1) % 1024 == 0:
            print(json.dumps({'inspected': index + 1}), flush=True)
    pair_failures = [rid for rid in rows if counts[(rid, 0)] != 1 or counts[(rid, 1)] != 1]
    ordered = sorted(ids)
    contiguous = len({pid for pid, seq in ids}) == 1 and [seq for pid, seq in ordered] == list(range(len(paths)))
    result = dict(purpose='INCIDENT_FORENSICS_ONLY', qualification_status='INVALID_ORIGINAL_RUN',
                  capture_count=len(paths), qualification_requests=len(rows), extras=extras,
                  checked_qualification_forwards=sum(counts.values()), failures=failures,
                  pair_failures=pair_failures, contiguous_forward_ids=contiguous,
                  incomplete_files=[str(p) for p in args.captures.rglob('*.incomplete')],
                  numerical_comparison_performed=False, inventory=inventory)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inventory'}), flush=True)


if __name__ == '__main__':
    main()
