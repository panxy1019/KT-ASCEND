"""Route-only audit of frozen QD2 operands; never compares model logits.

Loads mmap-backed captures, touches only routing tensors. Ambiguous top5 ties
are checked against necessary/possible protection sets, not CPU tie-breaking.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(data):
    import torch
    ids = data["r62.ktep.layer7.topk_ids"].long()
    weights = data["r62.ktep.layer7.routing_weights"]
    immediate = data["r62.cpu.layer7.immediate_ids"]
    deferred = data["r62.cpu.layer7.deferred_ids"]
    if ids.shape != weights.shape or ids.shape != immediate.shape or ids.shape != deferred.shape:
        raise ValueError("SELECTOR_SHAPE")
    if ids.shape[1] != 6 or not torch.isfinite(weights).all():
        raise ValueError("SELECTOR_OPERANDS")
    if not torch.equal(weights, data["r62.cpu.layer7.routing_weights"]):
        raise ValueError("SELECTOR_WEIGHT_TRANSFER")
    if not torch.all((immediate == ids) ^ (deferred == ids)):
        raise ValueError("SELECTOR_PARTITION")
    if not torch.all((immediate == ids) | (immediate == -1)) or not torch.all((deferred == ids) | (deferred == -1)):
        raise ValueError("SELECTOR_INVALID_MASK_ID")
    ordered = weights.sort(dim=-1, descending=True).values
    threshold = ordered[:, 4:5]
    tied = ordered[:, 4] == ordered[:, 5]
    # On strict rows all top5 IDs are certain. On tied rows only > threshold
    # are mandatory. >= threshold defines a conservative possible set.
    certain_mask = (weights > threshold) | ((weights >= threshold) & ~tied[:, None])
    certain = set(ids[certain_mask].tolist())
    possible = set(ids[weights >= threshold].tolist())
    observed = set(ids[immediate == ids].tolist())
    if not certain <= observed <= possible:
        raise ValueError("SELECTOR_UNION_MISMATCH")
    for expert in set(ids.reshape(-1).tolist()):
        mask = ids == expert
        if not bool(torch.all((immediate[mask] == expert) == (expert in observed))):
            raise ValueError("SELECTOR_NOT_GLOBAL_UNION")
    owned = torch.zeros_like(ids, dtype=torch.bool)
    for expert in (16, 38, 41, 50):
        owned |= ids == expert
    row_low = (weights < threshold) & ~tied[:, None]
    local_cpu_candidate = row_low & owned
    suppressed = local_cpu_candidate & (immediate == ids)
    terminal_ids = data["r62.ktep.layer8.topk_ids"]
    if not torch.equal(terminal_ids, data["r62.cpu.layer8.immediate_ids"]):
        raise ValueError("SELECTOR_TERMINAL_NOT_IMMEDIATE")
    return dict(rows=int(ids.shape[0]), tied_rows=int(tied.sum()),
                strict_union_forward=int(not tied.any()),
                protected_union_size=len(observed),
                cpu_routes=int(owned.sum()),
                unique_row_low_cpu_candidates=int(local_cpu_candidate.sum()),
                cross_row_protected_cpu_candidates=int(suppressed.sum()),
                actual_deferred_cpu_routes=int((owned & (deferred == ids)).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    files = sorted(args.captures.glob("Pass*.pt"))
    if len(files) != 16384:
        raise ValueError("EXPECTED_FROZEN_RETRY_16384")
    totals, rows = Counter(), []
    for index, path in enumerate(files):
        if index % 2:
            continue  # frozen QD2 target, not serving lookahead
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        tx = data["capture.transaction"]
        if tx["forward_sequence_id"] != index:
            raise ValueError("FORWARD_SEQUENCE_DRIFT")
        metrics = audit(data)
        totals.update(metrics)
        rows.append(dict(file=path.name, rid=tx["batch"]["request_ids"], **metrics))
        if len(rows) % 512 == 0:
            print(json.dumps(dict(audited=len(rows), totals=dict(totals))), flush=True)
    report = dict(status="PASS", source=str(args.captures), target_forwards=len(rows),
                  numerical_metrics_read=False, totals=dict(totals), rows=rows,
                  tie_policy="certain/possible union; no CPU tie-order substitution",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
