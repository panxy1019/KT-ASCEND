"""Read-only causal transition audit; does not require optional legacy submit JSONL."""
import argparse
import json
from pathlib import Path


def validate_transition(events, deferred_routes):
    required = {}
    for layer in (7, 8):
        for kind in ("pending_state", "sync", "sync_complete"):
            matches = [(i, e) for i, e in enumerate(events) if e["event"] == kind and e.get("layer_idx") == layer]
            if len(matches) != 1:
                raise ValueError("MISSING_OR_DUPLICATE_TRANSITION")
            required[layer, kind] = matches[0]
    pending7 = [7] if deferred_routes else []
    if required[7, "pending_state"][1]["pending_layers"] != pending7:
        raise ValueError("PRODUCER_PENDING_ROUTE_MISMATCH")
    if required[8, "pending_state"][1]["pending_layers"] != []:
        raise ValueError("TERMINAL_PENDING_NOT_EMPTY")
    for layer in (7, 8):
        allowance = int(bool(deferred_routes)) if layer == 7 else 0
        for kind in ("sync", "sync_complete"):
            if required[layer, kind][1]["allow_pending"] != allowance:
                raise ValueError("SYNC_ALLOWANCE_MISMATCH")
    consumes = [(i, e) for i, e in enumerate(events) if e["event"] == "previous_pending_consumed"]
    if len(consumes) != int(bool(deferred_routes)):
        raise ValueError("CONSUMER_EXACTLY_ONCE_MISMATCH")
    order = [required[7, kind][0] for kind in ("pending_state", "sync", "sync_complete")]
    if consumes:
        index, consume = consumes[0]
        if (consume["producer_layer"], consume["consumer_layer"]) != (7, 8):
            raise ValueError("PRODUCER_CONSUMER_ID_MISMATCH")
        order.append(index)
    order += [required[8, kind][0] for kind in ("pending_state", "sync", "sync_complete")]
    if order != sorted(set(order)):
        raise ValueError("TRANSITION_ORDER_MISMATCH")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse transition audit overwrite")
    import torch
    manifest = json.loads(args.manifest.read_text())
    count, failures = 0, []
    for row in manifest["rows"]:
        data = torch.load(row["capture_file"], map_location="cpu", weights_only=False)
        try:
            validate_transition(data["capture.transaction"]["deferred_events"],
                                row["deferred_event_count_current_row"] + row["deferred_event_count_prefix"])
        except ValueError as error:
            failures.append(dict(position_id=row["position_id"], capture_file=row["capture_file"], error=str(error)))
        count += 1
    report = dict(status="PASS" if not failures else "FAIL", count=count, failures=failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
