"""Audit every committed forward, including serving's one-token lookahead."""
import argparse
import json
from pathlib import Path
import torch
from assemble_capture import byte_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.root / "smoke_result.json").read_text())
    requests = {r["request"]["request_id"]: r["request"] for r in result["rows"]}
    records = []
    for path in sorted((args.root / "tensors").rglob("Pass*.pt")):
        data = torch.load(path, weights_only=False, map_location="cpu")
        tx = data["capture.transaction"]
        ids = tx["batch"]["request_ids"]
        if len(ids) != 1 or ids[0] not in requests:
            raise ValueError("UNEXPECTED_FORWARD_REQUEST")
        request = requests[ids[0]]
        for name, expected in tx["tensor_byte_sha256"].items():
            if byte_sha(data[name]) != expected:
                raise ValueError("TRANSACTION_HASH_MISMATCH")
        inputs = data["capture.forward_batch_info.input_ids"].reshape(-1).tolist()
        positions = data["capture.forward_batch_info.positions"].reshape(-1).tolist()
        if inputs == request["input_ids"] and positions == list(range(len(inputs))):
            kind = "REQUEST_TARGET"
        elif inputs == [request["output_token"]] and positions == [len(request["input_ids"])]:
            kind = "SERVING_LOOKAHEAD_NOT_USED_FOR_REQUEST_LOGIT"
        else:
            raise ValueError("UNKNOWN_FORWARD_HISTORY")
        records.append(dict(pid=tx["pid"], sequence=tx["forward_sequence_id"],
                            request_id=ids[0], kind=kind, path=str(path)))
    seq = [(row["pid"], row["sequence"]) for row in records]
    if seq != sorted(set(seq)) or len({pid for pid, _ in seq}) != 1:
        raise ValueError("SEQUENCE_CORRUPTION")
    if [index for _, index in seq] != list(range(len(seq))):
        raise ValueError("MISSING_FORWARD_SEQUENCE")
    for rid in requests:
        targets = [r for r in records if r["request_id"] == rid and r["kind"] == "REQUEST_TARGET"]
        if len(targets) != 1:
            raise ValueError("REQUEST_TARGET_NOT_EXACTLY_ONCE")
    if list((args.root / "tensors").rglob("*.incomplete")):
        raise ValueError("UNCOMMITTED_FORWARD")
    report = dict(status="PASS", requests=len(requests), forwards=len(records), records=records)
    output = args.root / "all_forward_audit.json"
    if output.exists():
        raise ValueError("refuse audit overwrite")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
