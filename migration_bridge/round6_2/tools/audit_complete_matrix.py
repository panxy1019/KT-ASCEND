"""Additional whole-matrix audit: exactly one target/lookahead per request.

Revalidates both forwards' sequence lengths, finite logits, CPU ownership and
terminal transitions using the smoke-tested actual-forward parser. No metric
comparison, reserve fitting, model inference or capture mutation.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from freeze_corpora import canonical_sha
from verify_reserved_inputs import checked


def grouped_pairs(records, request_ids):
    groups = defaultdict(list)
    for record in records:
        groups[record["request_id"]].append(record)
    if set(groups) != set(request_ids):
        raise ValueError("MATRIX_REQUEST_SET_MISMATCH")
    for group in groups.values():
        if Counter(row["kind"] for row in group) != {"TARGET": 1, "SERVING_LOOKAHEAD": 1}:
            raise ValueError("MATRIX_PER_REQUEST_PAIR_MISMATCH")
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse complete-matrix audit overwrite")
    import torch
    from run_free_generation_captured import inspect_capture
    manifest = checked(args.root / "manifest.json")
    audit = json.loads((args.root / "forward_audit.json").read_text())
    if audit["status"] != "PASS" or canonical_sha(audit["rows"]) != manifest["forward_audit_sha256"]:
        raise ValueError("MATRIX_AUDIT_HASH_MISMATCH")
    rows = manifest["rows"]
    if manifest["row_count"] != 8192 or len(rows) != 8192 or manifest["complete_forward_count"] != 16384:
        raise ValueError("MATRIX_COUNT_MISMATCH")
    request_ids = [row["request_id"] for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("MATRIX_DUPLICATE_REQUEST")
    groups = grouped_pairs(audit["rows"], request_ids)
    if len({row["capture_file"] for row in audit["rows"]}) != 16384:
        raise ValueError("MATRIX_DUPLICATE_CAPTURE")
    checked_forwards = 0
    for row in rows:
        pair = {record["kind"]: record for record in groups[row["request_id"]]}
        if pair["TARGET"]["capture_file"] != row["capture_file"]:
            raise ValueError("MATRIX_TARGET_MANIFEST_MISMATCH")
        data = torch.load(row["capture_file"], map_location="cpu", weights_only=False)
        ids = data["capture.forward_batch_info.input_ids"].reshape(-1).tolist()
        if canonical_sha(ids) != row["request_input_hash"]:
            raise ValueError("MATRIX_TARGET_INPUT_HASH_MISMATCH")
        for step, kind in enumerate(("TARGET", "SERVING_LOOKAHEAD")):
            record, _ = inspect_capture(Path(pair[kind]["capture_file"]), dict(input_ids=ids),
                                        [row["serving_token"]], step, row["request_id"],
                                        row["capture_provenance"], manifest["mode"])
            if step == 0:
                if record["logits_sha256"] != row["logits_sha256"]:
                    raise ValueError("MATRIX_TARGET_LOGIT_HASH_MISMATCH")
                if len(record["deferred_routes"]) != row["deferred_event_count_current_row"] + row["deferred_event_count_prefix"]:
                    raise ValueError("MATRIX_TARGET_EXPOSURE_MISMATCH")
            checked_forwards += 1
        if checked_forwards % 1024 == 0:
            print(json.dumps(dict(checked_forwards=checked_forwards)), flush=True)
    result = dict(status="PASS", mode=manifest["mode"], requests=8192, checked_forwards=checked_forwards,
                  target_lookahead_exactly_once=True, all_logits_finite=True,
                  terminal_transition_checks=True, original_manifest_sha256=manifest["sha256"],
                  numerical_metrics_read=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
