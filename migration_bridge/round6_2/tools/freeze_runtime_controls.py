"""Recover exact pre-existing diagnostic inputs; no qualification selection."""
import argparse
import hashlib
import json
from pathlib import Path
from freeze_corpora import canonical_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse diagnostic fixture overwrite")
    old = json.loads(args.legacy_request.read_text())
    ids = [row[1] for row in old["response"]["meta_info"]["input_token_logprobs"]]
    if len(ids) != 54 or canonical_sha(ids) != old["request_input_ids_sha256"]:
        raise ValueError("LEGACY_DIAGNOSTIC_INPUT_IDENTITY_MISMATCH")
    result = dict(version="round6.2-runtime-controls-v1", qualification_corpus=False,
                  selection="reuse historical explicit replay and first historical whole-request no-hit input; no QD2/HD2 outputs used",
                  source=str(args.legacy_request), source_sha256=hashlib.sha256(args.legacy_request.read_bytes()).hexdigest(),
                  cases=dict(explicit_replay=dict(input_ids=ids, input_sha256=canonical_sha(ids)),
                             cpu_not_hit=dict(input_ids=[1000], input_sha256=canonical_sha([1000]))))
    result["sha256"] = canonical_sha(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(path=str(args.output), sha256=result["sha256"])))


if __name__ == "__main__":
    main()
