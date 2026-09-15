"""Strict sanity adapter around the unchanged frozen Round4A4 evaluator.

Only callable after a successful HD2 report. Saves every raw HTTP response;
does not change prompts, scoring, tie policy, token budget or decoding.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from verify_reserved_inputs import checked


def validate_response(response):
    meta = response["meta_info"]
    rows = meta["input_token_logprobs"]
    if not rows or rows[-1][0] is None:
        raise ValueError("QUALITY_MISSING_CONDITIONAL_LOGPROB")
    # The first input token can legitimately have no conditional probability.
    for index, row in enumerate(rows):
        if row[0] is None and index == 0:
            continue
        if row[0] is None or not math.isfinite(float(row[0])):
            raise ValueError("QUALITY_NONFINITE_INPUT_LOGPROB")
        if type(row[1]) is not int or row[1] < 0:
            raise ValueError("QUALITY_INVALID_TOKEN_ID")
    output = meta.get("output_token_logprobs")
    if not output or any(row[0] is None or not math.isfinite(float(row[0])) for row in output):
        raise ValueError("QUALITY_NONFINITE_OR_MISSING_OUTPUT")
    return response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluator", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--heldout-report", type=Path, required=True)
    p.add_argument("--mode", choices=["defer0", "defer1"], required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:31000")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    heldout = json.loads(args.heldout_report.read_text())
    if heldout.get("action") != "heldout" or heldout.get("status") != "PASS":
        raise ValueError("HD2_NOT_PASSED")
    manifest = checked(args.manifest)
    if manifest["sha256"] != "c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce":
        raise ValueError("QUALITY_MANIFEST_DRIFT")
    if hashlib.sha256(args.evaluator.read_bytes()).hexdigest() != "3d1884744e828668c60704f836ce4265ecf137d1e1da8770f98a69118c3de926":
        raise ValueError("FROZEN_EVALUATOR_DRIFT")
    raw = args.output.with_suffix(".raw.jsonl")
    provenance = args.output.with_suffix(".provenance.json")
    if any(path.exists() for path in (args.output, raw, provenance)):
        raise ValueError("refuse quality artifact overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("frozen_quality", args.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    post = evaluator.post_json
    provenance.write_text(json.dumps(dict(
        mode=args.mode, protocol_changed=False,
        evaluator_sha256=hashlib.sha256(args.evaluator.read_bytes()).hexdigest(),
        manifest_sha256=manifest["sha256"],
        heldout_report_sha256=hashlib.sha256(args.heldout_report.read_bytes()).hexdigest()), indent=2) + "\n")
    with raw.open("x") as stream:
        def checked_post(url, payload):
            response = post(url, payload)
            stream.write(json.dumps(dict(url=url, payload=payload, response=response), ensure_ascii=False) + "\n")
            stream.flush()
            return validate_response(response)
        evaluator.post_json = checked_post
        sys.argv = [str(args.evaluator), "--base-url", args.base_url, "--manifest", str(args.manifest),
                    "--mode", args.mode, "--output", str(args.output)]
        evaluator.main()


if __name__ == "__main__":
    main()
