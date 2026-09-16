"""Offline frozen batch1 quality execution. No fitting, deletion, or early scoring stop.

This tool cannot run against the preparation contract. Both paths must share
the exact pre-output frozen protocol, inputs, parser, and deployed source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import urllib.request
from owned_server import OwnedServer
from quality_answers_v2 import score
from run_compact_controls import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_inputs(protocol_path, inputs_path):
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get("status") != "FROZEN_BEFORE_PRIMARY_OUTPUT" or protocol.get("primary_execution_allowed") is not True:
        raise ValueError("PRIMARY_PROTOCOL_NOT_FROZEN")
    if protocol["margin_absolute"] != 0.02 or protocol["primary_batch"] != 1:
        raise ValueError("APPROVED_SCOPE_DRIFT")
    if sha(inputs_path) != protocol["primary_input_sha256"]:
        raise ValueError("PRIMARY_INPUT_DRIFT")
    for name, digest in protocol["evaluator_sha256"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("EVALUATOR_DRIFT:"+name)
    rows = json.loads(Path(inputs_path).read_text())["rows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("DUPLICATE_PRIMARY_ID")
    counts = {b:sum(r["benchmark"] == b for r in rows) for b in ("gsm8k", "cmmlu")}
    expected = ({"gsm8k":32,"cmmlu":256} if protocol.get("version") == "round6.4-fast-quality-v1" else {"gsm8k":1319,"cmmlu":2048})
    if counts != expected or len(rows) != sum(expected.values()):
        raise ValueError("PRIMARY_SAMPLE_DRIFT")
    for r in rows:
        budget = 512 if r["benchmark"] == "gsm8k" else 32
        if r["max_new_tokens"] != budget or len(r["input_ids"])+budget > 1024:
            raise ValueError("PRIMARY_BUDGET_OR_CAPACITY_DRIFT")
        expected_stop = ["\nQuestion:", "\n\nQuestion:", "\nQ:"] if r["benchmark"] == "gsm8k" else ["\n"]
        if r.get("stop") != expected_stop or r["input_ids"][0] != 100000:
            raise ValueError("PRIMARY_ANSWER_BOUNDARY_OR_BOS_DRIFT")
    return protocol, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--profile", choices=("D1_NODEFER", "D1"), required=True)
    p.add_argument("--off-reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol, rows = frozen_inputs(args.protocol, args.inputs)
    binding = dict(protocol_sha256=sha(args.protocol), primary_sha256=sha(args.inputs))
    prior = None
    if args.profile == "D1":
        if args.off_reference is None:
            raise ValueError("COMPLETE_OFF_REFERENCE_REQUIRED")
        prior = json.loads((args.off_reference/"quality-report.json").read_text())
        if prior["status"] != "PRIMARY_EXECUTION_COMPLETE" or prior["profile"] != "D1_NODEFER" or prior["binding"] != binding:
            raise ValueError("OFF_REFERENCE_NOT_COMPARABLE")
        if [r["id"] for r in prior["rows"]] != [r["id"] for r in rows]:
            raise ValueError("OFF_REFERENCE_ROWS")
    if args.output.exists():
        raise ValueError("REFUSE_QUALITY_EVIDENCE_OVERWRITE")
    result = dict(status="RUNNING", profile=args.profile, binding=binding, batch=1,
                  rows=[], started_unix=time.time(), primary_endpoint="batch1_only")
    def cancel(signum, frame):
        raise InterruptedError("OWNED_QUALITY_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    try:
        with OwnedServer(args.output, args.profile, batch=1, capture=False, timeout=172800, tokenizer_compat=True):
            actual = json.loads((args.output/"provenance.json").read_text())
            for path, digest in protocol["actual_source_sha256"].items():
                if actual["file_sha256"].get(path) != digest:
                    raise ValueError("FROZEN_ACTUAL_SOURCE_DRIFT:"+path)
            with (args.output/"requests.jsonl").open("x") as raw:
                for row in rows:
                    rid = f"r64-quality-{args.profile}-{row['id']}"
                    payload = dict(rid=rid, input_ids=row["input_ids"], sampling_params=dict(
                        temperature=0, max_new_tokens=row["max_new_tokens"], ignore_eos=False, stop=row["stop"]),
                        return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                    req = urllib.request.Request("http://127.0.0.1:31000/generate", data=json.dumps(payload).encode(),
                                                 headers={"Content-Type":"application/json"})
                    start = time.monotonic()
                    with urllib.request.urlopen(req, timeout=900) as response:
                        answer = json.load(response)
                    elapsed = time.monotonic()-start
                    raw.write(json.dumps(dict(id=row["id"], payload=payload, response=answer, seconds=elapsed))+"\n")
                    raw.flush()
                    tokens = validate(answer, rid, len(answer["output_ids"]))
                    assert len(tokens) <= row["max_new_tokens"]
                    assert answer["meta_info"]["prompt_tokens"] == len(row["input_ids"]), "PROMPT_TRUNCATION"
                    finish = answer["meta_info"]["finish_reason"]["type"]
                    assert finish in ("stop", "length"), "REQUEST_FAILURE"
                    item = dict(id=row["id"], benchmark=row["benchmark"], subject=row.get("subject"),
                        generated_tokens=len(tokens), seconds=elapsed, finish_reason=finish,
                        **score(row["benchmark"], answer["text"], row["answer"], truncated=finish == "length"))
                    result["rows"].append(item)
                    # Progress exposes counts/timing, not a sequential hypothesis test.
                    (args.output/"progress.json").write_text(json.dumps(dict(completed=len(result["rows"]), total=len(rows),
                        last_id=row["id"], elapsed_seconds=time.time()-result["started_unix"]))+"\n")
                    print(json.dumps(dict(completed=len(result["rows"]), id=row["id"], seconds=elapsed)), flush=True)
        result["status"] = "PRIMARY_EXECUTION_COMPLETE"
    except BaseException as exc:
        result.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        result["ended_unix"] = time.time()
        if args.output.exists():
            (args.output/"quality-report.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
