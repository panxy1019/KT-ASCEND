"""OFF-only dev/training pilot before quality protocol freeze; never test ON."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import urllib.request
from owned_server import OwnedServer
from run_compact_controls import validate
from quality_answers import score
from audit_quality_overlap import normalized

DEV_SHA = "77b9fd2d80c2f5882f9710c00c644628b2995f00bc629ff758c03521e4fe1769"
PRIMARY_SHA = "7d501ab27d76ec19ff64aebf1fc5d5c0e70ee64aa829faf99261ec5c111b3d4d"


def select_dev(dev, primary, count):
    forbidden = [normalized(r["question"]) for r in primary]
    gsm = [r for r in dev if r["benchmark"] == "gsm8k"][:count]
    cmmlu, subjects = [], set()
    for row in dev:
        if row["benchmark"] != "cmmlu" or row["subject"] in subjects:
            continue
        question = normalized(row["question"])
        if any(q == question or q in question or question in q for q in forbidden):
            continue
        cmmlu.append(row)
        subjects.add(row["subject"])
        if len(cmmlu) == count:
            break
    assert len(gsm) == len(cmmlu) == count
    return gsm + cmmlu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--count-per-benchmark", type=int, choices=(8,32), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol-v2", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("REFUSE_DEV_EVIDENCE_OVERWRITE")
    assert json.loads((args.runtime/"progress.json").read_text())["status"] == "RUNTIME_PASS"
    files = {"dev.json": DEV_SHA, "primary.json": PRIMARY_SHA}
    scorer = score
    if args.protocol_v2:
        from quality_answers_v2 import score as scorer
        manifest=json.loads((args.inputs/"MANIFEST.json").read_text())
        assert manifest["version"] == "round6.4-quality-draft-v2" and manifest["primary_execution_allowed"] is False
        assert json.loads(Path("/workspace/results/r64-tokenizer-serving-v1/probe-report.json").read_text())["status"] == "TOKENIZER_COMPAT_SERVING_PASS"
        files = manifest["files"]
    for name, sha in files.items():
        assert hashlib.sha256((args.inputs/name).read_bytes()).hexdigest() == sha
    rows = select_dev(json.loads((args.inputs/"dev.json").read_text())["rows"],
                      json.loads((args.inputs/"primary.json").read_text())["rows"], args.count_per_benchmark)
    result = dict(status="RUNNING", purpose="DEV_ONLY_PROTOCOL_CHECK", profile="D1_NODEFER", batch=1,
                  selected_ids=[r["id"] for r in rows], dev_sha256=files["dev.json"], primary_outputs_observed=False,
                  protocol_v2=args.protocol_v2,
                  rows=[], started_unix=time.time())
    def cancel(signum, frame):
        raise InterruptedError("OWNED_DEV_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    try:
        with OwnedServer(args.output, "D1_NODEFER", batch=1, capture=False, tokenizer_compat=args.protocol_v2):
            with (args.output/"requests.jsonl").open("x") as log:
                for row in rows:
                    assert row["split"] in ("dev", "train")
                    rid = "r64-dev-"+row["id"]
                    payload = dict(rid=rid, input_ids=row["input_ids"], sampling_params=dict(
                        temperature=0, max_new_tokens=row["max_new_tokens"], ignore_eos=False),
                        return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                    if args.protocol_v2:
                        payload["sampling_params"]["stop"] = row["stop"]
                    request = urllib.request.Request("http://127.0.0.1:31000/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
                    start = time.monotonic()
                    with urllib.request.urlopen(request, timeout=600) as response:
                        answer = json.load(response)
                    seconds = time.monotonic()-start
                    log.write(json.dumps(dict(payload=payload,response=answer,seconds=seconds))+"\n")
                    log.flush()
                    tokens = validate(answer, rid, len(answer["output_ids"]))
                    assert len(tokens) <= row["max_new_tokens"]
                    assert answer["meta_info"]["prompt_tokens"] == len(row["input_ids"]), "PROMPT_TRUNCATION"
                    finish = answer["meta_info"]["finish_reason"]["type"]
                    assert finish in ("length", "stop"), "REQUEST_FAILURE"
                    scored = scorer(row["benchmark"], answer["text"], row["answer"], truncated=finish == "length")
                    item = dict(id=row["id"], benchmark=row["benchmark"], seconds=seconds,
                                generated_tokens=len(tokens), finish_reason=finish, **scored)
                    result["rows"].append(item)
                    print(json.dumps(item),flush=True)
        result["status"] = "DEV_EXECUTION_COMPLETE_PENDING_PROTOCOL_REVIEW"
    except BaseException as exc:
        result.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        result["ended_unix"] = time.time()
        if args.output.exists():
            (args.output/"dev-report.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
