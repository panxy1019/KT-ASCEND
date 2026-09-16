"""Retain exact Round4A4 MC evaluator as historical smoke, not primary quality.

Round6.4 explicitly replaces the rejected HD2 numerical prerequisite. It does
not modify the old evaluator, manifest, option scoring, or tie behavior.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
from owned_server import OwnedServer

MANIFEST = "c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce"
EVALUATOR = "3d1884744e828668c60704f836ce4265ecf137d1e1da8770f98a69118c3de926"


def checked_manifest(path):
    value = json.loads(Path(path).read_text())
    raw = json.dumps({k:v for k,v in value.items() if k != "sha256"},
                     ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != MANIFEST or value["sha256"] != MANIFEST:
        raise ValueError("HISTORICAL_MANIFEST_DRIFT")
    if len(value["benchmarks"]) != 2 or any(len(x["samples"]) != 128 for x in value["benchmarks"].values()):
        raise ValueError("HISTORICAL_COUNTS")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--evaluator", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    checked_manifest(args.manifest)
    assert hashlib.sha256(args.evaluator.read_bytes()).hexdigest() == EVALUATOR
    sys.path.insert(0, "/workspace/results/round6_2-prequalification/tools")
    from run_quality_checked import validate_response
    spec = importlib.util.spec_from_file_location("historical_frozen_evaluator", args.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original_post = evaluator.post_json
    args.output.mkdir(exist_ok=False)
    report = dict(status="RUNNING", started_unix=time.time(), completed=[],
                  purpose="HISTORICAL_COMPATIBILITY_SMOKE_NOT_PRIMARY", manifest_sha256=MANIFEST,
                  evaluator_sha256=EVALUATOR, primary_quality_qualified=False)
    def cancel(signum, frame):
        raise InterruptedError("OWNED_HISTORICAL_SMOKE_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    try:
        for profile, mode in (("D1_NODEFER", "defer0"), ("D1", "defer1")):
            root = args.output/profile
            with OwnedServer(root, profile, capacity=True, capture=False):
                with (root/"requests.jsonl").open("x") as raw:
                    def post(url, payload):
                        response = original_post(url, payload)
                        raw.write(json.dumps(dict(payload=payload,response=response))+"\n")
                        raw.flush()
                        return validate_response(response)
                    evaluator.post_json = post
                    sys.argv = [str(args.evaluator), "--manifest", str(args.manifest),
                                "--mode", mode, "--output", str(root/"quality.json")]
                    evaluator.main()
            report["completed"].append(profile)
        a, b = [json.loads((args.output/p/"quality.json").read_text()) for p in ("D1_NODEFER","D1")]
        assert [(r["benchmark"],r["sample_id"]) for r in a["rows"]] == [(r["benchmark"],r["sample_id"]) for r in b["rows"]]
        report["off_summary"], report["on_summary"] = a["summaries"], b["summaries"]
        report["discordant_correctness"] = sum(x["correct"] != y["correct"] for x,y in zip(a["rows"],b["rows"]))
        report["status"] = "HISTORICAL_SMOKE_COMPLETE_PENDING_REVIEW"
    except BaseException as exc:
        report.update(status="FAILED",exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output/"smoke-report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
