"""Fixed atomic cohorts with native generation; no multi-RID teacher forcing.

Full transaction coverage is audited after owned process shutdown. Captures
use the retained compact observer, so this is not a performance measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import urllib.request
from owned_server import OwnedServer
from run_compact_controls import checked, validate

MANIFEST_SHA = "4fdb040ef1d2b100688761c931331705cb7a8c9d0223c96c09d4e704f4b7b24a"


def response_map(response, rids, budget):
    if not isinstance(response, list) or len(response) != len(rids):
        raise ValueError("BATCH_RESPONSE_COUNT")
    mapped = {r["meta_info"]["id"]:r for r in response}
    if len(mapped) != len(rids) or set(mapped) != set(rids):
        raise ValueError("BATCH_RID_MISMATCH")
    return {rid:validate(mapped[rid], rid, budget) for rid in rids}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--batch", type=int, choices=(2,4), required=True)
    p.add_argument("--profile", choices=("D1", "D1_NODEFER"), required=True)
    p.add_argument("--repeats", type=int, choices=(1,2,10), required=True)
    p.add_argument("--diagnostic-full-sequences", help="Comma-separated extra full snapshots; diagnostic only, never performance")
    p.add_argument("--restart-reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeats == 2 and not args.diagnostic_full_sequences:
        raise ValueError("TWO_REPEATS_ARE_DIAGNOSTIC_ONLY")
    if args.diagnostic_full_sequences:
        assert all(s.isdigit() for s in args.diagnostic_full_sequences.split(","))
    if args.output.exists():
        raise ValueError("REFUSE_BATCH_EVIDENCE_OVERWRITE")
    assert json.loads((args.runtime/"progress.json").read_text())["status"] == "RUNTIME_PASS"
    manifest = checked(args.manifest)
    assert manifest["sha256"] == MANIFEST_SHA
    prompts = {r["id"]:r for r in manifest["requests"]}
    cohorts = manifest["cohorts"][str(args.batch)]
    references = {}
    if args.restart_reference:
        previous = json.loads(args.restart_reference.read_text())
        assert previous["status"] == "BATCH_CAPTURE_GENERATION_PASS" and previous["repeats"] == 10
        assert previous["batch"] == args.batch and previous["profile"] == args.profile and previous["manifest_sha256"] == MANIFEST_SHA
        assert args.repeats == 1
        references = previous["references"]
    report = dict(status="RUNNING", batch=args.batch, profile=args.profile, repeats=args.repeats,
                  manifest_sha256=MANIFEST_SHA, references=references, rows=[], started_unix=time.time(),
                  diagnostic_full_sequences=args.diagnostic_full_sequences)
    def cancel(signum, frame):
        raise InterruptedError("OWNED_BATCH_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    try:
        extra = {"KT_R63_FULL_DIAGNOSTIC_SEQUENCES":args.diagnostic_full_sequences} if args.diagnostic_full_sequences else None
        with OwnedServer(args.output, args.profile, batch=args.batch, capture=True, extra_env=extra) as server:
            with (args.output/"requests.jsonl").open("x") as log:
                for repeat in range(args.repeats):
                    for ci, cohort in enumerate(cohorts):
                        rids = [f"r64-b{args.batch}-{args.profile}-r{repeat}-c{ci}-{pid}" for pid in cohort]
                        payload = dict(rid=rids, input_ids=[prompts[pid]["input_ids"] for pid in cohort],
                                       sampling_params=dict(temperature=0, max_new_tokens=64, ignore_eos=True),
                                       return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                        request = urllib.request.Request("http://127.0.0.1:31000/generate", data=json.dumps(payload).encode(),
                                                         headers={"Content-Type":"application/json"})
                        start = time.monotonic()
                        with urllib.request.urlopen(request, timeout=600) as response:
                            answers = json.load(response)
                        log.write(json.dumps(dict(payload=payload, response=answers, seconds=time.monotonic()-start))+"\n")
                        log.flush()
                        tokens = response_map(answers, rids, 64)
                        for pid, rid in zip(cohort, rids):
                            report["rows"].append(dict(rid=rid, prompt_id=pid, cohort=ci, repeat=repeat,
                                                       input_ids=prompts[pid]["input_ids"], output_ids=tokens[rid]))
                            if pid in references and references[pid] != tokens[rid]:
                                raise ValueError(f"TOKEN_MISMATCH_REQUIRES_ACTUAL_SCHEDULE_AUDIT:{rid}")
                            references.setdefault(pid, tokens[rid])
                        print(json.dumps(dict(batch=args.batch, profile=args.profile, repeat=repeat,
                                              cohort=ci, requests=len(report["rows"]), exact=True)), flush=True)
                report["status"] = "BATCH_CAPTURE_GENERATION_PASS"
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        report["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.output.exists():
            (args.output/"batch-report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
