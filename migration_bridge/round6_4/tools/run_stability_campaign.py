"""One of two independent batch1 stability campaigns at quality capacity.

Six frozen engineering prompts x64 tokens x3 cycles. Exact same-path output,
metadata task drain, and owned RSS. HBM is collected by the read-only host
monitor separately; absence of that evidence cannot be called an HBM PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import urllib.request
from owned_server import OwnedServer
from run_compact_controls import checked, S6_SHA, validate
from run_d1_lifecycle import PROMPTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    corpus = checked(args.corpus)
    assert corpus["sha256"] == S6_SHA
    prompts = {r["id"]:r for r in corpus["prompts"]}
    refs = {}
    if args.reference:
        prior = json.loads(args.reference.read_text())
        assert prior["status"] == "STABILITY_RUNTIME_PASS_PENDING_HBM_REVIEW"
        assert prior["requests"] == 18 and prior["corpus_sha256"] == S6_SHA
        refs = prior["references"]
        assert set(refs) == set(PROMPTS)
    if args.output.exists():
        raise ValueError("REFUSE_STABILITY_OVERWRITE")
    report = dict(status="RUNNING", corpus_sha256=S6_SHA, requests=0, references=refs,
                  started_unix=time.time(), memory_samples=[], capacity=1024, batch=1,
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None)
    def cancel(signum, frame):
        raise InterruptedError("OWNED_STABILITY_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    try:
        import psutil
        with OwnedServer(args.output, "D1", capture=False, tokenizer_compat=True, extra_env={
            "KT_DEBUG_DEFERRED_TRACE_FILE":str(args.output/"events.jsonl"), "KT_DEBUG_DEFERRED_TRACE_LIMIT":"200000"}) as server:
            def memory():
                processes = [psutil.Process(server.server.pid)]
                processes += processes[0].children(recursive=True)
                return dict(time_unix=time.time(), summed_owned_rss_bytes=sum(x.memory_info().rss for x in processes if x.is_running()))
            report["memory_samples"].append(memory())
            with (args.output/"requests.jsonl").open("x") as log:
                for cycle in range(3):
                    for pid in PROMPTS:
                        rid = f"r64-stability-{cycle}-{pid}"
                        payload = dict(rid=rid, input_ids=prompts[pid]["input_ids"], sampling_params=dict(
                            temperature=0, max_new_tokens=64, ignore_eos=True), return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                        req = urllib.request.Request("http://127.0.0.1:31000/generate", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
                        start = time.monotonic()
                        with urllib.request.urlopen(req, timeout=600) as response:
                            result = json.load(response)
                        log.write(json.dumps(dict(request=payload, response=result, seconds=time.monotonic()-start))+"\n")
                        log.flush()
                        tokens = validate(result, rid, 64)
                        if pid in refs:
                            assert refs[pid] == tokens, "STABILITY_SAME_PATH_MISMATCH"
                        refs.setdefault(pid, tokens)
                        report["requests"] += 1
                        report["memory_samples"].append(memory())
                        print(json.dumps(dict(requests=report["requests"], cycle=cycle, prompt=pid, exact=True)), flush=True)
        sys.path.insert(0, "/workspace/results/round6_2-prequalification/tools")
        from audit_deferred_transitions import validate_transition
        events = [json.loads(s) for s in (args.output/"events.jsonl").read_text().splitlines()]
        groups, group = [], []
        for i,e in enumerate(events):
            assert e["trace_sequence"] == i+1
            if e["event"] == "submit" and e["layer_idx"] == 7:
                if group:
                    groups.append(group)
                group = []
            group.append(e)
        if group:
            groups.append(group)
        assert len(groups) == 18*65, "STABILITY_PHYSICAL_FORWARD_COUNT"
        for group in groups:
            submits = [e for e in group if e["event"] == "submit"]
            assert [e["layer_idx"] for e in submits] == [7,8]
            validate_transition(group, submits[0]["deferred_cpu_routes"])
        report.update(status="STABILITY_RUNTIME_PASS_PENDING_HBM_REVIEW", generated_tokens=18*64,
                      physical_forwards=len(groups), pending_drained=True, hbm_review="PENDING_HOST_MONITOR")
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        if args.output.exists():
            (args.output/"stability-report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
