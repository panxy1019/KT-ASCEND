"""Count real routed CPU tasks and completion boundaries in a serving loop.

Separate instrumentation run at original batch1/context512. Not performance
or the later two-campaign quality-capacity stability qualification.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import urllib.request
from owned_server import OwnedServer
from run_compact_controls import checked, S6_SHA, validate

PROMPTS = ("s6_en_01", "s6_zh_01", "s6_math_01", "s6_json_01", "s6_code_01", "s6_code_02")


def audit_events(events):
    sys.path.insert(0, "/workspace/results/round6_2-prequalification/tools")
    from audit_deferred_transitions import validate_transition
    groups, current = [], []
    for i, event in enumerate(events):
        assert event["trace_sequence"] == i+1, "TRACE_GAP"
        if event["event"] == "submit" and event["layer_idx"] == 7:
            if current:
                groups.append(current)
            current = []
        current.append(event)
    if current:
        groups.append(current)
    tasks = delayed_tasks = syncs = 0
    expert_hits = {}
    for group in groups:
        submits = [e for e in group if e["event"] == "submit"]
        assert len(submits) == 2 and [e["layer_idx"] for e in submits] == [7,8], "FORWARD_BOUNDARY"
        outgoing = submits[0]["deferred_cpu_routes"]
        validate_transition(group, outgoing)
        for item in submits:
            immediate, deferred = item["immediate_cpu_routes"], item["deferred_cpu_routes"]
            assert item["rows"] > 0 and item["outgoing_pending"] == bool(deferred)
            tasks += int(immediate > 0)+int(deferred > 0)
            delayed_tasks += int(deferred > 0)
            for expert in set(item["immediate_cpu_experts"]+item["deferred_cpu_experts"]):
                key = f"L{item['layer_idx']}:E{expert}"
                expert_hits[key] = expert_hits.get(key,0)+1
        syncs += sum(e["event"] == "sync_complete" for e in group)
    assert tasks >= 1000 and delayed_tasks > 0, "INSUFFICIENT_REAL_TASK_EXPOSURE"
    assert syncs == 2*len(groups)
    return dict(physical_forwards=len(groups), real_cpu_tasks_with_selected_routes=tasks,
                real_deferred_tasks=delayed_tasks, worker_sync_complete=syncs,
                expert_forward_hits=expert_hits, pending_drained=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("REFUSE_LIFECYCLE_OVERWRITE")
    assert json.loads((args.runtime/"progress.json").read_text())["status"] == "RUNTIME_PASS"
    corpus = checked(args.corpus)
    assert corpus["sha256"] == S6_SHA
    prompts = {r["id"]:r for r in corpus["prompts"]}
    previous = json.loads((args.runtime/"same_process/report.json").read_text())
    references = {r["prompt_id"]:r["reference64"] for r in previous["rows"]}
    report = dict(status="RUNNING", observer="legacy_metadata_trace_only", capacity="original512", requests=0,
                  selected_s6=list(PROMPTS), cycles=6, rows=[], started_unix=time.time())
    def cancel(signum, frame):
        raise InterruptedError("OWNED_LIFECYCLE_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    try:
        import psutil
        with OwnedServer(args.output, "D1", capacity=False, extra_env={
            "KT_DEBUG_DEFERRED_TRACE_FILE":str(args.output/"deferred-events.jsonl"),
            "KT_DEBUG_DEFERRED_TRACE_LIMIT":"200000"}) as server:
            def memory():
                parent = psutil.Process(server.server.pid)
                children = [parent]+parent.children(recursive=True)
                return dict(time_unix=time.time(), summed_owned_process_rss=sum(p.memory_info().rss for p in children if p.is_running()),
                            process_count=len(children))
            report["memory_samples"] = [memory()]
            with (args.output/"requests.jsonl").open("x") as raw:
                for cycle in range(6):
                    for pid in PROMPTS:
                        rid = f"r64-life-{cycle}-{pid}"
                        payload = dict(rid=rid,input_ids=prompts[pid]["input_ids"],sampling_params=dict(temperature=0,max_new_tokens=64,ignore_eos=True),
                                       return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                        req = urllib.request.Request("http://127.0.0.1:31000/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
                        started = time.monotonic()
                        with urllib.request.urlopen(req,timeout=600) as response:
                            answer = json.load(response)
                        raw.write(json.dumps(dict(payload=payload,response=answer,seconds=time.monotonic()-started))+"\n")
                        raw.flush()
                        assert validate(answer,rid,64) == references[pid], "LIFECYCLE_TRAJECTORY_MISMATCH"
                        report["requests"] += 1
                        report["memory_samples"].append(memory())
                        print(json.dumps(dict(requests=report["requests"],rid=rid,exact=True)),flush=True)
        events = [json.loads(s) for s in (args.output/"deferred-events.jsonl").read_text().splitlines()]
        report["task_audit"] = audit_events(events)
        report["status"] = "D1_REAL_TASK_LIFECYCLE_PASS"
    except BaseException as exc:
        report.update(status="FAILED",exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        if args.output.exists():
            (args.output/"lifecycle-report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
