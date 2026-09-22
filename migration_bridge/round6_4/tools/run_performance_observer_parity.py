"""Same-profile recorder/streaming parity before formal performance timing.

Do not confuse this exact engineering check with Deferred OFF/ON equivalence.
Four frozen S6 prompts, one quality-configuration reference then five repeats
with performance observers disabled and native streaming enabled, per profile.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import urllib.request
from owned_server import OwnedServer
from run_compact_controls import checked, S6_SHA, PROMPTS, validate
from stream_timing import request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--performance-manifest", type=Path)
    args = p.parse_args()
    corpus = checked(args.corpus)
    assert corpus["sha256"] == S6_SHA
    prompts = {r["id"]:r for r in corpus["prompts"]}
    selected, budget = PROMPTS, 64
    if args.performance_manifest:
        from run_performance_phase import FAST_MANIFEST_SHA
        assert hashlib.sha256(args.performance_manifest.read_bytes()).hexdigest() == FAST_MANIFEST_SHA
        spec = checked(args.performance_manifest)
        prompts = {r['id']:r for r in spec['shapes']}
        selected, budget = tuple(prompts),256
    args.output.mkdir(exist_ok=False)
    report = dict(status="RUNNING",started_unix=time.time(),completed=[],profile_references={},requests=0)
    def cancel(signum,frame):
        raise InterruptedError("OWNED_PERFORMANCE_PARITY_CANCELLED")
    signal.signal(signal.SIGTERM,cancel)
    try:
        for profile in ("D1_NODEFER","D1"):
            refs = {}
            for performance in (False,True):
                mode = "performance" if performance else "quality"
                root = args.output/f"{profile}-{mode}"
                extra = None if performance else {'KT_DEBUG_DEFERRED_TRACE_FILE':str(root/'events.jsonl'),'KT_DEBUG_DEFERRED_TRACE_LIMIT':'200000'}
                with OwnedServer(root,profile,performance=performance,tokenizer_compat=True,extra_env=extra):
                    with (root/"requests.jsonl").open("x") as raw:
                        for repeat in range(5 if performance else 1):
                            for pid in selected:
                                rid = f"r64-observer-parity-{profile}-{mode}-{repeat}-{pid}"
                                if performance:
                                    answer = request(rid,prompts[pid]["input_ids"],budget)["response"]
                                else:
                                    payload = dict(rid=rid,input_ids=prompts[pid]["input_ids"],sampling_params=dict(
                                        temperature=0,max_new_tokens=budget,ignore_eos=True),return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                                    req = urllib.request.Request("http://127.0.0.1:31000/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
                                    with urllib.request.urlopen(req,timeout=600) as response:
                                        answer = json.load(response)
                                raw.write(json.dumps(dict(prompt=pid,repeat=repeat,response=answer))+"\n")
                                raw.flush()
                                tokens = validate(answer,rid,budget)
                                if performance:
                                    assert tokens == refs[pid], "SAME_PROFILE_OBSERVER_OR_STREAM_DRIFT"
                                else:
                                    refs[pid] = tokens
                                report["requests"] += 1
                                print(json.dumps(dict(profile=profile,mode=mode,repeat=repeat,prompt=pid,exact=True)),flush=True)
                report["completed"].append(f"{profile}-{mode}")
                if not performance and profile == 'D1':
                    events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
                    hits=sum(x.get('deferred_cpu_routes',0) for x in events if x['event']=='submit')
                    assert hits>0, 'PERFORMANCE_WORKLOAD_NO_DEFERRED_EXPOSURE'
                    report['same_workload_deferred_route_hits']=hits
            report["profile_references"][profile] = refs
        report["status"] = "PERFORMANCE_OBSERVER_PARITY_PASS"
    except BaseException as exc:
        report.update(status="FAILED",exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output/"parity-report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
