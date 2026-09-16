"""Same-profile recorder/streaming parity before formal performance timing.

Do not confuse this exact engineering check with Deferred OFF/ON equivalence.
Four frozen S6 prompts, one quality-configuration reference then five repeats
with performance observers disabled and native streaming enabled, per profile.
"""
import argparse
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
    args = p.parse_args()
    corpus = checked(args.corpus)
    assert corpus["sha256"] == S6_SHA
    prompts = {r["id"]:r for r in corpus["prompts"]}
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
                with OwnedServer(root,profile,performance=performance):
                    with (root/"requests.jsonl").open("x") as raw:
                        for repeat in range(5 if performance else 1):
                            for pid in PROMPTS:
                                rid = f"r64-observer-parity-{profile}-{mode}-{repeat}-{pid}"
                                if performance:
                                    answer = request(rid,prompts[pid]["input_ids"],64)["response"]
                                else:
                                    payload = dict(rid=rid,input_ids=prompts[pid]["input_ids"],sampling_params=dict(
                                        temperature=0,max_new_tokens=64,ignore_eos=True),return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                                    req = urllib.request.Request("http://127.0.0.1:31000/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
                                    with urllib.request.urlopen(req,timeout=600) as response:
                                        answer = json.load(response)
                                raw.write(json.dumps(dict(prompt=pid,repeat=repeat,response=answer))+"\n")
                                raw.flush()
                                tokens = validate(answer,rid,64)
                                if performance:
                                    assert tokens == refs[pid], "SAME_PROFILE_OBSERVER_OR_STREAM_DRIFT"
                                else:
                                    refs[pid] = tokens
                                report["requests"] += 1
                                print(json.dumps(dict(profile=profile,mode=mode,repeat=repeat,prompt=pid,exact=True)),flush=True)
                report["completed"].append(f"{profile}-{mode}")
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
