"""Dedicated, bounded vendor-profiler diagnostic of one owned scheduler only.

Not a formal performance measurement. Trace collection success alone does
not prove CPU compute overlap; CPU/NPU timeline interpretation is separate.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import psutil
from owned_server import OwnedServer
from run_compact_controls import checked, S6_SHA, PROMPTS, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    corpus = checked(args.corpus)
    assert corpus["sha256"] == S6_SHA
    prompts = {r["id"]:r for r in corpus["prompts"]}
    if args.output.exists():
        raise ValueError("REFUSE_TRACE_OVERWRITE")
    report = dict(status="RUNNING",started_unix=time.time(),npu=0,formal_performance=False)
    sampler = None
    def cancel(signum,frame):
        raise InterruptedError("OWNED_OVERLAP_TRACE_CANCELLED")
    signal.signal(signal.SIGTERM,cancel)
    def generate(pid,budget):
        rid = f"r64-overlap-{pid}-{budget}"
        payload = dict(rid=rid,input_ids=prompts[pid]["input_ids"],sampling_params=dict(
            temperature=0,max_new_tokens=budget,ignore_eos=True),return_logprob=True,logprob_start_len=-1)
        request = urllib.request.Request("http://127.0.0.1:31000/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=600) as response:
            result = json.load(response)
        validate(result,rid,budget)
        return dict(payload=payload,response=result,time_unix=time.time())
    try:
        with OwnedServer(args.output,"D1",tokenizer_compat=True,extra_env={
            "KT_DEBUG_DEFERRED_TRACE_FILE":str(args.output/"events.jsonl"),"KT_DEBUG_DEFERRED_TRACE_LIMIT":"100000"}) as server:
            with (args.output/"requests.jsonl").open("x") as raw:
                raw.write(json.dumps(generate(PROMPTS[0],16))+"\n")
                owned = psutil.Process(server.server.pid).children(recursive=True)
                schedulers = [p for p in owned if "sglang::scheduler" in " ".join(p.cmdline())]
                if len(schedulers) != 1:
                    raise ValueError("OWNED_SCHEDULER_NOT_UNIQUE")
                scheduler = schedulers[0]
                command = ["/usr/local/Ascend/ascend-toolkit/latest/bin/msprof","--dynamic=on",
                    f"--pid={scheduler.pid}","--duration=20","--sys-devices=0","--host-sys=cpu",
                    f"--host-sys-pid={scheduler.pid}",f"--output={args.output/'vendor-trace'}"]
                report.update(command=command,owned_scheduler_pid=scheduler.pid,process_create_time=scheduler.create_time())
                with (args.output/"profiler.log").open("x") as log:
                    sampler = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    time.sleep(2)
                    for pid in PROMPTS:
                        raw.write(json.dumps(generate(pid,64))+"\n")
                        raw.flush()
                    report["profiler_returncode"] = sampler.wait(timeout=180)
                if report["profiler_returncode"]:
                    raise RuntimeError("OWNED_VENDOR_PROFILER_FAILED")
        report["status"] = "TRACE_CAPTURED_PENDING_OVERLAP_ANALYSIS"
    except BaseException as exc:
        report.update(status="FAILED",exception=repr(exc))
        raise
    finally:
        if sampler is not None and sampler.poll() is None:
            sampler.terminate()
            sampler.wait(timeout=30)
        report["ended_unix"] = time.time()
        if args.output.exists():
            (args.output/"trace-report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
