"""Four preselected S6 requests: production Deferred vs private-buffer oracle.

Use successful D1 runtime histories, never select requests based on mismatch.
The old two rejected-QD3 histories remain supplemental historical diagnostics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_compact_controls import ensure_free_port, checked, S6_SHA, PROMPTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    state = json.loads((args.runtime/"progress.json").read_text())
    assert state["status"] == "RUNTIME_PASS" and state["completed"] == ["same_process", "clean_restart"]
    previous = json.loads((args.runtime/"same_process/report.json").read_text())
    assert previous["status"] == "RUNTIME_PASS" and previous["request_count"] == 156
    corpus = checked(args.corpus)
    assert corpus["sha256"] == S6_SHA
    prompts = {r["id"]: r for r in corpus["prompts"]}
    references = {r["prompt_id"]: r["reference64"] for r in previous["rows"]}
    assert args.output.parent == Path("/workspace/results") and args.output.name.startswith("r64-")
    ensure_free_port()
    args.output.mkdir(exist_ok=False)
    manifest = args.output/"force.json"
    manifest.write_text(json.dumps(dict(token_budget=64, session_id="r64-engineering-oracle-v1",
        requests=[dict(rid=f"r64-oracle-{pid}", input_ids=prompts[pid]["input_ids"],
                       baseline_output_ids=references[pid]) for pid in PROMPTS]), indent=2)+"\n")
    old = Path("/workspace/results/round6_2-prequalification/tools")
    retained = Path("/workspace/results/round6_3-prequalification/tools")
    report = dict(status="RUNNING", started_unix=time.time(), selected_s6=list(PROMPTS),
                  runtime_binding=state["binding"], manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  tool_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), *[retained/n for n in ("run_stateful_probe.py", "probe_explicit_diagnostic.py", "launch_stateful_smoke.sh", "audit_stateful_index.py")]]})

    def cancel(signum, frame):
        raise InterruptedError("OWNED_ORACLE_CANCELLED")

    signal.signal(signal.SIGTERM, cancel)
    try:
        for mode in ("production", "explicit"):
            server_root = args.output.parent / f"r63-{args.output.name}-{mode}-server"
            session = args.output/mode
            session.mkdir()
            env = {k:v for k,v in os.environ.items() if not k.startswith(("KT_R61_", "KT_R62_", "KT_R63_", "KT_DEBUG_"))}
            env.update(KT_R63_COMPACT_CAPTURE="0", KT_R63_OBSERVER_PROFILE="full", PYTHONPATH=f"{retained}:{old}")
            if mode == "explicit":
                env["KT_R61_EXPLICIT_REPLAY"] = "1"
            with (session/"process.log").open("x") as log:
                def call(script, *arguments):
                    subprocess.run([sys.executable, str(script), *map(str, arguments)], env=env,
                        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
                server = subprocess.Popen(["bash", str(retained/"launch_stateful_smoke.sh"), "D1", str(server_root), str(manifest)],
                                          env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    call(old/"wait_capture_ready.py", "--log", server_root/"guard/process.log", "--captures", server_root/"tensors", "--output", server_root/"readiness.json")
                    call(retained/"prepare_stateful_smoke.py", "--arm", "--baseline-result", manifest, "--output", server_root/"armed.sha256")
                    call(retained/("run_stateful_probe.py" if mode == "production" else "probe_explicit_diagnostic.py"),
                         "--requests", manifest, "--index", server_root/"index.jsonl", "--trace", server_root/"forcing.jsonl", "--output", session/"capture")
                finally:
                    if server.poll() is None:
                        server.send_signal(signal.SIGTERM)
                        server.wait(timeout=60)
                call(retained/"audit_stateful_index.py", "--server-root", server_root,
                     "--result", session/"capture/result.json", "--output", session/"final-integrity.json")
        report["status"] = "CAPTURE_COMPLETE_PENDING_EXACT_COMPARISON"
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
