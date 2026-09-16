"""HD3/FD3 natural A, matched-history B, natural B; numerical metrics diagnostic.

Reuse the audited stateful capture/index protocol. No R6.3 threshold or
candidate approval is imported. Finite and transaction integrity remain hard.
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
from owned_server import OwnedServer

FROZEN = {"hd3.json":("4e54ac5c0e6ab9f25991ca29d54e32d1632d4e687f72d344430d4b5691b8590e",128),
          "fd3.json":("9b8703afbdfd3bd925096725168f9d68914bd116e726336a972f5376443f7fe9",8)}


def capture_environment(inherited):
    """Deployed legacy tools are not in their original repository layout."""
    env = dict(inherited)
    dependency = '/workspace/results/round6_2-prequalification/tools'
    env['PYTHONPATH'] = dependency + os.pathsep + env.get('PYTHONPATH', '')
    return env


def read_inputs(folder):
    requests = []
    for name, (digest, count) in FROZEN.items():
        raw = (Path(folder)/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("FROZEN_HOLDOUT_DRIFT:"+name)
        corpus = json.loads(raw)
        if corpus["token_budget"] != 64 or len(corpus["requests"]) != count:
            raise ValueError("HOLDOUT_COUNT_OR_BUDGET")
        requests.extend(corpus["requests"])
    if len({r["rid"] for r in requests}) != 136 or any(len(r["input_ids"])+64 > 512 for r in requests):
        raise ValueError("HOLDOUT_ID_OR_CAPACITY")
    return dict(version="r64-descriptive-native-v1", token_budget=64, requests=requests)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--entry-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    entry = json.loads(args.entry_audit.read_text())
    if entry.get("status") != "R64_DESCRIPTIVE_ENTRY_APPROVED" or not entry.get("no_holdout_fitting"):
        raise ValueError("HOLDOUT_ENTRY_NOT_AUDITED")
    for name, (digest, _) in FROZEN.items():
        assert entry["corpus_raw_sha256"][name] == digest
    payload = read_inputs(args.inputs)
    args.output.mkdir(exist_ok=False)
    native = args.output/"native.json"
    native.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    old = Path("/workspace/results/round6_3-prequalification/tools")
    capture_env = capture_environment(os.environ)
    # Fail before allocating a model if the deployed helper cannot be imported.
    subprocess.run([sys.executable, '-c', 'from audit_deferred_transitions import validate_transition'],
                   env=capture_env, check=True)
    report = dict(status="RUNNING", started_unix=time.time(), completed=[],
                  role="DESCRIPTIVE_NOT_QUALITY_GATE", entry_audit_sha256=hashlib.sha256(args.entry_audit.read_bytes()).hexdigest())
    def cancel(signum, frame):
        raise InterruptedError("OWNED_HOLDOUT_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    child = None
    def call(cmd, log):
        nonlocal child
        child = subprocess.Popen([sys.executable, *map(str, cmd)], stdout=log, stderr=subprocess.STDOUT, env=capture_env)
        try:
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
    force = args.output/"force.json"
    try:
        for mode, profile in (("off_native","D1_NODEFER"), ("on_forced","D1"), ("on_native","D1")):
            root = args.output/mode
            env = {}
            if mode == "on_forced":
                env = dict(KT_R63_FORCE_MANIFEST=str(force), KT_R63_FORCE_MANIFEST_SHA256=hashlib.sha256(force.read_bytes()).hexdigest(),
                           KT_R63_FORCE_ARM_FILE=str(root/"armed.sha256"), KT_R63_FORCE_TRACE=str(root/"forcing.jsonl"))
            report["active"] = mode
            (args.output/"progress.json").write_text(json.dumps(report, indent=2)+"\n")
            with (args.output/f"{mode}.log").open("x") as log:
                with OwnedServer(root, profile, batch=1, capture=True, capacity=False, extra_env=env):
                    manifest = force if mode == "on_forced" else native
                    if mode == "on_forced":
                        call([old/"prepare_stateful_smoke.py", "--arm", "--baseline-result", force, "--output", root/"armed.sha256"], log)
                    cmd = [old/"run_stateful_probe.py", "--requests", manifest, "--index", root/"index.jsonl", "--output", root/"capture"]
                    if mode == "on_forced":
                        cmd += ["--trace", root/"forcing.jsonl"]
                    call(cmd, log)
                call([old/"audit_stateful_index.py", "--server-root", root, "--result", root/"capture/result.json",
                      "--output", root/"final-integrity.json"], log)
                if mode == "off_native":
                    call([old/"prepare_stateful_smoke.py", "--baseline-result", root/"capture/result.json", "--output", force], log)
            report["completed"].append(mode)
            print(json.dumps(dict(mode=mode, status="CAPTURE_INTEGRITY_PASS")), flush=True)
        report["status"] = "DESCRIPTIVE_CAPTURE_COMPLETE_PENDING_ANALYSIS"
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output/"progress.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
