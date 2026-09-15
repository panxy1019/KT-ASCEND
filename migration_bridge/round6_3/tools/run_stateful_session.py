"""Own one short prequalification server lifecycle; never stop other services."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["D1", "D1_NODEFER"], required=True)
    p.add_argument("--server-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--forced", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    old = Path("/workspace/results/round6_2-prequalification/tools")
    env = dict(os.environ, PYTHONPATH=str(old))
    results = dict(started_unix=time.time(), profile=args.profile, forced=args.forced,
                   server_root=str(args.server_root), requests=str(args.requests), status="STARTING")
    def call(command):
        subprocess.run([sys.executable, *map(str, command)], env=env, check=True, timeout=1200)
    launch = ["bash", str(here / "launch_stateful_smoke.sh"), args.profile, str(args.server_root)]
    if args.forced:
        launch.append(str(args.requests))
    with (args.output / "launcher.log").open("x") as log:
        server = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT, env=env)
        results["owned_launcher_pid"] = server.pid
        try:
            call([old / "wait_capture_ready.py", "--log", args.server_root / "guard/process.log",
                  "--captures", args.server_root / "tensors", "--output", args.server_root / "readiness.json"])
            if args.forced:
                call([here / "prepare_stateful_smoke.py", "--arm", "--baseline-result", args.requests,
                      "--output", args.server_root / "armed.sha256"])
            command = [here / "run_stateful_probe.py", "--requests", args.requests,
                       "--index", args.server_root / "index.jsonl", "--output", args.output / "capture"]
            if args.forced:
                command += ["--trace", args.server_root / "forcing.jsonl"]
            call(command)
            results["status"] = "CAPTURE_COMPLETE"
        except BaseException as exc:
            results.update(status="FAILED", exception=repr(exc))
            raise
        finally:
            # Only our directly spawned (unreaped) launcher/guard gets TERM.
            # Its own guard owns the model process group and performs cleanup.
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
                server.wait(timeout=60)
            results.update(ended_unix=time.time(), launcher_returncode=server.returncode)
            (args.output / "session.json").write_text(json.dumps(results, indent=2) + "\n")
    call([here / "audit_stateful_index.py", "--server-root", args.server_root,
          "--result", args.output / "capture/result.json", "--output", args.output / "final-integrity.json"])
    results["status"] = "PREQUALIFICATION_SESSION_PASS"
    (args.output / "session.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
