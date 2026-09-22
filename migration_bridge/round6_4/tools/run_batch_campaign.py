"""Sequential, fail-closed batch engineering campaign; never quality promotion."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def stages():
    # Short tests first. Each profile has its own references: OFF != ON is legal.
    for batch in (2, 4):
        for profile in ("D1_NODEFER", "D1"):
            yield batch, profile, "pilot", 1, None
    for batch in (2, 4):
        for profile in ("D1_NODEFER", "D1"):
            yield batch, profile, "repeat", 10, None
            # Two independent process restarts, not two requests in one process.
            for restart in (1, 2):
                yield batch, profile, f"restart{restart}", 1, "repeat"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--lifecycle", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    assert json.loads((args.oracle/"comparison.json").read_text())["status"] == "ENGINEERING_ORACLE_PASS"
    assert json.loads((args.lifecycle/"lifecycle-report.json").read_text())["status"] == "D1_REAL_TASK_LIFECYCLE_PASS"
    assert json.loads((args.runtime/"progress.json").read_text())["status"] == "RUNTIME_PASS"
    if args.output.parent != Path("/workspace/results") or not args.output.name.startswith("r64-"):
        raise ValueError("DEDICATED_CAMPAIGN_ROOT_REQUIRED")
    args.output.mkdir(exist_ok=False)
    tools = Path(__file__).parent
    report = dict(status="RUNNING", completed=[], started_unix=time.time(), quality_qualified=False)
    child = None
    def cancel(signum, frame):
        raise InterruptedError("OWNED_BATCH_CAMPAIGN_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    def run(cmd, log):
        nonlocal child
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        finally:
            if child.poll() is None:
                # The child serving wrapper owns the guard and drains its own
                # process group. Never signal a device-wide/host-wide process.
                child.send_signal(signal.SIGTERM)
                child.wait(timeout=120)
    try:
        for batch, profile, kind, repeats, reference_kind in stages():
            name = f"b{batch}-{profile}-{kind}"
            root = args.output/name
            report["active"] = name
            (args.output/"campaign.json").write_text(json.dumps(report, indent=2)+"\n")
            cmd = [sys.executable, str(tools/"run_batch_capture.py"), "--manifest", str(args.manifest),
                   "--runtime", str(args.runtime), "--batch", str(batch), "--profile", profile,
                   "--repeats", str(repeats), "--output", str(root)]
            if reference_kind:
                cmd += ["--restart-reference", str(args.output/f"b{batch}-{profile}-{reference_kind}"/"batch-report.json")]
            with (args.output/f"{name}.log").open("x") as log:
                run(cmd, log)
                run([sys.executable, str(tools/"audit_batch_capture.py"), "--root", str(root)], log)
            audit = json.loads((root/"batch-audit.json").read_text())
            assert audit["status"] == "BATCH_TRANSACTION_AUDIT_PASS"
            if reference_kind:
                prior = json.loads((args.output/f"b{batch}-{profile}-{reference_kind}"/"batch-audit.json").read_text())
                assert set(audit["schedule_sha256_by_repeat"].values()) == set(prior["schedule_sha256_by_repeat"].values()), "RESTART_ACTUAL_SCHEDULE_MISMATCH"
            report["completed"].append(name)
            print(json.dumps(dict(completed=name, status="PASS")), flush=True)
        report["status"] = "BATCH_ENGINEERING_PASS_NOT_QUALITY_QUALIFIED"
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output/"campaign.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
