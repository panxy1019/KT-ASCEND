"""Five fixed ABBA blocks; independent process per phase, no winner selection."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_compact_controls import checked
from run_performance_phase import MANIFEST_RAW_SHA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--entry-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    entry = json.loads(args.entry_audit.read_text())
    assert entry["status"] == "R64_PERFORMANCE_ENTRY_APPROVED"
    assert entry["manifest_raw_sha256"] == MANIFEST_RAW_SHA
    manifest = checked(args.manifest)
    assert manifest["phase_order"] == ["D1_NODEFER","D1","D1","D1_NODEFER"]
    assert manifest["paired_blocks"] == 5
    args.output.mkdir(exist_ok=False)
    report = dict(status="RUNNING",completed=[],started_unix=time.time(),manifest_sha256=MANIFEST_RAW_SHA)
    def cancel(signum,frame):
        raise InterruptedError("OWNED_PERFORMANCE_CAMPAIGN_CANCELLED")
    signal.signal(signal.SIGTERM,cancel)
    reference_source = None
    try:
        for block in range(5):
            for phase,profile in enumerate(manifest["phase_order"]):
                name = f"block{block}-phase{phase}-{profile}"
                root = args.output/name
                report["active"] = name
                (args.output/"campaign.json").write_text(json.dumps(report,indent=2)+"\n")
                with (args.output/f"{name}.log").open("x") as log:
                    child = subprocess.Popen([sys.executable,str(Path(__file__).with_name("run_performance_phase.py")),
                        "--manifest",str(args.manifest),"--profile",profile,"--output",str(root)],stdout=log,stderr=subprocess.STDOUT)
                    try:
                        code = child.wait()
                        if code:
                            raise subprocess.CalledProcessError(code,child.args)
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            child.wait(timeout=120)
                result = json.loads((root/"performance-report.json").read_text())
                assert result["status"] == "PERFORMANCE_PHASE_COMPLETE" and len(result["rows"]) == 6
                actual = json.loads((root/"provenance.json").read_text())["file_sha256"]
                if reference_source is None:
                    reference_source = actual
                else:
                    assert actual == reference_source, "PERFORMANCE_SOURCE_DRIFT"
                report["completed"].append(name)
                print(json.dumps(dict(completed=name)),flush=True)
        report["status"] = "PERFORMANCE_CAPTURE_COMPLETE_PENDING_ANALYSIS"
    except BaseException as exc:
        report.update(status="FAILED",exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output/"campaign.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
