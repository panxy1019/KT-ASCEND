"""Finite gated continuation: observer1 parity -> independent pilot -> size plan.

Does not derive numerical thresholds, create QD3/HD3, or start qualification.
All model processes remain under the existing owned-process lifecycle guards.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/workspace/results")
TOOLS = ROOT / "round6_3-prequalification/tools"
FROZEN = ROOT / "round6_3-prequalification"


def call(script, *arguments, compact=False):
    env = dict(os.environ, KT_R63_COMPACT_CAPTURE="1" if compact else "0",
               PYTHONPATH=str(ROOT / "round6_2-prequalification/tools"))
    subprocess.run([sys.executable, str(TOOLS / script), *map(str, arguments)],
                   env=env, check=True, timeout=1500)


def checked(path, status):
    data = json.loads(path.read_text())
    if data.get("status") != status:
        raise ValueError(f"PRECONDITION_FAILED:{path}")
    return data


def main():
    # This historical pipeline is intentionally disabled: its original order
    # omitted true observer-OFF qualification. Preserve its outputs; a fresh
    # corrected pipeline must explicitly bind an OFF/reference PASS artifact.
    raise RuntimeError("SUPERSEDED_MISSING_TRUE_OBSERVER_OFF_GATE")
    checked(ROOT / "r63-long-observer0-parity.json", "PASS")
    pilot = FROZEN / "pilot-input.json"
    if hashlib.sha256(pilot.read_bytes()).hexdigest() != "0e309656587683068e515cd34024a360c6c8ca36924e3ee8a6a64d5f5b42e6ef":
        raise ValueError("PILOT_INPUT_DRIFT")
    audit = checked(FROZEN / "pilot-token-audit.json", "PASS")
    if audit["pilot_sha256"] != hashlib.sha256(pilot.read_bytes()).hexdigest():
        raise ValueError("PILOT_TOKEN_AUDIT_DRIFT")
    print("Waiting for owned full-observer D1 control to finish naturally", flush=True)
    deadline = time.monotonic() + 1200
    ready = ROOT / "r63-long-force1-full-control/result.json"
    while not ready.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("PREEXISTING_FULL_OBSERVER_NOT_FINISHED")
        time.sleep(2)
    if not json.loads(ready.read_text())["success"]:
        raise ValueError("FULL_OBSERVER_CONTROL_FAILED")
    checked(ROOT / "r63-long-force1-full-session/session.json", "PREQUALIFICATION_SESSION_PASS")
    checked(ROOT / "r63-long-force1-full-session/final-integrity.json", "PASS")
    print("Starting D1 compact observer with identical frozen histories", flush=True)
    call("run_stateful_session.py", "--profile", "D1", "--server-root",
         ROOT / "r63-long-force1-compact-server", "--output", ROOT / "r63-long-force1-compact-session",
         "--requests", FROZEN / "long-force.json", "--forced", compact=True)
    call("compare_observers.py", "--full", ROOT / "r63-long-force1-full-session/capture/result.json",
         "--compact", ROOT / "r63-long-force1-compact-session/capture/result.json",
         "--output", ROOT / "r63-long-observer1-parity.json")
    checked(ROOT / "r63-long-observer1-parity.json", "PASS")
    print("Both observer modes pass; starting independent pilot baseline", flush=True)
    call("run_stateful_session.py", "--profile", "D1_NODEFER", "--server-root",
         ROOT / "r63-pilot-baseline-server", "--output", ROOT / "r63-pilot-baseline-session",
         "--requests", pilot, compact=True)
    call("prepare_stateful_smoke.py", "--baseline-result", ROOT / "r63-pilot-baseline-session/capture/result.json",
         "--output", FROZEN / "pilot-force.json")
    print("Pilot histories frozen; starting D1 pilot replay", flush=True)
    call("run_stateful_session.py", "--profile", "D1", "--server-root",
         ROOT / "r63-pilot-defer1-server", "--output", ROOT / "r63-pilot-defer1-session",
         "--requests", FROZEN / "pilot-force.json", "--forced", compact=True)
    call("assemble_stateful_positions.py", "--session", ROOT / "r63-pilot-defer1-session",
         "--corpus", pilot, "--forcing", FROZEN / "pilot-force.json",
         "--output", ROOT / "r63-pilot-positions.json")
    call("pilot_sample_size.py", "--pilot", ROOT / "r63-pilot-defer1-session/capture/result.json",
         "--output", ROOT / "r63-pilot-size-plan.json")
    print("PILOT_COMPLETE: no numerical reserves derived; QD3/HD3 still not created or run", flush=True)


if __name__ == "__main__":
    main()
