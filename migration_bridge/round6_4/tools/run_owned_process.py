"""Retained process-group guard plus reaping of its own orphaned descendants.

Does not signal any additional PID or change the launched model. Linux
subreaper status applies only to this guard's descendant tree. Existing
container-PID1 zombies are neither adopted nor modified by this wrapper.
"""
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys
import time


def enable_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "SET_OWNED_SUBREAPER_FAILED")


def reap_children(timeout=10):
    reaped = []
    deadline = time.monotonic()+timeout
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return dict(status="OWNED_DESCENDANTS_REAPED", reaped=reaped)
        if pid:
            reaped.append(dict(pid=pid, wait_status=status))
        elif time.monotonic() >= deadline:
            return dict(status="OWNED_DESCENDANTS_STILL_ALIVE", reaped=reaped)
        else:
            time.sleep(.05)


def main():
    output = Path(sys.argv[sys.argv.index("--output")+1])
    if not str(output).startswith("/workspace/results/r64-"):
        raise ValueError("OWNED_REAPER_SCOPE")
    enable_subreaper()
    try:
        runpy.run_path("/workspace/results/round6_2-prequalification/tools/run_owned_process.py",run_name="__main__")
    finally:
        result = reap_children()
        if output.exists():
            (output/"reaping.json").write_text(json.dumps(result,indent=2)+"\n")
        if result["status"] != "OWNED_DESCENDANTS_REAPED":
            raise RuntimeError("OWNED_DESCENDANTS_NOT_REAPED")


if __name__ == "__main__":
    main()
