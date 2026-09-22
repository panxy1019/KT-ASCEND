"""Run INSIDE the project container; terminate only the child session we create.

No process-name matching, docker stop, host-wide kill, SSH, or remote cleanup.
The launched command must keep its workers in this foreground process group.
This guard is supplementary: memory preflight and request-level timeouts remain
required. It does not reserve HBM or detect every possible device failure.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ERROR = re.compile(
    r"out of memory|\boom\b|\btimed?\s*out\b|timeout(?:error| expired)|"
    r"device (?:error|fault)|ACL_ERROR_RT|\bEZ\d{4}\b|"
    r"Traceback \(most recent call last\)", re.IGNORECASE)


def run(command: list[str], output: Path, timeout: float, grace: float = 10,
        interval: float = .5) -> dict:
    if not command or timeout <= 0 or grace < 0 or interval <= 0:
        raise ValueError("invalid command or timing")
    # Refuse reuse so no previous run's output/provenance is overwritten.
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cancelled = []
    previous_handlers = {}

    def cancel(signum, frame):
        cancelled.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, cancel)
    result = {"command": command, "started_unix": started,
              "scope": "new_child_process_group_only", "signals_sent": []}
    child = None
    try:
        with (output / "process.log").open("wb") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            result.update(pid=child.pid, pgid=child.pid)
            (output / "identity.json").write_text(json.dumps(result, indent=2) + "\n")
            offset, suffix = 0, b""
            deadline = time.monotonic() + timeout
            reason, matched = None, None
            while True:
                with (output / "process.log").open("rb") as source:
                    source.seek(offset)
                    # Bounded read; also scan any remaining log before success.
                    data = source.read(1024 * 1024)
                    offset += len(data)
                window = suffix + data
                hit = ERROR.search(window.decode("utf-8", errors="replace"))
                suffix = window[-256:]
                if hit:
                    reason, matched = "ERROR_LOG", hit.group(0)
                    break
                if cancelled:
                    reason = "CANCELLED"
                    break
                if time.monotonic() >= deadline:
                    reason = "TIMEOUT"
                    break
                # Observe exit without reaping: retain the leader PID until all
                # group cleanup finishes, so a recycled PID cannot be targeted.
                status = os.waitid(os.P_PID, child.pid,
                                   os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if status is not None:
                    if len(data) == 1024 * 1024:
                        continue
                    code = status.si_status if status.si_code == os.CLD_EXITED else -status.si_status
                    reason = "EXIT_ZERO" if code == 0 else "EXIT_NONZERO"
                    break
                time.sleep(interval)
            result.update(reason=reason, matched_error=matched)
            # Also clean workers if the leader already exited. The unreaped
            # leader reserves the PGID throughout TERM/grace/KILL cleanup.
            os.killpg(child.pid, signal.SIGTERM)
            result["signals_sent"].append("SIGTERM")
            time.sleep(grace)
            os.killpg(child.pid, signal.SIGKILL)
            result["signals_sent"].append("SIGKILL")
            child.wait()
            result["returncode"] = child.returncode
    finally:
        # Also clean our live child after a guard exception.
        if child is not None and child.returncode is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            result["signals_sent"].append("SIGKILL_EXCEPTION")
        result["ended_unix"] = time.time()
        result["success"] = result.get("reason") == "EXIT_ZERO"
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run(command, args.output, args.timeout_seconds)
    print(json.dumps(result))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
