#!/usr/bin/env python3
"""One-shot model-checkpoint cache retirement after loader references close."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from targeted_cache_retirement_v2 import (
    retire,
    scan_process_references,
    validate_registered_targets,
)


def container_running(name: str) -> bool:
    value = subprocess.check_output(
        ["docker", "inspect", "-f", "{{.State.Running}}", name], text=True
    ).strip()
    return value == "true"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=150)
    args = parser.parse_args()
    wait_path = args.output.with_name(args.output.stem + "-wait.json")
    if args.output.exists() or wait_path.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    paths = validate_registered_targets(manifest["targets"], expected_count=131)
    started = time.time()
    report = {
        "schema": 1,
        "status": "WAITING_FOR_MODEL_LOAD_REFERENCE_RETIREMENT",
        "started_unix": started,
        "container": args.container,
        "target_count": len(paths),
        "polls": [],
        "global_drop_caches": False,
        "bank_files_included": False,
    }
    try:
        while time.time() - started < args.timeout:
            running = container_running(args.container)
            text = args.server_log.read_text(errors="replace") if args.server_log.exists() else ""
            loaded = "131/131" in text
            references = scan_process_references(paths) if loaded and running else []
            report["polls"].append({
                "time": time.time(), "container_running": running,
                "model_load_marker": loaded, "active_references": len(references),
            })
            wait_path.write_text(json.dumps(report, indent=2) + "\n")
            if not running:
                raise RuntimeError("owned model container exited before cache retirement")
            if loaded and not references:
                result = retire(
                    args.manifest, args.output,
                    expected_count=131, minimum_drop_bytes=1 << 30,
                )
                report.update(
                    status="PASS_ROUND15_B_STARTUP_MODEL_CACHE_RETIREMENT",
                    ended_unix=time.time(),
                    retirement_status=result["status"],
                    resident_bytes_released=result["resident_bytes_released"],
                )
                wait_path.write_text(json.dumps(report, indent=2) + "\n")
                print(report["status"], report["resident_bytes_released"], flush=True)
                return
            time.sleep(1)
        raise TimeoutError("model checkpoint references did not become idle in time")
    except BaseException as error:
        report.update(
            status="FAIL_ROUND15_B_STARTUP_MODEL_CACHE_RETIREMENT",
            ended_unix=time.time(), error=f"{type(error).__name__}: {error}",
        )
        wait_path.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
