#!/usr/bin/env python3
"""Capture the small, non-secret runtime identity needed by Round15."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def command(values):
    try:
        return {
            "status": "PASS",
            "stdout": subprocess.check_output(
                values, text=True, timeout=30, stderr=subprocess.STDOUT
            ).strip(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"status": "UNAVAILABLE", "error": repr(error)}


def package(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os_release = {}
    release = Path("/etc/os-release")
    if release.is_file():
        for line in release.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip('"')
    result = {
        "schema": 1,
        "status": "PASS_ROUND15_RUNTIME_ENVIRONMENT_PROBE",
        "scope": args.scope,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "libc": platform.libc_ver(),
        "os_release": os_release,
        "packages": {"torch": package("torch"), "torch_npu": package("torch-npu")},
        "docker_server": command(["docker", "version", "--format", "{{json .Server}}"]),
    }
    if args.image:
        inspected = command(["docker", "image", "inspect", args.image])
        if inspected["status"] == "PASS":
            document = json.loads(inspected["stdout"])[0]
            inspected = {
                "status": "PASS",
                "id": document.get("Id"),
                "repo_digests": document.get("RepoDigests"),
                "os": document.get("Os"),
                "architecture": document.get("Architecture"),
                "created": document.get("Created"),
            }
        result["frozen_image"] = inspected
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"], args.scope, flush=True)


if __name__ == "__main__":
    main()
