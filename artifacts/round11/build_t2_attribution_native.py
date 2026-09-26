"""Build one attribution-only native runtime from the frozen candidate build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import re


BASELINE_NATIVE_SHA256 = "614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d"
IMAGE = "601e76d17e9f"
EXPECTED_FLAGS = {
    "CMAKE_BUILD_TYPE": "Release",
    "CPUINFER_ENABLE_LTO": "OFF",
    "KTRANSFORMERS_USE_ASCEND_NPU": "ON",
    "LLAMA_ARM_DOTPROD": "ON",
    "LLAMA_ARM_FP16": "ON",
    "LLAMA_ARM_SVE": "OFF",
    "LLAMA_ARM_BF16": "OFF",
    "LLAMA_ARM_I8MM": "OFF",
    "LLAMA_NATIVE": "OFF",
    "PYTHON_EXECUTABLE": "/usr/local/python3.11.15/bin/python3",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_cache_flags(raw):
    found = {}
    for line in raw.splitlines():
        if "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in EXPECTED_FLAGS:
            found[key] = value
    if found != EXPECTED_FLAGS:
        raise ValueError(f"CMake cache flag mismatch: {found}")
    return found


def runtime_manifest(root, *, skip_native=False):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(root))
        if skip_native and relative.startswith("kt_kernel/kt_kernel_ext.") and relative.endswith(".so"):
            continue
        result[relative] = sha256(path)
    return result


def build_command(source: Path, build: Path, identity: str):
    """Reuse the frozen cache at the absolute build path recorded by CMake."""
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        f"kt-round11-t2-attribution-build-{identity}",
        "--network",
        "none",
        "--cpus",
        "8",
        "--memory",
        "32g",
        "--memory-swap",
        "32g",
        "--pids-limit",
        "2048",
        "-v",
        f"{source}:/round9/candidate-native-src:ro",
        "-v",
        f"{build}:/build",
        "-w",
        "/build",
        IMAGE,
        "bash",
        "-lc",
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; exec cmake --build . --parallel 8",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/mnt/data1/kt-round9"))
    parser.add_argument("--source-builder", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--identity", default="v1")
    args = parser.parse_args()
    if re.fullmatch(r"v[1-9][0-9]*", args.identity) is None:
        raise ValueError("attribution build identity must be vN")
    base = args.base.resolve(strict=True)
    baseline_source = base / "candidate-native-src"
    baseline_build = base / "candidate-native-build"
    baseline_runtime = base / "candidate-runtime-python"
    source = base / f"round11-t2-attribution-native-src-{args.identity}"
    build = base / f"round11-t2-attribution-native-build-{args.identity}"
    runtime = base / f"round11-t2-attribution-runtime-{args.identity}"
    source_receipt = base / f"round11-t2-attribution-source-{args.identity}.json"
    build_log = base / f"round11-t2-attribution-build-{args.identity}.log"
    report = {
        "schema": 1,
        "status": "RUNNING",
        "started_at": time.time(),
        "baseline_source": str(baseline_source),
        "baseline_build": str(baseline_build),
        "baseline_runtime": str(baseline_runtime),
        "attribution_source": str(source),
        "attribution_build": str(build),
        "attribution_runtime": str(runtime),
        "image": IMAGE,
        "optimization_candidate": False,
        "automatic_retries": 0,
    }
    if args.receipt.exists() or any(path.exists() for path in (source, build, runtime, source_receipt, build_log)):
        raise FileExistsError("Round11 T2 attribution build identity already exists")
    try:
        baseline_native = next((baseline_runtime / "kt_kernel").glob("kt_kernel_ext.*.so"))
        if sha256(baseline_native) != BASELINE_NATIVE_SHA256:
            raise ValueError("frozen baseline native identity changed")
        flags = validate_cache_flags((baseline_build / "CMakeCache.txt").read_text())
        subprocess.run(
            [
                "python3",
                str(args.source_builder.resolve(strict=True)),
                "--source",
                str(baseline_source),
                "--output",
                str(source),
                "--receipt",
                str(source_receipt),
            ],
            check=True,
            timeout=900,
        )
        shutil.copytree(baseline_build, build, symlinks=True)
        command = build_command(source, build, args.identity)
        report["build_command"] = command
        with build_log.open("xb") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
        built = list(build.glob("kt_kernel_ext.*.so"))
        if len(built) != 1:
            raise RuntimeError("attribution build did not produce exactly one extension")
        shutil.copytree(baseline_runtime, runtime, symlinks=True)
        target = runtime / "kt_kernel" / built[0].name
        shutil.copy2(built[0], target)
        if runtime_manifest(baseline_runtime, skip_native=True) != runtime_manifest(runtime, skip_native=True):
            raise RuntimeError("attribution runtime changed non-native payload")
        attribution_sha = sha256(target)
        if attribution_sha == BASELINE_NATIVE_SHA256:
            raise RuntimeError("attribution native binary did not change")
        report.update(
            status="PASS_ROUND11_T2_ATTRIBUTION_BUILD",
            ended_at=time.time(),
            cmake_flags=flags,
            baseline_native_sha256=BASELINE_NATIVE_SHA256,
            attribution_native_sha256=attribution_sha,
            attribution_native_path=str(target),
            source_receipt_sha256=sha256(source_receipt),
            build_log_sha256=sha256(build_log),
            semantic_scope="ENV_GATED_TASK_QUEUE_TIMESTAMPS_ONLY",
        )
    except BaseException as error:
        report.update(
            status="FAIL_ROUND11_T2_ATTRIBUTION_BUILD",
            ended_at=time.time(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["attribution_native_sha256"], flush=True)


if __name__ == "__main__":
    main()
