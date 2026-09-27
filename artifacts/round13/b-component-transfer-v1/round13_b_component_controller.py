#!/usr/bin/env python3
"""Launch one frozen Round13 B QSA Decode component block on physical12."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback


ROOT = Path("/mnt/data1/kt-round9")
IMAGE = "601e76d17e9f"
TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MODEL = Path("/home/admin/models/Qwen3.8-Flash-Next") / REVISION
IDENTITY = Path("/home/admin/models/Qwen3.8-Flash-Next") / (
    REVISION
    + ".round9-support/destination-verification/verification-1789886190714568317.json"
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(values, timeout=30):
    return subprocess.check_output(values, text=True, timeout=timeout)


def validate_admission(document, now=None):
    now = time.time() if now is None else now
    if (
        document.get("status") != "PASS"
        or document.get("windows") != 12
        or document.get("training_id") != TRAINING_ID
        or document.get("selected_physical_cores") != 160
        or not isinstance(document.get("ended_at"), (int, float))
        or not 0 <= now - document["ended_at"] <= 900
        or float(document.get("min_mean_idle", 0)) <= 0.80
        or len(document.get("cpus", [])) != 160
        or len(set(document.get("cpus", []))) != 160
    ):
        raise ValueError("Round13 B component admission is invalid or stale")
    numa = document.get("min_numa_free_gib", {})
    if set(numa) != {"2", "3", "4", "7"} or any(
        float(numa[node]) < 48 for node in numa
    ):
        raise ValueError("Round13 B component NUMA admission failed")
    return list(map(int, document["cpus"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=(215, 216), required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--numa-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    block = {215: "A", 216: "B"}[args.version]
    candidate = args.candidate.resolve(strict=True)
    if candidate != ROOT / "round13-b-qsa-component-v1":
        raise ValueError("Round13 B component root is not frozen")
    contract = json.loads((candidate / "CONTRACT.json").read_text())
    if contract.get("versions", {}).get(str(args.version)) != block:
        raise ValueError("Round13 B component contract/version mismatch")
    manifest_path = candidate / "source-sha256.json"
    expected = json.loads(manifest_path.read_text())
    actual = {
        str(path.relative_to(candidate)): sha(path)
        for path in sorted(candidate.rglob("*"))
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
    }
    if actual != expected:
        raise RuntimeError("Round13 B component source manifest mismatch")
    admission = json.loads(args.admission.read_text())
    cpus = validate_admission(admission)
    if args.output.exists():
        raise FileExistsError(args.output)
    output = args.output.resolve()
    source = output / "source"
    source.mkdir(parents=True)
    shutil.copy2(args.admission, output / "admission.json")
    shutil.copy2(args.monitor, source / "round13_monitor_owned.py")
    shutil.copy2(args.numa_policy, source / "numa_policy.py")
    name = f"kt-round13-b-profile-{block.lower()}-v{args.version}"
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; "
        "exec timeout -k 10 1800 python /candidate/round13_b_qsa_component_probe.py "
        f"--model /models/{REVISION} --identity-receipt /identity/verification.json "
        f"--block {block} --iterations {contract['iterations']} "
        "--output /out/result.json > /out/probe.log 2>&1"
    )
    create = [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        "none",
        "--restart",
        "no",
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--cpuset-mems",
        "2,3,4,7",
        "--cpus",
        "136",
        "--memory",
        "64g",
        "--memory-swap",
        "64g",
        "--pids-limit",
        "2048",
        "--shm-size",
        "1g",
        "--cap-add",
        "SYS_NICE",
    ]
    for device in ("davinci12", "davinci_manager", "devmm_svm", "hisi_hdc"):
        create += ["--device", f"/dev/{device}"]
    for mount in (
        f"{candidate}:/candidate:ro",
        f"{output}:/out",
        f"{MODEL}:/models/{REVISION}:ro",
        f"{IDENTITY}:/identity/verification.json:ro",
        "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    ):
        create += ["-v", mount]
    create += [
        "-e",
        "PYTHONPATH=/candidate",
        "-e",
        "ASCEND_RT_VISIBLE_DEVICES=0",
        "-e",
        "OMP_NUM_THREADS=2",
        "-e",
        "OPENBLAS_NUM_THREADS=2",
        "-e",
        "MKL_NUM_THREADS=2",
        "--entrypoint",
        "bash",
        IMAGE,
        "-c",
        script,
    ]
    report = {
        "schema": 1,
        "status": "RUNNING",
        "version": args.version,
        "block": block,
        "container": name,
        "started_at": time.time(),
        "automatic_retries": 0,
        "candidate_manifest_sha256": sha(manifest_path),
        "admission_sha256": sha(args.admission),
    }
    created = False
    guard = None
    guard_log = None
    try:
        container_id = command(create).strip()
        created = True
        inspect = json.loads(command(["docker", "inspect", name]))[0]
        (output / "container-created.json").write_text(
            json.dumps(inspect, indent=2) + "\n"
        )
        if (
            inspect["Id"] != container_id
            or inspect["HostConfig"]["NanoCpus"] != 136_000_000_000
            or inspect["HostConfig"]["Memory"] != 64 * 1024**3
            or inspect["HostConfig"]["CpusetMems"] != "2,3,4,7"
        ):
            raise RuntimeError("Round13 B component container identity mismatch")
        guard_command = [
            "python3",
            str(source / "round13_monitor_owned.py"),
            "--container",
            name,
            "--output",
            str(output / "monitor.jsonl"),
            "--ready-file",
            str(output / "guard-ready.json"),
            "--resource-mode",
            "shared",
            "--seconds",
            "1800",
            "--owned-limit-mib",
            "11264",
            "--numa-nodes",
            "2",
            "3",
            "4",
            "7",
            "--training-container-id",
            TRAINING_ID,
        ]
        guard_log = (output / "guard.log").open("xb")
        guard = subprocess.Popen(
            guard_command,
            stdout=guard_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 60
        while not (output / "guard-ready.json").exists():
            if guard.poll() is not None:
                raise RuntimeError("Round13 B component guard exited before ready")
            if time.monotonic() >= deadline:
                raise TimeoutError("Round13 B component guard readiness timeout")
            time.sleep(0.2)
        validate_admission(admission)
        command(["docker", "start", name])
        report["container_id"] = container_id
        exit_code = int(command(["docker", "wait", name], timeout=1800).strip())
        guard.wait(timeout=60)
        guard_log.close()
        guard_log = None
        rows = [
            json.loads(line)
            for line in (output / "monitor.jsonl").read_text().splitlines()
            if line.strip()
        ]
        exit_ack = next(
            (
                row
                for row in reversed(rows)
                if row.get("action", "").startswith("OWNED_EXITED")
            ),
            None,
        )
        result = json.loads((output / "result.json").read_text())
        state = json.loads(command(["docker", "inspect", name]))[0]["State"]
        if (
            exit_code
            or state.get("OOMKilled") is not False
            or exit_ack is None
            or result.get("status") != "PASS_ROUND13_B_QSA_DECODE_COMPONENT"
            or result.get("block") != block
        ):
            raise RuntimeError("Round13 B component result failed")
        command(["docker", "rm", container_id])
        created = False
        proc = command(
            ["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"]
        )
        (output / "post-cleanup-proc-mem.txt").write_text(proc)
        if "No process in device" not in proc:
            raise RuntimeError("physical12 not empty")
        report.update(
            status="PASS_ROUND13_B_COMPONENT_CONTROLLER",
            ended_at=time.time(),
            exit_code=exit_code,
            guard_exit_ack=exit_ack,
            owned_container_removed=True,
            physical12_after="EMPTY",
            result_sha256=sha(output / "result.json"),
        )
    except BaseException as error:
        report.update(
            status="FAIL_ROUND13_B_COMPONENT_CONTROLLER",
            ended_at=time.time(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if created:
            try:
                state = json.loads(command(["docker", "inspect", name]))[0][
                    "State"
                ]
                if state.get("Running"):
                    subprocess.run(
                        ["docker", "stop", "-t", "10", name], timeout=30
                    )
            except BaseException:
                pass
            if guard is not None:
                try:
                    guard.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    guard.terminate()
            subprocess.run(["docker", "rm", name], timeout=30)
        raise
    finally:
        if guard_log is not None:
            guard_log.close()
        (output / "controller.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], args.version, block, flush=True)


if __name__ == "__main__":
    main()
