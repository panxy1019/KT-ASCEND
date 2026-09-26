"""Launch the single frozen Round11 QSA component candidate on physical12."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback


TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
ROOT = Path("/mnt/data1/kt-round9")
IMAGE = "601e76d17e9f"
VERSION = 144
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MODEL_ROOT = Path("/home/admin/models/Qwen3.8-Flash-Next") / REVISION
IDENTITY_RECEIPT = Path(
    "/home/admin/models/Qwen3.8-Flash-Next/"
    f"{REVISION}.round9-support/destination-verification/"
    "verification-1789886190714568317.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_admission(document, *, now=None):
    now = time.time() if now is None else now
    ended = document.get("ended_at")
    numa = document.get("min_numa_free_gib", {})
    cpus = document.get("cpus")
    if (
        document.get("status") != "PASS"
        or document.get("windows") != 12
        or document.get("training_id") != TRAINING_ID
        or document.get("selected_physical_cores") != 160
        or not isinstance(ended, (int, float))
        or ended > now + 5
        or now - ended > 900
        or float(document.get("min_mean_idle", 0)) <= 0.80
        or not isinstance(cpus, list)
        or len(cpus) != 160
        or len(set(cpus)) != 160
    ):
        raise ValueError("Round11 T5 QSA component admission identity is invalid or stale")
    if set(numa) != {"2", "3", "4", "7"} or any(float(numa[node]) < 48 for node in numa):
        raise ValueError("Round11 T5 QSA component NUMA admission is below 48 GiB")
    return [int(cpu) for cpu in cpus]


def validate_candidate_manifest(candidate: Path):
    manifest_path = candidate / "source-sha256.json"
    expected = json.loads(manifest_path.read_text())
    actual = {
        str(path.relative_to(candidate)): sha256(path)
        for path in sorted(candidate.rglob("*"))
        if (
            path.is_file()
            and path != manifest_path
            and "__pycache__" not in path.parts
        )
    }
    if actual != expected:
        raise ValueError("QSA component candidate manifest mismatch")
    return expected


def stage_guard_sources(monitor: Path, numa_policy: Path, source: Path):
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(monitor, source / "round11_monitor_owned.py")
    shutil.copy2(numa_policy, source / "numa_policy.py")


def build_container_command(*, version, output, cpus, candidate):
    if version != VERSION:
        raise ValueError(f"QSA component candidate must use frozen version {VERSION}")
    name = f"kt-round11-t5-qsa-candidate-v{version}"
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "ulimit -c 0; exec timeout -k 10 1800 python "
        "/candidate/round11_t5_qsa_component_probe.py "
        f"--model /models/{REVISION} "
        "--identity-receipt /identity/verification.json "
        "--output /out/result.json --iterations 9 > /out/probe.log 2>&1"
    )
    command = [
        "docker", "create", "--name", name,
        "--network", "none", "--restart", "no",
        "--cpuset-cpus", ",".join(map(str, cpus)),
        "--cpuset-mems", "2,3,4,7", "--cpus", "136",
        "--memory", "64g", "--memory-swap", "64g",
        "--pids-limit", "2048", "--shm-size", "1g", "--cap-add", "SYS_NICE",
    ]
    for device in ("davinci12", "davinci_manager", "devmm_svm", "hisi_hdc"):
        command += ["--device", f"/dev/{device}"]
    for mount in (
        f"{candidate}:/candidate:ro",
        f"{output}:/out",
        f"{MODEL_ROOT}:/models/{REVISION}:ro",
        f"{IDENTITY_RECEIPT}:/identity/verification.json:ro",
        "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    ):
        command += ["-v", mount]
    command += [
        "-e", "PYTHONPATH=/candidate",
        "-e", "ASCEND_RT_VISIBLE_DEVICES=0",
        "-e", "OMP_NUM_THREADS=2",
        "-e", "OPENBLAS_NUM_THREADS=2",
        "-e", "MKL_NUM_THREADS=2",
        "--entrypoint", "bash", IMAGE, "-c", script,
    ]
    return command


def command(arguments, *, timeout=30):
    return subprocess.check_output(arguments, text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--numa-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.version != VERSION:
        raise ValueError(f"QSA component candidate must use frozen version {VERSION}")
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate = args.candidate.resolve(strict=True)
    if candidate != ROOT / f"round11-t5-qsa-component-v{VERSION}":
        raise ValueError("QSA component candidate root is not the frozen remote identity")
    candidate_manifest = validate_candidate_manifest(candidate)
    admission = json.loads(args.admission.read_text())
    cpus = validate_admission(admission)
    output = args.output.resolve()
    source = output / "source"
    stage_guard_sources(args.monitor, args.numa_policy, source)
    shutil.copy2(args.admission, output / "admission.json")
    name = f"kt-round11-t5-qsa-candidate-v{VERSION}"
    report = {
        "schema": 1,
        "status": "RUNNING",
        "version": VERSION,
        "container": name,
        "started_at": time.time(),
        "automatic_retries": 0,
        "candidate_sha256": candidate_manifest,
        "candidate_manifest_sha256": sha256(candidate / "source-sha256.json"),
        "monitor_sha256": sha256(source / "round11_monitor_owned.py"),
        "numa_policy_sha256": sha256(source / "numa_policy.py"),
        "admission_sha256": sha256(output / "admission.json"),
    }
    container_created = False
    guard = None
    guard_log = None
    try:
        create = build_container_command(
            version=VERSION,
            output=output,
            cpus=cpus,
            candidate=candidate,
        )
        container_id = command(create).strip()
        container_created = True
        inspect = json.loads(command(["docker", "inspect", name]))[0]
        (output / "container-created.json").write_text(json.dumps(inspect, indent=2) + "\n")
        if (
            inspect["Id"] != container_id
            or inspect["HostConfig"]["NanoCpus"] != 136_000_000_000
            or inspect["HostConfig"]["Memory"] != 64 * 1024**3
            or inspect["HostConfig"]["CpusetMems"] != "2,3,4,7"
        ):
            raise RuntimeError("T5 QSA component container identity mismatch")
        guard_command = [
            "python3", str(source / "round11_monitor_owned.py"),
            "--container", name,
            "--output", str(output / "monitor.jsonl"),
            "--ready-file", str(output / "guard-ready.json"),
            "--resource-mode", "shared", "--seconds", "1800",
            "--owned-limit-mib", "11264", "--numa-nodes", "2", "3", "4", "7",
            "--training-container-id", TRAINING_ID,
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
                raise RuntimeError("T5 QSA component guard exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("T5 QSA component guard readiness timeout")
            time.sleep(0.2)
        validate_admission(admission)
        command(["docker", "start", name])
        report["container_id"] = container_id
        exit_code = int(command(["docker", "wait", name], timeout=1800).strip())
        guard.wait(timeout=60)
        guard_log.close()
        guard_log = None
        state = json.loads(command(["docker", "inspect", name]))[0]["State"]
        monitor_rows = [
            json.loads(line)
            for line in (output / "monitor.jsonl").read_text().splitlines()
            if line.strip()
        ]
        exit_ack = next(
            (row for row in reversed(monitor_rows)
             if row.get("action", "").startswith("OWNED_EXITED")),
            None,
        )
        if exit_ack is None:
            raise RuntimeError("T5 QSA component guard exit ACK missing")
        result = json.loads((output / "result.json").read_text())
        if (
            exit_code != 0
            or state.get("OOMKilled") is not False
            or result.get("status") != "PASS_ROUND11_T5_QSA_COMPONENT"
        ):
            raise RuntimeError("T5 QSA component result/container failed")
        command(["docker", "rm", name])
        container_created = False
        proc_mem = command(["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"])
        (output / "post-cleanup-proc-mem.txt").write_text(proc_mem)
        if "No process in device" not in proc_mem:
            raise RuntimeError("physical12 is not empty after T5 QSA component cleanup")
        report.update(
            status="PASS_ROUND11_T5_QSA_COMPONENT_CONTROLLER",
            ended_at=time.time(), exit_code=exit_code, oom_killed=False,
            guard_exit_ack=exit_ack, owned_container_removed=True,
            result_sha256=sha256(output / "result.json"), physical12_after="EMPTY",
        )
    except BaseException as error:
        report.update(
            status="FAIL_ROUND11_T5_QSA_COMPONENT_CONTROLLER",
            ended_at=time.time(), error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if container_created:
            try:
                state = json.loads(command(["docker", "inspect", name]))[0]["State"]
                if state.get("Running"):
                    subprocess.run(["docker", "stop", "--time", "10", name], check=False, timeout=30)
            except BaseException:
                pass
            if guard is not None:
                try:
                    guard.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    guard.terminate()
            subprocess.run(["docker", "rm", name], check=False, timeout=30)
        raise
    finally:
        if guard_log is not None:
            guard_log.close()
        (output / "controller.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], VERSION, flush=True)


if __name__ == "__main__":
    main()
