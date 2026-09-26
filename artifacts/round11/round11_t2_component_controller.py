"""Launch one frozen Round11 T2 component process under the physical12 guard."""

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
ROLE_CONTRACT = {
    "baseline": (113, "OFF", "baseline"),
    "counters": (114, "ON", "baseline"),
    "attribution": (115, "OFF", "attribution"),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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
        raise ValueError("Round11 T2 component admission identity is invalid or stale")
    if set(numa) != {"2", "3", "4", "7"} or any(float(numa[node]) < 48 for node in numa):
        raise ValueError("Round11 T2 component NUMA admission is below 48 GiB")
    return [int(cpu) for cpu in cpus]


def _container_runtime_path(runtime: Path) -> str:
    relative = runtime.absolute().relative_to(ROOT.absolute())
    return "/round9/" + str(relative)


def build_container_command(*, role, version, output, cpus, runtime):
    expected_version, counters, runtime_role = ROLE_CONTRACT[role]
    if version != expected_version:
        raise ValueError(f"{role} must use frozen version {expected_version}")
    runtime_in_container = _container_runtime_path(runtime)
    name = f"kt-round11-t2-{role}-v{version}"
    attribution = (
        " --attribution-build-receipt /round9/round11-t2-attribution-build-v2.json"
        if runtime_role == "attribution"
        else ""
    )
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "ulimit -c 0; exec timeout -k 10 1800 python "
        "/out/source/round11_t2_attribution_probe.py "
        "--bank /round9/banks/candidate-layer0-mxfp4-v1/layer-00-mxfp4.gguf "
        "--capture /round9/evidence/20260925-plan15-p4a-v80/real-prefill.pt "
        "--output /out/result.json --phase-log /out/phase.log "
        "--admission /out/admission.json --iterations 3000 "
        f"--hardware-counters {counters}{attribution} > /out/probe.log 2>&1"
    )
    command = [
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
        command += ["--device", f"/dev/{device}"]
    for mount in (
        f"{ROOT}:/round9:ro",
        f"{output}:/out",
        "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    ):
        command += ["-v", mount]
    command += [
        "-e",
        f"PYTHONPATH={runtime_in_container}",
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
    return command


def command(arguments, *, timeout=30):
    return subprocess.check_output(arguments, text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(ROLE_CONTRACT), required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--numa-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_version, _counters, runtime_role = ROLE_CONTRACT[args.role]
    if args.version != expected_version:
        raise ValueError(f"{args.role} must use frozen version {expected_version}")
    if args.output.exists():
        raise FileExistsError(args.output)
    admission = json.loads(args.admission.read_text())
    cpus = validate_admission(admission)
    runtime = ROOT / (
        "round11-t2-attribution-runtime-v2"
        if runtime_role == "attribution"
        else "candidate-runtime-python"
    )
    runtime.resolve(strict=True)
    output = args.output.resolve()
    source = output / "source"
    source.mkdir(parents=True)
    shutil.copy2(args.admission, output / "admission.json")
    shutil.copy2(args.probe, source / "round11_t2_attribution_probe.py")
    shutil.copy2(args.monitor, source / "round11_monitor_owned.py")
    shutil.copy2(args.numa_policy, source / "numa_policy.py")
    name = f"kt-round11-t2-{args.role}-v{args.version}"
    report = {
        "schema": 1,
        "status": "RUNNING",
        "role": args.role,
        "version": args.version,
        "container": name,
        "started_at": time.time(),
        "automatic_retries": 0,
        "source_sha256": {
            path.name: sha256(path) for path in sorted(source.iterdir()) if path.is_file()
        },
        "admission_sha256": sha256(output / "admission.json"),
    }
    container_created = False
    guard = None
    try:
        create = build_container_command(
            role=args.role,
            version=args.version,
            output=output,
            cpus=cpus,
            runtime=runtime,
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
            raise RuntimeError("T2 component container identity mismatch")
        guard_command = [
            "python3",
            str(source / "round11_monitor_owned.py"),
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
                raise RuntimeError("T2 component guard exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("T2 component guard readiness timeout")
            time.sleep(0.2)
        validate_admission(admission)
        command(["docker", "start", name])
        report["container_id"] = container_id
        exit_code = int(command(["docker", "wait", name], timeout=1800).strip())
        guard.wait(timeout=60)
        guard_log.close()
        state = json.loads(command(["docker", "inspect", name]))[0]["State"]
        monitor_rows = [
            json.loads(line)
            for line in (output / "monitor.jsonl").read_text().splitlines()
            if line.strip()
        ]
        exit_ack = next(
            (row for row in reversed(monitor_rows) if row.get("action", "").startswith("OWNED_EXITED")),
            None,
        )
        if exit_ack is None:
            raise RuntimeError("T2 component guard exit ACK missing")
        result = json.loads((output / "result.json").read_text())
        if (
            exit_code != 0
            or state.get("OOMKilled") is not False
            or result.get("status") != "PASS_ROUND11_T2_DEEP_ATTRIBUTION"
        ):
            raise RuntimeError("T2 component result/container failed")
        command(["docker", "rm", name])
        container_created = False
        proc_mem = command(["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"])
        (output / "post-cleanup-proc-mem.txt").write_text(proc_mem)
        if "No process in device" not in proc_mem:
            raise RuntimeError("physical12 is not empty after T2 component cleanup")
        report.update(
            status="PASS_ROUND11_T2_COMPONENT_CONTROLLER",
            ended_at=time.time(),
            exit_code=exit_code,
            oom_killed=False,
            guard_exit_ack=exit_ack,
            owned_container_removed=True,
            result_sha256=sha256(output / "result.json"),
            phase_log_sha256=sha256(output / "phase.log"),
            physical12_after="EMPTY",
        )
    except BaseException as error:
        report.update(
            status="FAIL_ROUND11_T2_COMPONENT_CONTROLLER",
            ended_at=time.time(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if container_created:
            subprocess.run(["docker", "stop", "--time", "10", name], check=False, timeout=30)
            if guard is not None:
                try:
                    guard.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    guard.terminate()
            subprocess.run(["docker", "rm", name], check=False, timeout=30)
        raise
    finally:
        (output / "controller.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], args.role, args.version, flush=True)


if __name__ == "__main__":
    main()
