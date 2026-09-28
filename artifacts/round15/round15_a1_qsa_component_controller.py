#!/usr/bin/env python3
"""Launch, guard, and retire one frozen Round15 QSA component process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback


TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
BASE = Path("/mnt/data1/kt-round9")
IMAGE = "kt-round64-migration:20260916"
IMAGE_ID = "sha256:601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MODEL = Path("/home/admin/models/Qwen3.8-Flash-Next") / REVISION
IDENTITY = Path(
    "/home/admin/models/Qwen3.8-Flash-Next/"
    f"{REVISION}.round9-support/destination-verification/"
    "verification-1789886190714568317.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "source-sha256.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def validate_admission(document, now=None):
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
        or set(numa) != {"2", "3", "4", "7"}
        or any(float(numa[node]) < 48 for node in numa)
    ):
        raise RuntimeError("Round15 A1 admission invalid or stale")
    return [int(cpu) for cpu in cpus]


def command(args, timeout=30, stderr=None):
    return subprocess.check_output(args, text=True, timeout=timeout, stderr=stderr)


def load_monitor(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    candidate = Path(__file__).resolve().parent
    contract = json.loads((candidate / "ROUND15_A1_QSA_COMPONENT_CONTRACT.json").read_text())
    version, block = contract["version"], contract["block"]
    if (version, block) not in ((339, "A"), (340, "B")):
        raise RuntimeError("Round15 A1 component identity mismatch")
    manifest_path = candidate / "source-sha256.json"
    if tree(candidate) != json.loads(manifest_path.read_text()):
        raise RuntimeError("Round15 A1 component snapshot manifest mismatch")
    if command(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], timeout=30).strip() != IMAGE_ID:
        raise RuntimeError("Round15 A1 frozen image identity mismatch")
    cpus = validate_admission(json.loads(args.admission.read_text()))
    out = BASE / "evidence" / contract["evidence"]
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "admission.json").write_bytes(args.admission.read_bytes())
    name = contract["container"]
    report = {
        "schema": 1,
        "status": "RUNNING",
        "version": version,
        "block": block,
        "container": name,
        "started_at": time.time(),
        "automatic_retries": 0,
        "snapshot_manifest_sha256": sha256(manifest_path),
        "image_id": IMAGE_ID,
    }
    report_path = out / "controller.json"
    created = False
    guard = None
    guard_log = None
    container_id = None
    try:
        script = (
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; "
            "exec timeout -k 10 1800 python /candidate/round15_a1_qsa_component_probe.py "
            f"--model /models/{REVISION} --identity-receipt /identity/verification.json "
            f"--version {version} --block {block} --output /out/result.json "
            "> /out/probe.log 2>&1"
        )
        create = [
            "docker", "create", "--name", name, "--network", "none", "--restart", "no",
            "--cpuset-cpus", ",".join(map(str, cpus)), "--cpuset-mems", "2,3,4,7",
            "--cpus", "136", "--memory", "96g", "--memory-swap", "96g",
            "--pids-limit", "2048", "--shm-size", "2g", "--cap-add", "SYS_NICE",
        ]
        for device in ("davinci12", "davinci_manager", "devmm_svm", "hisi_hdc"):
            create += ["--device", f"/dev/{device}"]
        for mount in (
            f"{candidate}:/candidate:ro",
            f"{out}:/out",
            f"{MODEL}:/models/{REVISION}:ro",
            f"{IDENTITY}:/identity/verification.json:ro",
            "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
        ):
            create += ["-v", mount]
        create += [
            "-e", "ASCEND_RT_VISIBLE_DEVICES=0",
            "-e", "OMP_NUM_THREADS=2",
            "-e", "OPENBLAS_NUM_THREADS=2",
            "-e", "MKL_NUM_THREADS=2",
            "--entrypoint", "bash", IMAGE, "-c", script,
        ]
        container_id = command(create, timeout=60).strip()
        created = True
        inspect = json.loads(command(["docker", "inspect", name]))[0]
        (out / "container-created.json").write_text(json.dumps(inspect, indent=2) + "\n")
        if (
            inspect["Id"] != container_id
            or inspect["Image"] != IMAGE_ID.removeprefix("sha256:")
            and inspect["Image"] != IMAGE_ID
            or inspect["HostConfig"]["NanoCpus"] != 136_000_000_000
            or inspect["HostConfig"]["Memory"] != 96 * 1024**3
            or inspect["HostConfig"]["CpusetMems"] != "2,3,4,7"
        ):
            raise RuntimeError("Round15 A1 component container identity mismatch")
        guard_log = (out / "guard.log").open("xb")
        guard = subprocess.Popen(
            [
                "python3", str(candidate / "monitor_owned_server.py"),
                "--container", name,
                "--output", str(out / "monitor.jsonl"),
                "--ready-file", str(out / "guard-ready.json"),
                "--resource-mode", "shared",
                "--seconds", "1800",
                "--owned-limit-mib", "11264",
                "--numa-nodes", "2", "3", "4", "7",
                "--training-container-id", TRAINING_ID,
            ],
            stdout=guard_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 60
        while not (out / "guard-ready.json").exists():
            if guard.poll() is not None:
                raise RuntimeError("Round15 A1 guard exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("Round15 A1 guard readiness timeout")
            time.sleep(0.2)
        validate_admission(json.loads(args.admission.read_text()))
        command(["docker", "start", name])
        exit_code = int(command(["docker", "wait", name], timeout=1800).strip())
        guard.wait(timeout=90)
        guard_log.close()
        guard_log = None
        state = json.loads(command(["docker", "inspect", name]))[0]["State"]
        monitor = load_monitor(out / "monitor.jsonl")
        exit_ack = next(
            (row for row in reversed(monitor) if str(row.get("action", "")).startswith("OWNED_EXITED")),
            None,
        )
        result = json.loads((out / "result.json").read_text())
        if (
            exit_code != 0
            or state.get("OOMKilled") is not False
            or exit_ack is None
            or result.get("status") != contract["process_pass_status"]
            or int(result.get("peak_npu_allocated", 1 << 60))
            > int(contract["allocator_peak_limit_mib"]) * 1024**2
        ):
            raise RuntimeError("Round15 A1 result/container/resource gate failed")
        command(["docker", "rm", name])
        created = False
        proc_mem = command(["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"])
        (out / "post-cleanup-proc-mem.txt").write_text(proc_mem)
        from round15_p0_control import classify_physical12_after, read_proc_cgroup

        physical = classify_physical12_after(proc_mem, read_proc_cgroup)
        report.update(
            status=contract["controller_pass_status"],
            ended_at=time.time(),
            container_id=container_id,
            exit_code=exit_code,
            oom_killed=False,
            guard_exit_ack=exit_ack,
            owned_container_removed=True,
            physical12_after=physical["status"],
            physical12_processes=physical["processes"],
            result_sha256=sha256(out / "result.json"),
            component_gate_pass=result.get("component_gate_pass"),
        )
    except BaseException as error:
        report.update(
            status="FAIL_ROUND15_A1_QSA_COMPONENT_CONTROLLER",
            ended_at=time.time(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if created:
            try:
                state = json.loads(command(["docker", "inspect", name]))[0]["State"]
                if state.get("Running"):
                    subprocess.run(
                        ["docker", "stop", "--time", "10", name],
                        check=False,
                        timeout=30,
                    )
                subprocess.run(["docker", "rm", name], check=False, timeout=30)
                report["owned_container_removed_after_failure"] = True
            except BaseException as cleanup_error:
                report["cleanup_error"] = repr(cleanup_error)
        raise
    finally:
        if guard is not None and guard.poll() is None:
            guard.terminate()
            try:
                guard.wait(timeout=10)
            except subprocess.TimeoutExpired:
                guard.kill()
        if guard_log is not None:
            guard_log.close()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report.get("component_gate_pass"), flush=True)


if __name__ == "__main__":
    main()
