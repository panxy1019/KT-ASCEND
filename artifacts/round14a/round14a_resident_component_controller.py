#!/usr/bin/env python3
"""Launch and retire one frozen Round14a residency component on physical12."""

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
    REVISION + ".round9-support/destination-verification/verification-1789886190714568317.json")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def command(values, timeout=30):
    return subprocess.check_output(values, text=True, timeout=timeout)


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "source-sha256.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def validate_admission(document, now=None):
    now = time.time() if now is None else now
    ended = document.get("ended_at")
    cpus = document.get("cpus")
    numa = document.get("min_numa_free_gib", {})
    if (document.get("status") != "PASS" or document.get("windows") != 12 or
            document.get("training_id") != TRAINING_ID or
            document.get("selected_physical_cores") != 160 or
            not isinstance(ended, (int, float)) or ended > now + 5 or now - ended > 900 or
            float(document.get("min_mean_idle", 0)) <= .80 or
            not isinstance(cpus, list) or len(cpus) != 160 or len(set(cpus)) != 160 or
            set(numa) != {"2", "3", "4", "7"} or
            any(float(numa[node]) < 48 for node in numa)):
        raise RuntimeError("Round14a residency component admission invalid or stale")
    return list(map(int, cpus))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    contract = json.loads((candidate / "CONTRACT.json").read_text())
    run = contract["runs"].get(str(args.version))
    if run is None:
        raise ValueError("version outside frozen Round14a residency component matrix")
    manifest_path = candidate / "source-sha256.json"
    if tree(candidate) != json.loads(manifest_path.read_text()):
        raise RuntimeError("Round14a residency component source manifest mismatch")
    admission = json.loads(args.admission.read_text())
    cpus = validate_admission(admission)
    if args.output.exists():
        raise FileExistsError(args.output)
    output = args.output.resolve()
    output.mkdir(parents=True)
    shutil.copy2(args.admission, output / "admission.json")
    strategy, layer = run["strategy"], run["layer"]
    name = f"kt-round14a-b-{strategy}-l{layer}-v{args.version}"
    bank = f"layer-{layer:02d}-mxfp4.gguf"
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; "
        "exec timeout -k 10 1800 python /candidate/round14a_resident_component_probe.py "
        f"--model /models/{REVISION} --identity-receipt /identity/verification.json "
        f"--bank /round9/banks/candidate-mxfp4-all48-v1/{bank} "
        "--bank-catalog /round9/banks/candidate-mxfp4-all48-v1/catalog.json "
        f"--profiles /candidate/ROUND14A_RESIDENT_PROFILES.json "
        f"--profiles-sha256 {contract['resident_profiles_sha256']} "
        f"--strategy {strategy} --layer {layer} --output /out/result.json "
        "> /out/probe.log 2>&1"
    )
    create = ["docker", "create", "--name", name, "--network", "none", "--restart", "no",
        "--cpuset-cpus", ",".join(map(str, cpus)), "--cpuset-mems", "2,3,4,7",
        "--cpus", "136", "--memory", "96g", "--memory-swap", "96g",
        "--pids-limit", "2048", "--shm-size", "2g", "--cap-add", "SYS_NICE"]
    for device in ("davinci12", "davinci_manager", "devmm_svm", "hisi_hdc"):
        create += ["--device", f"/dev/{device}"]
    for mount in (f"{candidate}:/candidate:ro", f"{output}:/out",
            f"{MODEL}:/models/{REVISION}:ro", f"{IDENTITY}:/identity/verification.json:ro",
            f"{ROOT}:/round9:ro", f"{ROOT}/round12a-kt-b-v177/overlay-python:/overlay:ro",
            "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"):
        create += ["-v", mount]
    create += ["-e", "PYTHONPATH=/candidate:/overlay:/round9/candidate-runtime-python",
        "-e", "ASCEND_RT_VISIBLE_DEVICES=0", "-e", "OMP_NUM_THREADS=2",
        "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
        "--entrypoint", "bash", IMAGE, "-c", script]
    report = {"schema": 1, "status": "RUNNING", "version": args.version,
        "strategy": strategy, "layer": layer, "capacity": run["capacity"],
        "result_key": run["result_key"], "container": name, "started_at": time.time(),
        "automatic_retries": 0, "candidate_manifest_sha256": sha(manifest_path),
        "admission_sha256": sha(args.admission)}
    created = False
    guard = None
    guard_log = None
    try:
        container_id = command(create).strip(); created = True
        inspect = json.loads(command(["docker", "inspect", name]))[0]
        (output / "container-created.json").write_text(json.dumps(inspect, indent=2) + "\n")
        if (inspect["Id"] != container_id or
                inspect["HostConfig"]["NanoCpus"] != 136_000_000_000 or
                inspect["HostConfig"]["Memory"] != 96 * 1024**3 or
                inspect["HostConfig"]["CpusetMems"] != "2,3,4,7"):
            raise RuntimeError("Round14a residency component container identity mismatch")
        guard_log = (output / "guard.log").open("xb")
        guard = subprocess.Popen(["python3", str(args.monitor), "--container", name,
            "--output", str(output / "monitor.jsonl"),
            "--ready-file", str(output / "guard-ready.json"), "--resource-mode", "shared",
            "--seconds", "1800", "--owned-limit-mib", "11264", "--numa-nodes", "2", "3", "4", "7",
            "--training-container-id", TRAINING_ID], stdout=guard_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 60
        while not (output / "guard-ready.json").exists():
            if guard.poll() is not None: raise RuntimeError("component guard exited before ready")
            if time.monotonic() >= deadline: raise TimeoutError("component guard ready timeout")
            time.sleep(.2)
        validate_admission(admission)
        command(["docker", "start", name])
        exit_code = int(command(["docker", "wait", name], timeout=1800).strip())
        guard.wait(timeout=60); guard_log.close(); guard_log = None
        rows = [json.loads(line) for line in (output / "monitor.jsonl").read_text().splitlines() if line.strip()]
        exit_ack = next((row for row in reversed(rows) if row.get("action", "").startswith("OWNED_EXITED")), None)
        result = json.loads((output / "result.json").read_text())
        state = json.loads(command(["docker", "inspect", name]))[0]["State"]
        if (exit_code or state.get("OOMKilled") is not False or exit_ack is None or
                result.get("status") != "PASS_ROUND14A_RESIDENT_COMPONENT" or
                result.get("capacity") != run["capacity"] or
                int(result.get("peak_allocated_bytes", 1 << 60)) > 10240 * 1024**2):
            raise RuntimeError("Round14a residency component result failed")
        command(["docker", "rm", container_id]); created = False
        proc = command(["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"])
        (output / "post-cleanup-proc-mem.txt").write_text(proc)
        if "No process in device" not in proc: raise RuntimeError("physical12 not empty")
        report.update(status="PASS_ROUND14A_B_RESIDENT_COMPONENT_CONTROLLER", ended_at=time.time(),
            exit_code=exit_code, guard_exit_ack=exit_ack, owned_container_removed=True,
            physical12_after="EMPTY", result_sha256=sha(output / "result.json"))
    except BaseException as error:
        report.update(status="FAIL_ROUND14A_B_RESIDENT_COMPONENT_CONTROLLER", ended_at=time.time(),
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if created:
            try:
                state = json.loads(command(["docker", "inspect", name]))[0]["State"]
                if state.get("Running"): subprocess.run(["docker", "stop", "-t", "10", name], timeout=30)
            except BaseException: pass
            if guard is not None:
                try: guard.wait(timeout=40)
                except subprocess.TimeoutExpired: guard.terminate()
            subprocess.run(["docker", "rm", name], timeout=30)
        raise
    finally:
        if guard_log is not None: guard_log.close()
        (output / "controller.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], args.version, strategy, layer, flush=True)


if __name__ == "__main__":
    main()
