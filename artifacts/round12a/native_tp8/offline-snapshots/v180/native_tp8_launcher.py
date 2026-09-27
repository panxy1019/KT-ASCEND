#!/usr/bin/env python3
"""Create and start one admitted native-SGLang TP8 server."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


BASE = Path("/mnt/data1/kt-round9")
IMAGE = "601e76d17e9f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, timeout=30):
    return subprocess.check_output(args, text=True, timeout=timeout)


def validate_admission(document: dict, *, now=None) -> list[int]:
    now = time.time() if now is None else now
    ended = document.get("ended_at")
    cpus = document.get("cpus")
    numa = document.get("min_numa_free_gib", {})
    if (
        document.get("status") != "PASS_NATIVE_TP8_ADMISSION"
        or document.get("windows") != 12
        or document.get("physical_devices") != list(range(8))
        or document.get("overlap_policy") != "NO_OTHER_KT_PERFORMANCE_CONTAINER"
        or document.get("selected_physical_cores") != 160
        or not isinstance(ended, (int, float)) or ended > now + 5 or now - ended > 900
        or float(document.get("min_mean_idle", 0)) <= 0.80
        or not isinstance(cpus, list) or len(cpus) != 160 or len(set(cpus)) != 160
        or set(numa) != {"2", "3", "4", "7"}
        or any(float(numa[node]) < 48 for node in numa)
    ):
        raise ValueError("native TP8 admission is invalid or stale")
    return [int(cpu) for cpu in cpus]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root.parent != BASE:
        raise ValueError("native TP8 snapshot must be a direct immutable Round9 root")
    contract = json.loads((root / "ROUND12A_NATIVE_TP8_CONTRACT.json").read_text())
    version = int(contract["version"])
    if contract.get("mode") != "NATIVE_TP8" or version not in (180, 181, 182, 183):
        raise ValueError("native TP8 launcher received the wrong contract")
    name = contract["container"]
    expected = f"kt-round12a-native-tp8-{contract['block'].lower()}-v{version}"
    if name != expected:
        raise ValueError("native TP8 container identity mismatch")
    out = BASE / "evidence" / contract["evidence"]
    if out.exists():
        raise FileExistsError(out)
    external = json.loads(args.admission.read_text())
    cpus = validate_admission(external)
    running_names = command(["docker", "ps", "--format", "{{.Names}}"]).splitlines()
    if any(item.startswith(("kt-round10", "kt-round11", "kt-round12")) for item in running_names):
        raise RuntimeError("native TP8 performance run cannot overlap another KT project container")
    sys.path.insert(0, str(root))
    from native_tp8_admission import parse_first8_processes, training_identity
    training_identity()
    if any(parse_first8_processes(command(["npu-smi", "info"]))[device] for device in range(8)):
        raise RuntimeError("physical0-7 acquired a process after admission")

    profile = root / "native-tp8-server-profile.json"
    identity = root / "ROUND12A_NATIVE_TP8_IDENTITY.json"
    framework = root / "native-tp8-framework-profile.json"
    manifest = root / "round12-source-sha256.json"
    for path in (profile, identity, framework, manifest):
        path.resolve(strict=True)
    profile_doc = json.loads(profile.read_text())
    if (
        profile_doc.get("tp_size") != 8 or profile_doc.get("cpu_offload_gb") != 0
        or profile_doc.get("kt_weight_path") is not None or profile_doc.get("mem_fraction_static") != 0.70
    ):
        raise RuntimeError("native TP8 server profile changed")
    out.mkdir(parents=True)
    ready = {
        "schema": 1, "status": "ROUND12A_NATIVE_TP8_READY", "issued_at": time.time(),
        "external_admission_sha256": sha256(args.admission),
        "external_admission_ended_at": external["ended_at"],
        "profile_sha256": sha256(profile), "snapshot_manifest_sha256": sha256(manifest),
        "identity_sha256": sha256(identity), "framework_profile_sha256": sha256(framework),
        "physical_devices": list(range(8)), "per_device_hbm_ceiling_mib": 56832,
        "per_device_hbm_reserve_mib": 8704, "host_cgroup_limit_gib": 640,
        "cpus": cpus, "numa_nodes": [2, 3, 4, 7],
        "overlap_policy": "NO_OVERLAP_WITH_PHYSICAL12_PERFORMANCE_WINDOWS",
    }
    (out / "admission.json").write_text(json.dumps(ready, indent=2) + "\n")
    (out / "external-admission.json").write_bytes(args.admission.read_bytes())
    relative = root.relative_to(BASE)
    container_root = f"/round9/{relative}"
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; "
        "exec timeout -k 10 21600 python -u "
        f"{container_root}/native_tp8_server.py --profile {container_root}/native-tp8-server-profile.json "
        "--resolved-profile /out/resolved-profile.json --admission /out/admission.json "
        "> /out/server.log 2>&1"
    )
    create = [
        "docker", "create", "--name", name, "--network", "none", "--restart", "no",
        "--cpuset-cpus", ",".join(map(str, cpus)), "--cpuset-mems", "2,3,4,7",
        "--cpus", "136", "--memory", "640g", "--memory-swap", "640g",
        "--pids-limit", "8192", "--shm-size", "32g", "--cap-add", "SYS_NICE",
    ]
    for device in [*(f"davinci{i}" for i in range(8)), "davinci_manager", "devmm_svm", "hisi_hdc"]:
        create += ["--device", f"/dev/{device}"]
    for mount in (
        f"{BASE}:/round9:ro", "/home/admin/models/Qwen3.8-Flash-Next:/models:ro",
        f"{out}:/out", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    ):
        create += ["-v", mount]
    environments = (
        "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", "HCCL_CONNECT_TIMEOUT=1800",
        "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=2",
        "ROUND12A_NATIVE_TP8_MODE=1", f"PYTHONPATH={container_root}/overlay-python",
        f"ROUND9_FRAMEWORK_PROFILE={container_root}/native-tp8-framework-profile.json",
        f"ROUND9_FRAMEWORK_PROFILE_SHA256={sha256(framework)}",
        f"ROUND12A_NATIVE_TP8_IDENTITY={container_root}/ROUND12A_NATIVE_TP8_IDENTITY.json",
        f"ROUND12A_NATIVE_TP8_IDENTITY_SHA256={sha256(identity)}",
        f"ROUND12A_SNAPSHOT_MANIFEST_SHA256={sha256(manifest)}",
        "ROUND9_DYNAMIC_BOUNDARY_ENABLE=0", "ROUND9_REQUEST_COST_ENABLE=0",
    )
    for environment in environments:
        create += ["-e", environment]
    create += ["--entrypoint", "bash", IMAGE, "-c", script]
    container_id = command(create).strip()
    inspect = json.loads(command(["docker", "inspect", name]))[0]
    (out / "container-created.json").write_text(json.dumps(inspect, indent=2) + "\n")
    if (
        inspect["Id"] != container_id or inspect["HostConfig"]["Memory"] != 640 * 1024**3
        or inspect["HostConfig"]["MemorySwap"] != 640 * 1024**3
        or inspect["HostConfig"]["NanoCpus"] != 136_000_000_000
        or inspect["HostConfig"]["CpusetMems"] != "2,3,4,7"
    ):
        raise RuntimeError("native TP8 container resource identity mismatch")
    guard_log = (out / "guard.log").open("x")
    guard = subprocess.Popen([
        "python3", str(root / "native_tp8_monitor.py"), "--container", name,
        "--output", str(out / "monitor.jsonl"), "--ready-file", str(out / "guard-ready.json"),
        "--seconds", "21600", "--numa-nodes", "2", "3", "4", "7",
    ], stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 90
    while not (out / "guard-ready.json").exists():
        if guard.poll() is not None:
            raise RuntimeError("native TP8 guard exited before readiness")
        if time.monotonic() >= deadline:
            raise TimeoutError("native TP8 guard readiness timeout")
        time.sleep(0.2)
    validate_admission(external)
    with (root / "execution-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget_path = root / "execution-budget.json"
        budget = json.loads(budget_path.read_text())
        if budget.get("stage") != contract["stage"] or budget.get("starts"):
            raise RuntimeError("native TP8 model-start budget is unavailable")
        start = {"mode": "NATIVE_TP8", "name": name, "container_id": container_id, "time": time.time(), "output": str(out)}
        budget["starts"] = [start]
        budget["model_starts_used"] = 1
        temporary = budget_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(budget, indent=2) + "\n")
        temporary.replace(budget_path)
        (out / "model-start.json").write_text(json.dumps(start, indent=2) + "\n")
    command(["docker", "start", name])
    print("STARTED_ROUND12A_NATIVE_TP8", name, container_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
