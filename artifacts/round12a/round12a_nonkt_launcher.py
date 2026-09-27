#!/usr/bin/env python3
"""Create and start one admitted Round12a non-KT OffloaderV1 server."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


BASE = Path("/mnt/data1/kt-round9")
IMAGE = "601e76d17e9f"
TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(arguments, *, timeout=30):
    return subprocess.check_output(arguments, text=True, timeout=timeout)


def validate_admission(document: dict, *, now: float | None = None) -> list[int]:
    now = time.time() if now is None else now
    ended = document.get("ended_at")
    cpus = document.get("cpus")
    numa = document.get("min_numa_free_gib", {})
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
        raise ValueError("Round12a non-KT external admission is invalid or stale")
    return [int(cpu) for cpu in cpus]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root.parent != BASE:
        raise ValueError("non-KT snapshot must be an immutable Round12a root")
    contract = json.loads((root / "ROUND12A_MODEL_CONTRACT.json").read_text())
    if (
        contract.get("mode") != "NON_KT"
        or contract.get("stage")
        not in {"P2_FIRST_NONKT_REQUEST", "P3_KT_NONKT_ABBA"}
    ):
        raise ValueError("non-KT launcher received the wrong model contract")
    version = int(contract["version"])
    name = contract["container"]
    expected_names = {
        173: "kt-round12a-nonkt-smoke-v173",
        175: "kt-round12a-nonkt-a-v175",
        176: "kt-round12a-nonkt-b-v176",
    }
    if name != expected_names.get(version):
        raise ValueError("non-KT container/version identity mismatch")
    out = BASE / "evidence" / contract["evidence"]
    if out.exists():
        raise FileExistsError(out)
    external = json.loads(args.admission.read_text())
    cpus = validate_admission(external)
    inspected_training = json.loads(command(["docker", "inspect", TRAINING_ID]))
    if (
        len(inspected_training) != 1
        or inspected_training[0]["Id"] != TRAINING_ID
        or not inspected_training[0]["State"]["Running"]
    ):
        raise RuntimeError("shared training container identity changed")

    profile = root / "nonkt-server-profile.json"
    identity = root / "ROUND12A_NONKT_IDENTITY.json"
    manifest = root / "round12-source-sha256.json"
    framework = root / "nonkt-framework-profile.json"
    context = root / "context-reference-profile.json"
    for required in (profile, identity, manifest, framework, context):
        required.resolve(strict=True)
    profile_doc = json.loads(profile.read_text())
    if profile_doc.get("cpu_offload_gb") != 233 or profile_doc.get("kt_weight_path") is not None:
        raise RuntimeError("non-KT server profile changed")

    out.mkdir(parents=True)
    ready = {
        "schema": 1,
        "status": "ROUND12A_NONKT_READY",
        "issued_at": time.time(),
        "external_admission_sha256": sha256(args.admission),
        "external_admission_ended_at": external["ended_at"],
        "training_container_id": TRAINING_ID,
        "profile_sha256": sha256(profile),
        "snapshot_manifest_sha256": sha256(manifest),
        "identity_sha256": sha256(identity),
        "framework_profile_sha256": sha256(framework),
        "context_reference_profile_sha256": sha256(context),
        "allocator_limit_mib": 36352,
        "owned_limit_mib": 37376,
        "reserve_hbm_mib": 8704,
        "cpu_offload_gib": 233,
        "cpus": cpus,
        "numa_nodes": [2, 3, 4, 7],
    }
    (out / "admission.json").write_text(json.dumps(ready, indent=2) + "\n")
    (out / "external-admission.json").write_bytes(args.admission.read_bytes())
    (out / "payload-sha256.json").write_text(
        json.dumps(
            {
                "snapshot_manifest": ready["snapshot_manifest_sha256"],
                "identity": ready["identity_sha256"],
                "profile": ready["profile_sha256"],
                "framework": ready["framework_profile_sha256"],
                "context": ready["context_reference_profile_sha256"],
            },
            indent=2,
        )
        + "\n"
    )

    relative = root.relative_to(BASE)
    container_root = f"/round9/{relative}"
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "ulimit -c 0; exec timeout -k 10 21600 python -u "
        f"{container_root}/round12a_nonkt_server.py "
        f"--profile {container_root}/nonkt-server-profile.json "
        "--resolved-profile /out/resolved-profile.json "
        "--admission /out/admission.json > /out/server.log 2>&1"
    )
    create = [
        "docker", "create", "--name", name,
        "--network", "none", "--restart", "no",
        "--cpuset-cpus", ",".join(map(str, cpus)),
        "--cpuset-mems", "2,3,4,7", "--cpus", "136",
        "--memory", "640g", "--memory-swap", "640g",
        "--pids-limit", "4096", "--shm-size", "16g", "--cap-add", "SYS_NICE",
    ]
    for device in ("davinci12", "davinci_manager", "devmm_svm", "hisi_hdc"):
        create += ["--device", f"/dev/{device}"]
    for mount in (
        f"{BASE}:/round9:ro",
        "/home/admin/models/Qwen3.8-Flash-Next:/models:ro",
        f"{out}:/out",
        "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    ):
        create += ["-v", mount]
    environments = (
        "ASCEND_RT_VISIBLE_DEVICES=0",
        "OMP_NUM_THREADS=2",
        "OPENBLAS_NUM_THREADS=2",
        "MKL_NUM_THREADS=2",
        "ROUND9_MEMORY_OBSERVER=1",
        "ROUND12A_NONKT_MODE=1",
        f"ROUND9_CANDIDATE_PROFILE={container_root}/rebound/candidate-profile.json",
        f"PYTHONPATH={container_root}/overlay-python",
        f"ROUND9_FRAMEWORK_PROFILE={container_root}/nonkt-framework-profile.json",
        f"ROUND9_FRAMEWORK_PROFILE_SHA256={sha256(framework)}",
        f"ROUND9_CONTEXT_REFERENCE_PROFILE={container_root}/context-reference-profile.json",
        f"ROUND9_CONTEXT_REFERENCE_SHA256={sha256(context)}",
        f"ROUND12A_NONKT_IDENTITY={container_root}/ROUND12A_NONKT_IDENTITY.json",
        f"ROUND12A_NONKT_IDENTITY_SHA256={sha256(identity)}",
        f"ROUND12A_SNAPSHOT_MANIFEST_SHA256={sha256(manifest)}",
        f"ROUND9_QSA_SOURCE_BINDING={container_root}/ROUND12_QSA_SOURCE_BINDING.json",
        f"ROUND9_QSA_SOURCE_BINDING_SHA256={sha256(root / 'ROUND12_QSA_SOURCE_BINDING.json')}",
        "ROUND9_DYNAMIC_BOUNDARY_ENABLE=0",
        "ROUND9_REQUEST_COST_ENABLE=0",
    )
    for environment in environments:
        create += ["-e", environment]
    create += ["--entrypoint", "bash", IMAGE, "-c", script]
    container_id = command(create).strip()
    inspect = json.loads(command(["docker", "inspect", name]))[0]
    (out / "container-created.json").write_text(json.dumps(inspect, indent=2) + "\n")
    if (
        inspect["Id"] != container_id
        or inspect["HostConfig"]["Memory"] != 640 * 1024**3
        or inspect["HostConfig"]["MemorySwap"] != 640 * 1024**3
        or inspect["HostConfig"]["NanoCpus"] != 136_000_000_000
        or inspect["HostConfig"]["CpusetMems"] != "2,3,4,7"
    ):
        raise RuntimeError("non-KT container resource identity mismatch")

    guard_log = (out / "guard.log").open("x")
    guard = subprocess.Popen(
        [
            "python3", str(root / "source/tools/monitor_owned_server.py"),
            "--container", name,
            "--resource-mode", "shared",
            "--output", str(out / "monitor.jsonl"),
            "--ready-file", str(out / "guard-ready.json"),
            "--seconds", "21600",
            "--owned-limit-mib", "37376",
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
            raise RuntimeError("non-KT guard exited before readiness")
        if time.monotonic() >= deadline:
            raise TimeoutError("non-KT guard readiness timeout")
        time.sleep(0.2)
    validate_admission(external)
    with (root / "execution-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget_path = root / "execution-budget.json"
        budget = json.loads(budget_path.read_text())
        starts = budget.setdefault("starts", [])
        if budget.get("stage") != contract.get("budget_stage") or starts:
            raise RuntimeError("non-KT model start budget is unavailable")
        start = {
            "mode": "NON_KT",
            "name": name,
            "container_id": container_id,
            "time": time.time(),
            "output": str(out),
        }
        starts.append(start)
        budget["model_starts_used"] = 1
        temporary = budget_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(budget, indent=2) + "\n")
        temporary.replace(budget_path)
        (out / "model-start.json").write_text(json.dumps(start, indent=2) + "\n")
    command(["docker", "start", name])
    print("STARTED_ROUND12A_NONKT", name, container_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
