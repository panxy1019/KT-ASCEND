"""Fresh guarded launch for the standalone Round9 layerwise BF16 device gate.

This launcher admits only a synthetic layer-0 component probe.  It never stops
unrelated workloads and does not claim scheduler or full-model qualification.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tree(root, predicate=lambda path: True):
    root = pathlib.Path(root).resolve(strict=True)
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and predicate(path)
    }


def write_manifest(path, values):
    pathlib.Path(path).write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    return digest(path)


def command(arguments):
    return subprocess.check_output(arguments, text=True, timeout=30)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--version", type=int, required=True)
parser.add_argument("--resource-mode", choices=("shared", "exclusive"), default="exclusive")
parser.add_argument("--workspace-budget-mib", type=int, default=1024)
args = parser.parse_args()
if args.version <= 0:
    raise ValueError("version must be positive")
if not 1 <= args.workspace_budget_mib <= 1536:
    raise ValueError("workspace budget must be in [1,1536] MiB")

root = pathlib.Path("/mnt/data1/kt-round9")
source = pathlib.Path(__file__).resolve().parents[1]
installed_runtime = root / "candidate-runtime-python"
sglang = root / "candidate-sglang-v8" / "python"
for required in (root, source, installed_runtime, sglang):
    required.resolve(strict=True)

name = f"kt-round9-layerwise-v{args.version}"
output = root / "evidence" / f"20260921-layerwise-v{args.version}"
output.mkdir(exist_ok=False)

if "110.123.0.3" not in command(["hostname", "-I"]).split():
    raise RuntimeError("layerwise probe must run on the admitted host")
nodes = (2, 3, 6, 7)
cpus = []
physical_cores = set()
topology = []
for node in nodes:
    for cpu in range(node * 80, (node + 1) * 80, 2):
        cpu_path = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}")
        if not cpu_path.joinpath(f"node{node}").exists():
            raise RuntimeError("selected CPU does not belong to admitted NUMA node")
        core = (cpu_path / "topology/core_id").read_text().strip()
        socket = (cpu_path / "topology/physical_package_id").read_text().strip()
        identity = (socket, core)
        if identity in physical_cores:
            raise RuntimeError("duplicate physical CPU core")
        physical_cores.add(identity)
        cpus.append(cpu)
        topology.append({"cpu": cpu, "core": f"{socket}:{core}", "numa": node})
if len(physical_cores) != 160 or not set(cpus) <= os.sched_getaffinity(0):
    raise RuntimeError("required physical CPU set is unavailable")


def cpu_stats():
    result = {}
    for line in pathlib.Path("/proc/stat").read_text().splitlines():
        values = line.split()
        if values[0].startswith("cpu") and values[0] != "cpu":
            result[int(values[0][3:])] = list(map(int, values[1:]))
    return result


before = cpu_stats()
time.sleep(1)
after = cpu_stats()
idle = {
    cpu: (after[cpu][3] - before[cpu][3])
    / max(1, sum(after[cpu][:8]) - sum(before[cpu][:8]))
    for cpu in cpus
}
if sum(idle.values()) / len(idle) <= 0.8:
    raise RuntimeError("selected CPUs are busy; retry a fresh admission later")
meminfo = dict(
    line.split(":", 1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines()
)
if int(meminfo["MemAvailable"].split()[0]) <= 128 * 1024**2:
    raise RuntimeError("host has less than 128GiB available")
node_meminfo = {
    node: pathlib.Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
    for node in nodes
}
for raw in node_meminfo.values():
    free_kib = next(
        int(line.split()[-2]) for line in raw.splitlines() if "MemFree:" in line
    )
    if free_kib <= 8 * 1024**2:
        raise RuntimeError("probe requires 8GiB genuinely free on each selected NUMA node")

npu_smi = command(["npu-smi", "info"])
if re.search(r"\|\s*7\s+1\s*\|\s*\d+\s*\|", npu_smi):
    raise RuntimeError("refresh training PID admission before sharing an active process")
board_row = next(line for line in npu_smi.splitlines() if "0000:83:00.0" in line)
used_mib, capacity_mib = map(int, re.findall(r"(\d+)\s*/\s*(\d+)", board_row)[-1])
training_allowance_mib = 15360 if args.resource_mode == "shared" else 0
owned_hbm_budget_mib = 4096
board_reserve_mib = 8192
if (capacity_mib != 65536
        or used_mib + training_allowance_mib + owned_hbm_budget_mib
        + board_reserve_mib > capacity_mib):
    raise RuntimeError("physical15 HBM headroom does not admit the bounded probe")

frozen = output / "source"
for subdirectory in ("tests", "tools", "runtime"):
    shutil.copytree(
        source / subdirectory, frozen / subdirectory,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
frozen_runtime_manifest = output / "frozen-runtime-sha256.json"
installed_runtime_manifest = output / "installed-runtime-sha256.json"
sglang_manifest = output / "sglang-source-sha256.json"
frozen_runtime_sha = write_manifest(frozen_runtime_manifest, tree(frozen / "runtime"))
installed_runtime_sha = write_manifest(installed_runtime_manifest, tree(installed_runtime))
sglang_sha = write_manifest(
    sglang_manifest, tree(sglang, lambda path: path.suffix == ".py")
)
identity_manifest_sha256 = {
    "frozen_runtime": frozen_runtime_sha,
    "installed_runtime": installed_runtime_sha,
    "sglang": sglang_sha,
}
probe_source_sha256 = digest(frozen / "tests" / "layerwise_device_probe.py")
launcher_source_sha256 = digest(frozen / "tools" / "launch_layerwise_probe.py")

receipt = {
    "status": "ADMITTED_LAYERWISE_BF16_TRANSPORT_PENDING",
    "time": time.time(),
    "host": "110.123.0.3",
    "physical": 15,
    "scope": "LAYER0_SYNTHETIC_REAL_NPU_NOT_FULL_MODEL_OR_SCHEDULER",
    "capacity": 32,
    "max_group_rows": 32,
    "geometry": {"hidden": 2560, "intermediate": 640, "experts": 512, "layers": 48},
    "workspace_budget_mib": args.workspace_budget_mib,
    "resource_mode": args.resource_mode,
    "training_allowance_mib": training_allowance_mib,
    "owned_hbm_budget_mib": owned_hbm_budget_mib,
    "board_reserve_mib": board_reserve_mib,
    "host_memory_limit_gib": 64,
    "numa_nodes": list(nodes),
    "logical_cpus": cpus,
    "topology": topology,
    "idle_fraction": idle,
    "host_meminfo": meminfo,
    "node_meminfo": node_meminfo,
    "npu_smi": npu_smi,
    "identity_manifest_sha256": identity_manifest_sha256,
    "probe_source_sha256": probe_source_sha256,
    "launcher_source_sha256": launcher_source_sha256,
}
(output / "admission.json").write_text(json.dumps(receipt, indent=2) + "\n")

probe_arguments = (
    "--model /models/{revision} "
    "--identity-receipt /models/{revision}.round9-support/"
    "destination-verification/verification-1789886190714568317.json "
    "--output /out/result.json --admission /out/admission.json "
    "--frozen-runtime-manifest /out/frozen-runtime-sha256.json "
    "--installed-runtime-manifest /out/installed-runtime-sha256.json "
    "--sglang-manifest /out/sglang-source-sha256.json "
    "--frozen-runtime-manifest-sha256 {frozen_sha} "
    "--installed-runtime-manifest-sha256 {installed_sha} "
    "--sglang-manifest-sha256 {sglang_sha} "
    "--workspace-budget-mib {workspace}"
).format(
    revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
    frozen_sha=frozen_runtime_sha,
    installed_sha=installed_runtime_sha,
    sglang_sha=sglang_sha,
    workspace=args.workspace_budget_mib,
)
script = (
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
    "ulimit -c 0; "
    "exec timeout -k 10 1800 python -u "
    "/out/source/tests/layerwise_device_probe.py " + probe_arguments
    + " > /out/probe.log 2>&1"
)
docker_args = [
    "docker", "create", "--name", name, "--network", "none",
    "--cpuset-cpus", ",".join(map(str, cpus)),
    "--cpuset-mems", ",".join(map(str, nodes)),
    "--cpus", "136", "--memory", "64g", "--memory-swap", "64g",
    "--pids-limit", "1024", "--shm-size", "4g", "--cap-add", "SYS_NICE",
]
for device in ("davinci15", "davinci_manager", "devmm_svm", "hisi_hdc"):
    docker_args.extend(("--device", "/dev/" + device))
for mount in (
    f"{root}:/round9:ro",
    "/home/admin/models/Qwen3.8-Flash-Next:/models:ro",
    f"{output}:/out",
    "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
):
    docker_args.extend(("-v", mount))
for environment in (
    "PYTHONPATH=/out/source:/round9/candidate-sglang-v8/python:"
    "/round9/candidate-runtime-python",
    "ASCEND_RT_VISIBLE_DEVICES=0",
    "OMP_NUM_THREADS=8",
    "OPENBLAS_NUM_THREADS=8",
    "MKL_NUM_THREADS=8",
):
    docker_args.extend(("-e", environment))
docker_args.extend(("--entrypoint", "bash", "601e76d17e9f", "-c", script))
command(docker_args)
inspect = json.loads(command(["docker", "inspect", name]))[0]
(output / "container-created.json").write_text(json.dumps(inspect, indent=2) + "\n")
host_config = inspect["HostConfig"]
if (host_config["NanoCpus"] != 136 * 10**9
        or host_config["Memory"] != 64 * 1024**3
        or host_config["MemorySwap"] != host_config["Memory"]):
    raise RuntimeError("container resource limits differ from admission")

with (output / "guard.log").open("w") as guard_log:
    guard = subprocess.Popen(
        [
            "python3", str(frozen / "tools" / "monitor_owned_server.py"),
            "--container", name,
            "--output", str(output / "monitor.jsonl"),
            "--ready-file", str(output / "guard-ready.json"),
            "--resource-mode", args.resource_mode,
            "--seconds", "1850",
            "--owned-limit-mib", str(owned_hbm_budget_mib),
        ],
        stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
try:
    deadline = time.monotonic() + 40
    while not (output / "guard-ready.json").exists():
        if guard.poll() is not None:
            raise RuntimeError("resource guard failed")
        if time.monotonic() > deadline:
            raise RuntimeError("resource guard readiness timeout")
        time.sleep(0.2)
    if time.time() - receipt["time"] >= 60:
        raise RuntimeError("resource admission expired")
    if tree(frozen / "runtime") != json.loads(frozen_runtime_manifest.read_text()):
        raise RuntimeError("frozen runtime changed before container start")
    command(["docker", "start", name])
    print("STARTED", name, flush=True)
except BaseException:
    subprocess.run(
        ["docker", "stop", "--time", "10", name],
        check=False, timeout=30,
    )
    raise
