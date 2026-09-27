#!/usr/bin/env python3
"""Build one immutable Round14 A KT or native-TP8 comparison snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


STAGE = "ROUND14_A_KT_NATIVE_TP8_SYSTEM_COMPARISON"
ROLES = {
    235: ("KT", "A", "kt-round14-a-k-a-v235"),
    236: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v236"),
    237: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v237"),
    238: ("KT", "B", "kt-round14-a-k-b-v238"),
    239: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v239"),
    240: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v240"),
    241: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v241"),
    242: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v242"),
    247: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v247"),
    248: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v248"),
    251: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v251"),
    252: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v252"),
}
MUTABLE = {
    "round14-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def directory_digest(root: Path) -> str:
    document = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def patch_exact(value: str, old: str, new: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(
            f"snapshot source gate expected exactly once, found {count}: {old!r}"
        )
    return value.replace(old, new)


def contract(version: int) -> dict:
    if version not in ROLES:
        raise ValueError(f"unsupported Round14 A version {version}")
    backend, block, container = ROLES[version]
    short_backend = "k" if backend == "KT" else "n"
    evidence = f"20260927-round14-a-{short_backend}-{block.lower()}-v{version}"
    value = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": STAGE,
        "version": version,
        "backend": backend,
        "block": block,
        "container": container,
        "evidence": evidence,
        "generation_limit": 24,
        "abort_limit": 0,
        "automatic_retries": 0,
        "profiler": "OFF",
        "raw_result": "round14-a-raw.json",
        "final_result": "round14-a-final.json",
        "driver_pass_status": "PASS_ROUND14_A_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND14_A_MODEL_PROCESS",
        "workload_order": ["W2", "W0", "W1", "W4", "W3"],
        "comparison_semantics": "SYSTEM_DEPLOYMENT_COMPARISON_NOT_PURE_KT_CAUSAL",
        "driver_command": [
            "{DRIVER_ROOT}/round14_a_driver.py",
            "--version", str(version),
            "--backend", backend,
            "--block", block,
            "--out", "/out",
            "--fixture10", "{DRIVER_ROOT}/round10-t1-fixture.json",
            "--fixture11", "{DRIVER_ROOT}/round11-t1-fixture.json",
        ],
    }
    if backend == "KT":
        value.update({
            "parent": "ROUND13_FINAL_S0_UNIFORM_Q32",
            "physical_devices": [12],
            "precision": "MXFP4_CPU_PLUS_BF16_NZ29_Q32",
            "qsa_source": "VECTORIZED_PREFILL_DIRECT_BF16_ROW_DECODE",
            "server_profile": "H513_EXPERIMENTAL_FOR_W3",
            "default_profile_outside_experiment": "H129",
        })
    else:
        value.update({
            "mode": "NATIVE_TP8",
            "parent": "ROUND12A_NATIVE_TP8_OLD_QSA",
            "physical_devices": list(range(8)),
            "tp_size": 8,
            "precision": "BF16",
            "qsa_source": "QUALIFIED_ORIGINAL",
            "server_profile": "NATIVE_TP8_NO_HYBRID_THRESHOLD",
        })
    return value


def render_kt_control_plane(parent: Path, value: dict) -> dict[str, str]:
    parent = Path(parent)
    version = value["version"]
    launcher = (parent / "source/tools/launch_hybrid.py").read_text()
    launcher = patch_exact(
        launcher,
        "if a.version != 228 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
    )
    launcher = patch_exact(
        launcher,
        "name='kt-round13-f-final-a-v228';out=root/'evidence'/'20260927-round13-f-final-a-v228'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'",
    )
    launcher = patch_exact(
        launcher,
        "if budget.get('stage')!='ROUND13_F_FINAL':raise RuntimeError('wrong Round13 F budget')",
        f"if budget.get('stage')!='{STAGE}':raise RuntimeError('wrong Round14 A budget')",
    )

    monitor = (parent / "source/tools/monitor_owned_server.py").read_text()
    monitor = patch_exact(
        monitor,
        "        r'|c-(?:d0|d1)-(?:a|b)|f-(?:s0|final)-(?:a|b))-v[0-9]+)', name) is not None",
        "        r'|c-(?:d0|d1)-(?:a|b)|f-(?:s0|final)-(?:a|b))-v[0-9]+'\n"
        "        r'|kt-round14-a-k-[ab]-v(?:235|238))', name) is not None",
    )
    monitor = monitor.replace(
        "Round9/Round10/Round11/Round12/Round12a/Round13 container",
        "Round9/Round10/Round11/Round12/Round12a/Round13/Round14 container",
    )

    controller = (parent / "round13_model_controller.py").read_text()
    for old, new in (
        ("immutable Round13 KT", "immutable Round14 A KT"),
        ("round13-source-sha256.json", "round14-source-sha256.json"),
        ("ROUND13_MODEL_CONTRACT.json", "ROUND14_A_CONTRACT.json"),
        ("Round13 snapshot manifest mismatch", "Round14 A KT snapshot manifest mismatch"),
        ("round13_model_supervisor.py", "round14_a_supervisor.py"),
        ("round13-model-supervisor.json", "round14-a-supervisor.json"),
    ):
        controller = controller.replace(old, new)

    supervisor = (parent / "round13_model_supervisor.py").read_text()
    for old, new in (
        ("immutable Round13 KT", "immutable Round14 A KT"),
        ("fixed Round13 model identity required", "fixed Round14 A KT model identity required"),
        ("round13-model-supervisor.json", "round14-a-supervisor.json"),
        ("Round13 completion", "Round14 A completion"),
        ("Round13 log ceiling", "Round14 A log ceiling"),
        ("Round13 ledger", "Round14 A ledger"),
        ("round13-model-driver.log", "round14-a-driver.log"),
        ("Round13 driver", "Round14 A driver"),
        ("Round13 successful result", "Round14 A successful result"),
        ("bounded Round13 supervisor", "bounded Round14 A supervisor"),
        ("physical12 is not empty after Round13 cleanup", "physical12 is not empty after Round14 A cleanup"),
    ):
        supervisor = supervisor.replace(old, new)

    return {
        "source/tools/launch_hybrid.py": launcher,
        "source/tools/monitor_owned_server.py": monitor,
        "round14_a_controller.py": controller,
        "round14_a_supervisor.py": supervisor,
    }


def render_native_control_plane(parent: Path, value: dict) -> dict[str, str]:
    parent = Path(parent)
    controller = (parent / "native_tp8_controller.py").read_text()
    for old, new in (
        ("immutable native TP8 snapshot", "immutable Round14 A native TP8 snapshot"),
        ("round12-source-sha256.json", "round14-source-sha256.json"),
        ("ROUND12A_NATIVE_TP8_CONTRACT.json", "ROUND14_A_CONTRACT.json"),
        ("native TP8 snapshot manifest mismatch", "Round14 A native TP8 snapshot manifest mismatch"),
        ("native_tp8_supervisor.py", "round14_a_supervisor.py"),
        ("native-tp8-supervisor.json", "round14-a-supervisor.json"),
    ):
        controller = controller.replace(old, new)

    supervisor = (parent / "native_tp8_supervisor.py").read_text()
    for old, new in (
        ("native TP8 model process", "Round14 A native TP8 model process"),
        ("fixed native TP8 identity required", "fixed Round14 A native TP8 identity required"),
        ("native-tp8-supervisor.json", "round14-a-supervisor.json"),
        ("native-tp8-driver.log", "round14-a-driver.log"),
        ("native TP8 request ledger", "Round14 A request ledger"),
        ("native TP8 raw result failed", "Round14 A native raw result failed"),
        ('**raw, "status": "PASS_ROUND12A_NATIVE_TP8_MODEL_PROCESS",',
         '**raw, "status": contract["process_pass_status"],'),
        ('(args.out / "native-tp8-final.json").write_text',
         '(args.out / contract["final_result"]).write_text'),
        ('final_result="native-tp8-final.json"',
         'final_result=contract["final_result"]'),
    ):
        supervisor = supervisor.replace(old, new)
    supervisor = patch_exact(
        supervisor,
        'if raw.get("status") != contract["driver_pass_status"] or raw.get("generation_attempts") != 4:\n'
        '            raise RuntimeError("Round14 A native raw result failed")',
        'if (raw.get("status") != contract["driver_pass_status"]\n'
        '                or raw.get("generation_attempts") != contract["generation_limit"]\n'
        '                or raw.get("abort_controls", 0) != contract["abort_limit"]\n'
        '                or raw.get("automatic_retries") != 0\n'
        '                or raw.get("profiler") != contract["profiler"]):\n'
        '            raise RuntimeError("Round14 A native raw result failed")',
    )

    launcher = (parent / "native_tp8_launcher.py").read_text()
    launcher = launcher.replace(
        "ROUND12A_NATIVE_TP8_CONTRACT.json", "ROUND14_A_CONTRACT.json"
    )
    launcher = patch_exact(
        launcher,
        "version not in (180, 181, 182, 183)",
        f"version != {value['version']}",
    )
    launcher = patch_exact(
        launcher,
        "expected = f\"kt-round12a-native-tp8-{contract['block'].lower()}-v{version}\"",
        "expected = f\"kt-round14-a-n-{contract['block'].lower()}-v{version}\"",
    )
    launcher = patch_exact(
        launcher,
        'item.startswith(("kt-round10", "kt-round11", "kt-round12"))',
        'item.startswith(("kt-round10", "kt-round11", "kt-round12", "kt-round13", "kt-round14"))',
    )
    launcher = launcher.replace(
        "round12-source-sha256.json", "round14-source-sha256.json"
    )
    launcher = launcher.replace(
        "STARTED_ROUND12A_NATIVE_TP8", "STARTED_ROUND14_A_NATIVE_TP8"
    )

    monitor = (parent / "native_tp8_monitor.py").read_text()
    monitor = patch_exact(
        monitor,
        'r"kt-round12a-native-tp8-[ab]-v(?:180|181|182|183)"',
        f'r"{value["container"]}"',
    )

    admission = (parent / "native_tp8_admission.py").read_text()
    admission = patch_exact(
        admission,
        'if name.startswith("kt-round12") or name.startswith("kt-round11") or name.startswith("kt-round10"):',
        'if (name.startswith("kt-round10") or name.startswith("kt-round11")\n'
        '                or name.startswith("kt-round12") or name.startswith("kt-round13")\n'
        '                or name.startswith("kt-round14")):',
    )

    image_preflight = (parent / "native_tp8_image_preflight.py").read_text()
    image_preflight = patch_exact(
        image_preflight,
        "import argparse\n",
        "import argparse\nimport ast\n",
    )
    image_preflight = image_preflight.replace(
        "round12-source-sha256.json", "round14-source-sha256.json"
    )
    image_preflight = patch_exact(
        image_preflight,
        '    original = getattr(npu, "is_available", None)\n'
        '    if not callable(original):',
        '    original = getattr(npu, "is_available", None)\n'
        '    original_mem_get_info = getattr(npu, "mem_get_info", None)\n'
        '    original_get_device_name = getattr(npu, "get_device_name", None)\n'
        '    if (not callable(original) or not callable(original_mem_get_info)\n'
        '            or not callable(original_get_device_name)):',
    )
    image_preflight = patch_exact(
        image_preflight,
        '        "scope": "IMPORT_AND_SERVERARGS_ONLY",\n'
        '    }\n'
        '    npu.is_available = lambda: True',
        '        "scope": "IMPORT_AND_SERVERARGS_ONLY",\n'
        '        "memory_capacity_for_serverargs_mib": 65536,\n'
        '        "mem_get_info_restored": False,\n'
        '        "get_device_name_restored": False,\n'
        '    }\n'
        '    npu.is_available = lambda: True\n'
        '    npu.mem_get_info = lambda: (65536 * 1024**2, 65536 * 1024**2)\n'
        '    npu.get_device_name = lambda _device=0: "Ascend910B"',
    )
    image_preflight = patch_exact(
        image_preflight,
        '        npu.is_available = original\n'
        '        receipt["restored"] = True',
        '        npu.is_available = original\n'
        '        npu.mem_get_info = original_mem_get_info\n'
        '        npu.get_device_name = original_get_device_name\n'
        '        receipt["restored"] = True\n'
        '        receipt["mem_get_info_restored"] = True\n'
        '        receipt["get_device_name_restored"] = True',
    )
    image_preflight = patch_exact(
        image_preflight,
        'def run(snapshot: Path) -> dict:\n'
        '    snapshot = Path(snapshot).resolve(strict=True)\n',
        'def validate_driver_dependencies(snapshot: Path) -> list[str]:\n'
        '    snapshot = Path(snapshot).resolve(strict=True)\n'
        '    contract = json.loads((snapshot / "ROUND14_A_CONTRACT.json").read_text())\n'
        '    dependencies = []\n'
        '    for value in contract.get("driver_command", []):\n'
        '        if isinstance(value, str) and value.startswith("{DRIVER_ROOT}/"):\n'
        '            relative = value.removeprefix("{DRIVER_ROOT}/")\n'
        '            if not (snapshot / relative).is_file():\n'
        '                raise RuntimeError(f"frozen driver dependency is missing: {relative}")\n'
        '            dependencies.append(relative)\n'
        '    if not dependencies:\n'
        '        raise RuntimeError("frozen driver dependency list is empty")\n'
        '    return dependencies\n\n\n'
        'def run(snapshot: Path) -> dict:\n'
        '    snapshot = Path(snapshot).resolve(strict=True)\n'
        '    driver_dependencies = validate_driver_dependencies(snapshot)\n',
    )
    image_preflight = patch_exact(
        image_preflight,
        'def run(snapshot: Path) -> dict:\n'
        '    snapshot = Path(snapshot).resolve(strict=True)\n'
        '    driver_dependencies = validate_driver_dependencies(snapshot)\n',
        'def validate_local_import_dependencies(snapshot: Path, driver: Path) -> list[str]:\n'
        '    snapshot = Path(snapshot).resolve(strict=True)\n'
        '    driver = Path(driver).resolve(strict=True)\n'
        '    modules = set()\n'
        '    for node in ast.walk(ast.parse(driver.read_text())):\n'
        '        if isinstance(node, ast.ImportFrom) and node.module:\n'
        '            root = node.module.split(".", 1)[0]\n'
        '            if root.startswith("round"):\n'
        '                modules.add(root)\n'
        '        elif isinstance(node, ast.Import):\n'
        '            for alias in node.names:\n'
        '                root = alias.name.split(".", 1)[0]\n'
        '                if root.startswith("round"):\n'
        '                    modules.add(root)\n'
        '    dependencies = []\n'
        '    for module in sorted(modules):\n'
        '        relative = f"{module}.py"\n'
        '        if not (snapshot / relative).is_file():\n'
        '            raise RuntimeError(f"frozen lazy driver dependency is missing: {relative}")\n'
        '        dependencies.append(relative)\n'
        '    if not dependencies:\n'
        '        raise RuntimeError("frozen lazy driver dependency list is empty")\n'
        '    return dependencies\n\n\n'
        'def run(snapshot: Path) -> dict:\n'
        '    snapshot = Path(snapshot).resolve(strict=True)\n'
        '    driver_dependencies = validate_driver_dependencies(snapshot)\n'
        '    lazy_driver_dependencies = validate_local_import_dependencies(\n'
        '        snapshot, snapshot / "round14_a_driver.py"\n'
        '    )\n',
    )
    image_preflight = patch_exact(
        image_preflight,
        '            "qwen_import_disposition": "QWEN_IMPORT_DEFERRED_TO_DEVICE_LOAD",\n'
        '            **observed,\n',
        '            "qwen_import_disposition": "QWEN_IMPORT_DEFERRED_TO_DEVICE_LOAD",\n'
        '            "driver_dependencies": driver_dependencies,\n'
        '            "lazy_driver_dependencies": lazy_driver_dependencies,\n'
        '            **observed,\n',
    )

    return {
        "round14_a_controller.py": controller,
        "round14_a_supervisor.py": supervisor,
        "native_tp8_launcher.py": launcher,
        "native_tp8_monitor.py": monitor,
        "native_tp8_admission.py": admission,
        "native_tp8_image_preflight.py": image_preflight,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build(
    parent: Path,
    output: Path,
    version: int,
    driver: Path,
    *,
    fixture11: Path | None = None,
    protocol10: Path | None = None,
) -> dict:
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    driver = Path(driver).resolve(strict=True)
    value = contract(version)
    backend = value["backend"]
    if backend == "NATIVE_TP8":
        if fixture11 is None:
            raise ValueError("native Round14 A snapshot requires the frozen Round11 fixture")
        if protocol10 is None:
            raise ValueError("native Round14 A snapshot requires the frozen Round10 protocol")
        fixture11 = Path(fixture11).resolve(strict=True)
        protocol10 = Path(protocol10).resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    if backend == "KT" and not (parent / "ROUND13_MODEL_CONTRACT.json").exists():
        raise ValueError("KT parent is not the frozen Round13 final snapshot")
    if backend == "NATIVE_TP8" and not (parent / "ROUND12A_NATIVE_TP8_CONTRACT.json").exists():
        raise ValueError("native parent is not the frozen Round12a TP8 snapshot")

    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in ("host-controller.json", "execution-budget.lock", "round14-source-sha256.json"):
        (output / name).unlink(missing_ok=True)

    rendered = (
        render_kt_control_plane(parent, value)
        if backend == "KT"
        else render_native_control_plane(parent, value)
    )
    for relative, content in rendered.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    shutil.copy2(driver, output / "round14_a_driver.py")
    if backend == "NATIVE_TP8":
        shutil.copy2(fixture11, output / "round11-t1-fixture.json")
        shutil.copy2(protocol10, output / "round10_t1_protocol.py")
    _write_json(output / "ROUND14_A_CONTRACT.json", value)
    for argument in value["driver_command"]:
        if argument.startswith("{DRIVER_ROOT}/"):
            dependency = output / argument.removeprefix("{DRIVER_ROOT}/")
            if not dependency.is_file():
                raise RuntimeError(
                    f"Round14 A driver dependency is absent from snapshot: {dependency.name}"
                )
    budget = {
        "schema": 1,
        "stage": STAGE,
        "version": version,
        "backend": backend,
        "block": value["block"],
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": 24,
        "abort_limit": 0,
        "automatic_retries_limit": 0,
    }
    _write_json(output / "execution-budget.json", budget)

    if backend == "KT":
        parent_contract = json.loads((parent / "ROUND13_MODEL_CONTRACT.json").read_text())
        delta = {
            "schema": 1,
            "status": "FROZEN",
            "version": version,
            "parent": str(parent),
            "parent_runtime_sha256": parent_contract["runtime_sha256"],
            "parent_qsa_sha256": "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35",
            "model_runtime_change": "NONE",
            "changed_scope": "ROUND14_A_DRIVER_AND_BOUNDED_CONTROL_PLANE_ONLY",
            "source_runtime_parent_digest": directory_digest(parent / "source/runtime"),
            "source_runtime_output_digest": directory_digest(output / "source/runtime"),
        }
    else:
        identity = json.loads((parent / "ROUND12A_NATIVE_TP8_IDENTITY.json").read_text())
        delta = {
            "schema": 1,
            "status": "FROZEN",
            "version": version,
            "parent": str(parent),
            "parent_runtime_sha256": sha256(parent / "ROUND12A_NATIVE_TP8_IDENTITY.json"),
            "parent_qsa_sha256": identity["qsa_sha256"],
            "model_runtime_change": "NONE",
            "changed_scope": "ROUND14_A_DRIVER_AND_BOUNDED_CONTROL_PLANE_ONLY",
            "overlay_parent_digest": directory_digest(parent / "overlay-python"),
            "overlay_output_digest": directory_digest(output / "overlay-python"),
            "round11_fixture_sha256": sha256(output / "round11-t1-fixture.json"),
            "round10_protocol_sha256": sha256(output / "round10_t1_protocol.py"),
        }
    if any(
        delta[key.replace("parent", "output")] != value
        for key, value in delta.items()
        if key.endswith("_parent_digest")
    ):
        raise RuntimeError("model runtime changed while building Round14 A snapshot")
    _write_json(output / "ROUND14_A_SOURCE_DELTA.json", delta)

    manifest = tree(output)
    _write_json(output / "round14-source-sha256.json", manifest)
    if tree(output) != manifest:
        raise RuntimeError("Round14 A snapshot manifest is not self-consistent")
    report = {
        "schema": 1,
        "status": "PASS_ROUND14_A_SNAPSHOT_BUILD",
        "version": version,
        "backend": backend,
        "block": value["block"],
        "output": str(output.resolve()),
        "manifest_sha256": sha256(output / "round14-source-sha256.json"),
        "immutable_files": len(manifest),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True, choices=sorted(ROLES))
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--fixture11", type=Path)
    parser.add_argument("--protocol10", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(
        args.parent, args.output, args.version, args.driver,
        fixture11=args.fixture11,
        protocol10=args.protocol10,
    )
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        _write_json(args.report, report)
    print(report["status"], report["version"], report["backend"], report["block"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
