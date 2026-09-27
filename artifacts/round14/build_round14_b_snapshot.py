#!/usr/bin/env python3
"""Build one immutable Round14 B H-BOUNDED KT model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


STAGE = "ROUND14_B_REAL_DISTRIBUTION_AND_BOUNDED_HYBRID"
ROLES = {
    243: ("H512", "A"),
    244: ("H512", "B"),
    245: ("H768", "A"),
    246: ("H768", "B"),
    253: ("H512", "A"),
    254: ("H512", "B"),
    255: ("H768", "A"),
    256: ("H768", "B"),
    257: ("H512", "A"),
    258: ("H512", "B"),
    259: ("H768", "A"),
    260: ("H768", "B"),
    261: ("H512", "A"),
    262: ("H512", "B"),
    263: ("H768", "A"),
    264: ("H768", "B"),
    265: ("H512", "A"),
    266: ("H512", "B"),
    267: ("H768", "A"),
    268: ("H768", "B"),
    269: ("H512", "A"),
    270: ("H512", "B"),
    273: ("H512", "B"),
}
LIMITS = {
    "H512": {
        "max_input_tokens": 512,
        "max_output_tokens": 256,
        "max_total_tokens": 768,
        "hybrid_threshold_tokens": 513,
    },
    "H768": {
        "max_input_tokens": 768,
        "max_output_tokens": 256,
        "max_total_tokens": 1024,
        "hybrid_threshold_tokens": 769,
    },
}
MUTABLE = {
    "round14-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    payload = json.dumps(
        {
            str(path.relative_to(root)): sha256(path)
            for path in sorted(Path(root).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
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
        raise ValueError(f"unsupported Round14 B version {version}")
    profile, block = ROLES[version]
    limits = dict(LIMITS[profile])
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": STAGE,
        "version": version,
        "profile": profile,
        "block": block,
        "limits": limits,
        "outside_policy": "REJECT_BEFORE_SCHEDULER_DISPATCH",
        "container": f"kt-round14-b-{profile.lower()}-{block.lower()}-v{version}",
        "evidence": (
            f"20260927-round14-b-{profile.lower()}-{block.lower()}-v{version}"
        ),
        "generation_limit": 64,
        "abort_limit": 1,
        "automatic_retries": 0,
        "profiler": "OFF",
        "raw_result": "round14-b-raw.json",
        "final_result": "round14-b-final.json",
        "driver_pass_status": "PASS_ROUND14_B_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND14_B_MODEL_PROCESS",
        "parent": "ROUND13_FINAL_S0_UNIFORM_Q32",
        "physical_devices": [12],
        "precision": "MXFP4_CPU_PLUS_BF16_NZ29_Q32",
        "qsa_source": "VECTORIZED_PREFILL_DIRECT_BF16_ROW_DECODE",
        "driver_command": [
            "{DRIVER_ROOT}/round14_b_driver.py",
            "--version", str(version),
            "--profile", profile,
            "--block", block,
            "--out", "/out",
            "--fixture", "{DRIVER_ROOT}/ROUND14_B_REQUESTS.json",
            "--edge-fixture", "{DRIVER_ROOT}/round11-t1-fixture.json",
        ],
    }


def bounded_profile(value: dict) -> dict:
    return {
        "schema": 1,
        "mode": "H_BOUNDED",
        **value["limits"],
        "outside_policy": "REJECT",
    }


def render_tokenizer_manager(source: str) -> str:
    source = patch_exact(
        source,
        "from sglang.srt.managers.tokenizer_manager_multiitem_mixin import (\n"
        "    TokenizerManagerMultiItemMixin,\n"
        ")\n",
        "from sglang.srt.managers.tokenizer_manager_multiitem_mixin import (\n"
        "    TokenizerManagerMultiItemMixin,\n"
        ")\n"
        "from sglang.srt.round14_bounded_policy import enforce_from_environment\n",
    )
    source = patch_exact(
        source,
        "        self._validate_one_request(obj, input_ids)\n"
        "        return self._create_tokenized_object(\n",
        "        self._validate_one_request(obj, input_ids)\n"
        "        enforce_from_environment(obj, input_ids)\n"
        "        return self._create_tokenized_object(\n",
    )
    source = patch_exact(
        source,
        "            self._validate_one_request(obj[i], input_ids_list[i])\n"
        "            token_type_ids = (\n",
        "            self._validate_one_request(obj[i], input_ids_list[i])\n"
        "            enforce_from_environment(obj[i], input_ids_list[i])\n"
        "            token_type_ids = (\n",
    )
    return source


def refresh_overlay_receipt(output: Path) -> None:
    overlay = output / "overlay-python"
    tools = output / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        manager = overlay / "sglang/srt/managers/tokenizer_manager.py"
        policy = overlay / "sglang/srt/round14_bounded_policy.py"
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round14_b"] = {
            "status": "INSTALLED_PRE_DISPATCH_H_BOUNDED_GATE",
            "tokenizer_manager_sha256": sha256(manager),
            "policy_sha256": sha256(policy),
            "scheduler_dispatch_changed": False,
        }
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def render_control_plane(parent: Path, value: dict) -> dict[str, str]:
    launcher = (parent / "source/tools/launch_hybrid.py").read_text()
    launcher = patch_exact(
        launcher,
        "if a.version != 235 or a.cost_mode != 'OFF':",
        f"if a.version != {value['version']} or a.cost_mode != 'OFF':",
    )
    launcher = patch_exact(
        launcher,
        "name='kt-round14-a-k-a-v235';out=root/'evidence'/'20260927-round14-a-k-a-v235'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'",
    )
    launcher = patch_exact(
        launcher,
        "if budget.get('stage')!='ROUND14_A_KT_NATIVE_TP8_SYSTEM_COMPARISON':raise RuntimeError('wrong Round14 A budget')",
        f"if budget.get('stage')!='{STAGE}':raise RuntimeError('wrong Round14 B budget')",
    )
    launcher = patch_exact(
        launcher,
        "(out/'server-profile.json').write_text(json.dumps(values,indent=2)+'\\n')",
        "(out/'server-profile.json').write_text(json.dumps(values,indent=2)+'\\n')\n"
        "boundary_profile=source.parent/'ROUND14_H_BOUNDED_PROFILE.json'\n"
        "(out/'round14-h-bounded-profile.json').write_bytes(boundary_profile.read_bytes())",
    )
    launcher = patch_exact(
        launcher,
        "if framework:payload['framework_profile']=digest(out/'framework-profile.json')",
        "if framework:payload['framework_profile']=digest(out/'framework-profile.json')\n"
        "payload['round14_h_bounded_profile']=digest(out/'round14-h-bounded-profile.json')",
    )
    launcher = patch_exact(
        launcher,
        "args+=['-e','ROUND9_HYBRID_PROBE=1']",
        "args+=['-e','ROUND9_HYBRID_PROBE=1']\n"
        "for env in ('ROUND14_H_BOUNDED_PROFILE=/out/round14-h-bounded-profile.json',\n"
        "            f'ROUND14_H_BOUNDED_PROFILE_SHA256={digest(out/\"round14-h-bounded-profile.json\")}',\n"
        "            'ROUND14_H_BOUNDED_EVENTS=/out/round14-boundary-events.jsonl'):\n"
        " args+=['-e',env]",
    )

    controller = (parent / "round14_a_controller.py").read_text()
    for old, new in (
        ("Round14 A KT", "Round14 B H-BOUNDED KT"),
        ("ROUND14_A_CONTRACT.json", "ROUND14_B_CONTRACT.json"),
        ("round14_a_supervisor.py", "round14_b_supervisor.py"),
        ("round14-a-supervisor.json", "round14-b-supervisor.json"),
    ):
        controller = controller.replace(old, new)

    supervisor = (parent / "round14_a_supervisor.py").read_text()
    for old, new in (
        ("Round14 A KT", "Round14 B H-BOUNDED KT"),
        ("Round14 A", "Round14 B"),
        ("round14-a-supervisor.json", "round14-b-supervisor.json"),
        ("round14-a-driver.log", "round14-b-driver.log"),
    ):
        supervisor = supervisor.replace(old, new)

    monitor = (parent / "source/tools/monitor_owned_server.py").read_text()
    monitor = patch_exact(
        monitor,
        "        r'|kt-round14-a-k-[ab]-v(?:235|238))', name) is not None",
        "        r'|kt-round14-a-k-[ab]-v(?:235|238)'\n"
        "        r'|kt-round14-b-h(?:512|768)-[ab]-v(?:243|244|245|246|253|254|255|256|257|258|259|260|261|262|263|264|265|266|267|268|269|270|273))', name) is not None",
    )
    return {
        "source/tools/launch_hybrid.py": launcher,
        "source/tools/monitor_owned_server.py": monitor,
        "round14_b_controller.py": controller,
        "round14_b_supervisor.py": supervisor,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build(
    parent: Path,
    output: Path,
    version: int,
    driver: Path,
    policy: Path,
    fixture: Path,
) -> dict:
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    driver = Path(driver).resolve(strict=True)
    policy = Path(policy).resolve(strict=True)
    fixture = Path(fixture).resolve(strict=True)
    value = contract(version)
    if output.exists():
        raise FileExistsError(output)
    parent_contract = json.loads((parent / "ROUND14_A_CONTRACT.json").read_text())
    if parent_contract.get("version") != 235 or parent_contract.get("backend") != "KT":
        raise ValueError("Round14 B parent must be the frozen v235 KT S0 snapshot")

    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in (
        "host-controller.json", "execution-budget.lock",
        "round14-source-sha256.json", "ROUND14_A_CONTRACT.json",
        "ROUND14_A_SOURCE_DELTA.json",
    ):
        (output / name).unlink(missing_ok=True)

    for relative, content in render_control_plane(parent, value).items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)

    manager = output / "overlay-python/sglang/srt/managers/tokenizer_manager.py"
    manager.write_text(render_tokenizer_manager(manager.read_text()))
    hybrid_probe = (
        output / "overlay-python/sglang/srt/layers/round9/hybrid_probe.py"
    )
    hybrid_probe.write_text(patch_exact(
        hybrid_probe.read_text(),
        "if state['rows'] >= 4096:",
        "if state['rows'] >= 8192:",
    ))
    shutil.copy2(policy, output / "overlay-python/sglang/srt/round14_bounded_policy.py")
    shutil.copy2(driver, output / "round14_b_driver.py")
    shutil.copy2(fixture, output / "ROUND14_B_REQUESTS.json")
    refresh_overlay_receipt(output)

    framework_path = output / "framework-profile.json"
    framework = json.loads(framework_path.read_text())
    framework["prefill"]["threshold_tokens"] = value["limits"]["hybrid_threshold_tokens"]
    _write_json(framework_path, framework)
    streaming_path = output / "streaming-profile.json"
    streaming = json.loads(streaming_path.read_text())
    streaming["threshold_tokens"] = value["limits"]["hybrid_threshold_tokens"]
    _write_json(streaming_path, streaming)

    _write_json(output / "ROUND14_H_BOUNDED_PROFILE.json", bounded_profile(value))
    _write_json(output / "ROUND14_B_CONTRACT.json", value)
    _write_json(output / "execution-budget.json", {
        "schema": 1,
        "stage": STAGE,
        "version": version,
        "profile": value["profile"],
        "block": value["block"],
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": value["generation_limit"],
        "abort_limit": value["abort_limit"],
        "automatic_retries_limit": 0,
    })

    delta = {
        "schema": 1,
        "status": "FROZEN",
        "version": version,
        "parent": str(parent),
        "parent_manifest_sha256": sha256(parent / "round14-source-sha256.json"),
        "model_math_change": "NONE",
        "changed_scope": (
            "PRE_DISPATCH_BOUNDARY_GATE_PROFILE_THRESHOLD_ROUND14_B_DRIVER_"
            "AND_HOST_ONLY_HYBRID_PROBE_CAPACITY"
        ),
        "hybrid_probe_max_rows": 8192,
        "policy_source_sha256": sha256(policy),
        "request_fixture_sha256": sha256(fixture),
        "parent_runtime_digest": directory_digest(parent / "source/runtime"),
        "output_runtime_digest": directory_digest(output / "source/runtime"),
    }
    if delta["parent_runtime_digest"] != delta["output_runtime_digest"]:
        raise RuntimeError("model runtime changed while building Round14 B snapshot")
    _write_json(output / "ROUND14_B_SOURCE_DELTA.json", delta)

    manifest = tree(output)
    _write_json(output / "round14-source-sha256.json", manifest)
    if tree(output) != manifest:
        raise RuntimeError("Round14 B snapshot manifest is not self-consistent")
    return {
        "schema": 1,
        "status": "PASS_ROUND14_B_SNAPSHOT_BUILD",
        "version": version,
        "profile": value["profile"],
        "block": value["block"],
        "output": str(output.resolve()),
        "manifest_sha256": sha256(output / "round14-source-sha256.json"),
        "immutable_files": len(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True, choices=sorted(ROLES))
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(
        args.parent, args.output, args.version, args.driver, args.policy, args.fixture
    )
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        _write_json(args.report, report)
    print(report["status"], report["version"], report["profile"], report["block"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
