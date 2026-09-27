#!/usr/bin/env python3
"""Build immutable Round14 real-weight GDN component A/B snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROLES = {
    276: "A", 277: "B", 278: "A", 279: "B", 280: "A", 281: "B",
    282: "A", 283: "B", 284: "A", 285: "B", 286: "A", 287: "B",
}
STAGE = "ROUND14_C_GDN_BACKEND_COMPONENT"
MUTABLE = {"source-sha256.json", "host-controller.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Round14 C GDN {label} precondition mismatch")
    return source.replace(old, new)


def tree(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in MUTABLE
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def contract(version: int) -> dict:
    block = ROLES.get(version)
    if block is None:
        raise ValueError("unsupported Round14 C GDN component version")
    lower = block.lower()
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": STAGE,
        "version": version,
        "block": block,
        "container": f"kt-round14-c-gdn-component-{lower}-v{version}",
        "evidence": f"20260927-round14-c-gdn-component-{lower}-v{version}",
        "parent": "ROUND14_C_PROFILE_V274_S0_COMPUTE_IDENTITY",
        "candidate": "FROZEN_SGLANG_NPU_GDN_BACKEND",
        "candidate_default": "OFF",
        "component_process_limit": 1,
        "automatic_retries": 0,
        "relative_l2_gate": 0.03,
        "full_parent_improvement_gate": 0.15,
        "allocator_peak_limit_mib": 10240,
        "result": "result.json",
        "result_pass_status": "PASS_ROUND14_C_GDN_COMPONENT_PROCESS",
        "controller_pass_status": "PASS_ROUND14_C_GDN_COMPONENT_CONTROLLER",
    }


def patch_qwen(source: str) -> str:
    if "ROUND14_GDN_BACKEND_ENABLE" in source:
        raise RuntimeError("Round14 C GDN flag precondition mismatch")
    flag_anchor = (
        "        self.reference_config = GDNConfig(\n"
        "            num_key_heads=self.num_k_heads,\n"
        "            num_value_heads=self.num_v_heads,\n"
        "            key_head_dim=self.head_k_dim,\n"
        "            value_head_dim=self.head_v_dim,\n"
        "            conv_kernel_size=self.conv_kernel_size,\n"
        "            eps=self.layer_norm_epsilon,\n"
        "        )\n"
    )
    flag = flag_anchor + (
        "        self.round14_gdn_backend_enabled = (\n"
        "            os.environ.get(\"ROUND14_GDN_BACKEND_ENABLE\", \"0\") == \"1\"\n"
        "        )\n"
    )
    source = replace_once(source, flag_anchor, flag, "GDN flag")
    import_anchor = (
        "from sglang.srt.layers.round9.gdn import GDNConfig, GDNState, eager_gdn\n"
    )
    import_replacement = (
        "from sglang.srt.layers.round9.gdn import (\n"
        "    GDNConfig, GDNState, eager_gdn, sigmoid_gated_rms_norm,\n"
        ")\n"
    )
    source = replace_once(
        source, import_anchor, import_replacement, "sigmoid gate import"
    )
    forward_anchor = (
        "    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):\n"
        "        _guard_profile(forward_batch)\n"
    )
    branch = forward_anchor + (
        "        if self.round14_gdn_backend_enabled:\n"
        "            mixed_qkv, _ = self.in_proj_qkv(hidden_states)\n"
        "            z, _ = self.in_proj_z(hidden_states)\n"
        "            a, _ = self.in_proj_a(hidden_states)\n"
        "            b, _ = self.in_proj_b(hidden_states)\n"
        "            z = z.reshape(-1, self.num_v_heads, self.head_v_dim)\n"
        "            core = self.attn(\n"
        "                forward_batch=forward_batch, mixed_qkv=mixed_qkv,\n"
        "                a=a.contiguous(), b=b.contiguous(),\n"
        "            ).reshape(-1, self.num_v_heads, self.head_v_dim)\n"
        "            normalized = sigmoid_gated_rms_norm(\n"
        "                core, z, self.norm.weight, eps=self.layer_norm_epsilon\n"
        "            )\n"
        "            output, _ = self.out_proj(normalized.flatten(-2))\n"
        "            return output\n"
    )
    source = replace_once(source, forward_anchor, branch, "GDN backend branch")
    compile(source, "qwen4_exp.py", "exec")
    return source


def patch_monitor(source: str) -> str:
    old = (
        "r'|kt-round14-a-k-[ab]-v(?:235|238)|kt-round14-c-profile-[ab]-v(?:249|250|274|275))', name) is not None"
    )
    new = (
        "r'|kt-round14-a-k-[ab]-v(?:235|238)|kt-round14-c-profile-[ab]-v(?:249|250|274|275)'\n"
        "        r'|kt-round14-c-gdn-component-[ab]-v(?:276|277|278|279|280|281|282|283|284|285|286|287))', name) is not None"
    )
    source = replace_once(source, old, new, "monitor identity")
    compile(source, "monitor_owned_server.py", "exec")
    return source


def refresh_overlay_receipt(output: Path, qwen: Path) -> None:
    tools = output / "source/tools"
    import sys

    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        overlay = output / "overlay-python"
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round14_c_gdn_candidate"] = {
            "status": "INSTALLED_DEFAULT_OFF",
            "qwen_sha256": sha256(qwen),
            "flag": "ROUND14_GDN_BACKEND_ENABLE",
            "default": "0",
            "implementation": "NPU_GDN_CORE_PLUS_FROZEN_QWEN4_SIGMOID_GATE",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def build(parent: Path, output: Path, version: int, probe: Path, controller: Path) -> dict:
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    probe = Path(probe).resolve(strict=True)
    controller = Path(controller).resolve(strict=True)
    value = contract(version)
    if output.exists():
        raise FileExistsError(output)
    parent_contract = json.loads(
        (parent / "ROUND14_C_PROFILE_CONTRACT.json").read_text()
    )
    if parent_contract.get("version") != 274 or parent_contract.get("block") != "A":
        raise RuntimeError("Round14 C GDN requires frozen fresh-S0 v274 parent")
    shutil.copytree(
        parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in (
        "round14-source-sha256.json",
        "source-sha256.json",
        "host-controller.json",
        "execution-budget.lock",
    ):
        (output / name).unlink(missing_ok=True)
    qwen = output / "overlay-python/sglang/srt/models/qwen4_exp.py"
    qwen.write_text(patch_qwen(qwen.read_text()))
    refresh_overlay_receipt(output, qwen)
    shutil.copy2(probe, output / "round14_c_gdn_component_probe.py")
    shutil.copy2(controller, output / "round14_c_gdn_component_controller.py")
    monitor_source = parent / "source/tools/monitor_owned_server.py"
    monitor = output / "monitor_owned_server.py"
    monitor.write_text(patch_monitor(monitor_source.read_text()))
    shutil.copy2(parent / "source/tools/numa_policy.py", output / "numa_policy.py")
    (output / "ROUND14_C_GDN_COMPONENT_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    for path in (
        qwen,
        monitor,
        output / "numa_policy.py",
        output / probe.name,
        output / controller.name,
    ):
        compile(path.read_text(), str(path), "exec")
    manifest = tree(output)
    (output / "source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if tree(output) != manifest:
        raise RuntimeError("Round14 C GDN component manifest is inconsistent")
    return {
        "schema": 1,
        "status": "PASS_ROUND14_C_GDN_COMPONENT_SNAPSHOT_BUILD",
        "version": version,
        "block": value["block"],
        "output": str(output.resolve()),
        "manifest_sha256": sha256(output / "source-sha256.json"),
        "qwen_sha256": sha256(qwen),
        "immutable_files": len(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.parent, args.output, args.version, args.probe, args.controller)
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["version"], report["block"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
