"""Offline-only builder and gates for two fresh Round16 A1 snapshots.

The builder verifies the frozen v295 inputs before it creates a copy.  It then
patches only that copy.  The preflight and controller helpers read files and
return JSON-compatible data; they never import torch, probe a device, or run a
command.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping, Sequence

from .fresh_a1_profile import (
    BLOCK_IDS,
    RUNTIME_VERSION,
    GENERATION_LIMIT_TOTAL,
    GENERATIONS_PER_BLOCK,
    STAGE,
    block_contract,
    manifest_payload,
    runtime_container_name,
    runtime_evidence_name,
)


KTEP_RELATIVE = "overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py"
ROUND13_RELATIVE = (
    "overlay-python/sglang/srt/layers/round9/round13_attribution.py"
)
SCHEDULER_RELATIVE = "overlay-python/sglang/srt/managers/scheduler.py"
LAUNCHER_RELATIVE = "source/tools/launch_hybrid.py"
FIXTURE_RELATIVE = "round10-t1-fixture.json"
EXECUTION_BUDGET_RELATIVE = "execution-budget.json"
ROUND11_DRIVER_RELATIVE = "round11_t4_driver.py"
ROUND10_PROTOCOL_RELATIVE = "round10_t1_protocol.py"
CLEANUP_RELATIVE = "m3_cleanup.py"
RESOURCE_GATE_RELATIVE = "source/tools/candidate_resources.py"
MONITOR_RELATIVE = "source/tools/monitor_owned_server.py"
OBSERVER_RELATIVE = (
    "overlay-python/sglang/srt/layers/round9/round16_a1_observer.py"
)
BLOCK_CONTRACT_RELATIVE = "round16-a1-block.json"
SOURCE_MANIFEST_RELATIVE = "round16-a1-source-sha256.json"
RAW_RECEIPT_RELATIVE = "round16-a1-raw-receipt.json"
OVERLAY_RECEIPT_RELATIVE = "overlay-python/round9-candidate-overlay.json"
CONTROL_RELATIVES = (
    "round16/__init__.py",
    "round16/common/__init__.py",
    "round16/common/contracts.py",
    "round16/a/__init__.py",
    "round16/a/provider_manifest.py",
    "round16/a/registration.py",
    "round16/a/sglang_adapter.py",
    "round16/a/source_gate.py",
    "round16/a/fresh_a1_profile.py",
    "round16/a/fresh_a1_analysis.py",
    "round16/a/fresh_a1_snapshot.py",
    "round16/a/fresh_a1_device.py",
    "round16/a/fresh_a1_evidence.py",
    "round16/a/fresh_a1_remote.py",
    "tools/round16_a1_container_driver.py",
    "tools/round16_a1_remote_block.py",
)

V295_PRECONDITIONS: Mapping[str, str] = {
    KTEP_RELATIVE: "77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b",
    ROUND13_RELATIVE: "a4bd6c90bc05ec09e51d14194fb2eaf38e00dcb5578d48b91bfdfc1f4b392cc0",
    SCHEDULER_RELATIVE: "5b96f89ee24f49671b3c35960922d7f93b2133c62e087642d6bdde3ea20b1c6b",
    LAUNCHER_RELATIVE: "daad3cf0ad99c870c4a884ca0440c4f88d5c887984c4735dff2fe72b5c291f1c",
    FIXTURE_RELATIVE: "21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95",
    EXECUTION_BUDGET_RELATIVE: "665ded641f8848d65ca44776cf74dfd70909184339a619f3d70befee09c36974",
    ROUND11_DRIVER_RELATIVE: "38faa29ce4583254c59ee32bdf921329cf97aa0b04393a3a4bc5c9ef2d1ca2be",
    ROUND10_PROTOCOL_RELATIVE: "3b5d96864922ad4bd51d298ebefd039fe06e2de9a1bef8de4f800c8dcf2e0129",
    CLEANUP_RELATIVE: "b6df7cca2b2ac27d29acbd75515c58610372f0d20a13f050be11edfd7161a201",
    RESOURCE_GATE_RELATIVE: "8adbeadc39665fffd495a6ccd275bf7448c99414352cff531d27a5541ec012e9",
    MONITOR_RELATIVE: "fcffc16145353341ce6f986f0b80d127101211e9e0cfa917c71481e1c750392a",
}


class SnapshotPreconditionError(RuntimeError):
    """The frozen source or a unique injection anchor did not match."""


class PreflightError(RuntimeError):
    """An offline snapshot or ledger gate failed."""


class ControllerInputError(RuntimeError):
    """A device-stage controller input is not exactly reviewable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, error_type: type[RuntimeError]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type(f"JSON object required at {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _replace_line_once(source: str, stripped: str, replacement: Sequence[str]) -> str:
    lines = source.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.strip() == stripped]
    if len(matches) != 1:
        raise SnapshotPreconditionError(
            f"expected exactly one source anchor {stripped!r}; found {len(matches)}"
        )
    index = matches[0]
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    newline = "\n" if lines[index].endswith("\n") else ""
    lines[index : index + 1] = [indent + item + newline for item in replacement]
    return "".join(lines)


def _patch_scheduler(source: str) -> str:
    lines = source.splitlines(keepends=True)
    matches = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "before as round13_before, after as round13_after)"
    ]
    if len(matches) != 1:
        raise SnapshotPreconditionError(
            "expected exactly one round13 scheduler import close anchor"
        )
    index = matches[0]
    continuation_indent = lines[index][
        : len(lines[index]) - len(lines[index].lstrip())
    ]
    if len(continuation_indent) < 4:
        raise SnapshotPreconditionError("round13 scheduler import indentation drift")
    import_indent = continuation_indent[:-4]
    lines[index + 1 : index + 1] = [
        import_indent
        + "from sglang.srt.layers.round9.round16_a1_observer import (\n",
        continuation_indent
        + "before as round16_a1_before, after as round16_a1_after)\n",
    ]
    source = "".join(lines)
    source = _replace_line_once(
        source,
        "round13_before(self, batch)",
        ("round13_before(self, batch)", "round16_a1_before(self, batch)"),
    )
    return _replace_line_once(
        source,
        "round13_after(self)",
        ("round16_a1_after(self)", "round13_after(self)"),
    )


def _npu_replacement(
    *,
    assignment: str,
    boundary: str,
    call: str,
) -> tuple[str, ...]:
    return (
        'if os.environ.get("ROUND16_A1_PROFILE_ENABLE") == "1":',
        "    from sglang.srt.layers.round9.round16_a1_observer import observe_npu_call as _r16_observe_npu_call",
        f"    {assignment} = _r16_observe_npu_call(\"{boundary}\", self.kt_config.layer_idx, {call})",
        "else:",
        f"    {assignment} = {call}",
    )


def _patch_ktep(source: str) -> str:
    return _replace_line_once(
        source,
        "output = output + cpu_output",
        (
            'if os.environ.get("ROUND16_A1_PROFILE_ENABLE") == "1":',
            "    from sglang.srt.layers.round9.round16_a1_observer import observe_host_merge as _r16_observe_host_merge",
            "    output = _r16_observe_host_merge(self.kt_config.layer_idx, lambda: output + cpu_output)",
            "else:",
            "    output = output + cpu_output",
        ),
    )


def _patch_monitor(source: str) -> str:
    old = (
        "        r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297))', name) is not None"
    )
    new = (
        "        r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297)'\n"
        f"        r'|kt-round16-a1-[ab]-{RUNTIME_VERSION})', name) is not None"
    )
    if source.count(old) != 1:
        raise SnapshotPreconditionError("owned monitor allow-list anchor drift")
    source = source.replace(old, new)
    source = source.replace(
        "f.write(json.dumps({'time':time.time(),'action':'OWNED_EXITED','state':state['State']})+'\\n')",
        "f.write(json.dumps({'time':time.time(),'action':'OWNED_EXITED','container_id':state['Id'],'state':state['State']})+'\\n')",
    )
    source = source.replace(
        "f.write(json.dumps({'time':time.time(),'action':'OWNED_EXITED_DURING_SAMPLE'})+'\\n')",
        "f.write(json.dumps({'time':time.time(),'action':'OWNED_EXITED_DURING_SAMPLE','container_id':state['Id']})+'\\n')",
    )
    source = source.replace(
        "with a.output.open('a') as f:f.write(json.dumps(dict(time=time.time(),action='OWNED_EXITED_DURING_SAMPLE',sampling_error=repr(exc)))+'\\n')",
        "with a.output.open('a') as f:f.write(json.dumps(dict(time=time.time(),action='OWNED_EXITED_DURING_SAMPLE',container_id=state['Id'],sampling_error=repr(exc)))+'\\n')",
    )
    if source.count("'container_id':state['Id']") != 2 or source.count("container_id=state['Id']") != 1:
        raise SnapshotPreconditionError("owned monitor terminal ACK anchors drift")
    return source


def _patch_launcher(source: str, *, block_id: str) -> str:
    block = "A" if block_id == BLOCK_IDS["A"] else "B"
    lower = block.lower()
    source = _replace_line_once(
        source,
        "import argparse,hashlib,json,pathlib,re,shutil,subprocess,time",
        (
            "import argparse,hashlib,json,pathlib,re,shutil,subprocess,time",
            "import os",
        ),
    )
    source = _replace_line_once(
        source,
        "if continuous.get('status')!='PASS' or time.time()-continuous['ended_at']>600:",
        (
            "_r16_admission_sha = os.environ.get('ROUND16_A1_ADMISSION_SHA256', '')",
            "_r16_target = {'physical_id':12,'card_id':6,'chip_id':0,'pci_bdf':'0000:85:00.0','numa_nodes':[2,3,4,7]}",
            "_r16_validated = continuous.get('validated_at_epoch', continuous.get('ended_at'))",
            "if (len(_r16_admission_sha)!=64 or hashlib.sha256(a.cost_admission.read_bytes()).hexdigest()!=_r16_admission_sha or continuous.get('target')!=_r16_target or continuous.get('status')!='PASS' or not isinstance(_r16_validated,(int,float)) or isinstance(_r16_validated,bool) or not 0<=time.time()-_r16_validated<=600):",
        ),
    )
    source = _replace_line_once(
        source,
        "name='kt-round14a-a-gdn-abba-candidate-a-v295';out=root/'evidence'/'20260927-round14a-a-gdn-abba-candidate-a-v295'",
        (
            f"name='{runtime_container_name(block)}';out=root/'evidence'/'{runtime_evidence_name(block)}'",
        ),
    )
    source = _replace_line_once(
        source,
        "guard=subprocess.Popen(guard_args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)",
        (
            "guard=subprocess.Popen(guard_args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)",
            "try:",
            " (out/'guard-process.json').write_text(json.dumps({'pid':guard.pid,'container_id':inspect['Id']},indent=2)+'\\n')",
            "except BaseException:",
            " guard.terminate()",
            " try:",
            "  guard.wait(timeout=10)",
            " except subprocess.TimeoutExpired:",
            "  guard.kill()",
            "  guard.wait(timeout=10)",
            " raise",
        ),
    )
    source = _replace_line_once(
        source,
        "if budget.get('stage')!='ROUND14A_A_GDN_FULL_MODEL':raise RuntimeError('wrong Round14a GDN model budget')",
        (
            "if budget.get('stage')!='ROUND16_A1_FRESH_PROFILE':raise RuntimeError('wrong Round16 A1 model budget')",
        ),
    )
    source = _replace_line_once(
        source,
        "if digest(sglang/rel)!=sha:raise RuntimeError('layerwise MoE backend differs from device gate')",
        (
            "actual_source_sha = digest(sglang/rel)",
            "if actual_source_sha != sha:",
            " from round16.a.source_gate import require_parent_or_bound_delta as _r16_require_source_delta",
            " try:",
            "  _r16_require_source_delta(relative=rel, expected_parent_sha256=sha, actual_sha256=actual_source_sha, overlay_receipt=overlay)",
            " except RuntimeError as _r16_source_error:",
            "  raise RuntimeError('layerwise MoE backend differs from device gate') from _r16_source_error",
        ),
    )
    return _replace_line_once(
        source,
        "args+=['-e','ROUND14_GDN_BACKEND_ENABLE=1']",
        (
            "args+=['-e','ROUND14_GDN_BACKEND_ENABLE=1']",
            "_r16_a1_enabled = '1' if os.environ.get('ROUND16_A1_PROFILE_ENABLE') == '1' else '0'",
            "for _r16_a1_env in (",
            "    f'ROUND16_A1_PROFILE_ENABLE={_r16_a1_enabled}',",
            f"    'ROUND16_A1_BLOCK_ID={block_id}',",
            "    'ROUND16_A1_PROFILE_PATH=/out/round16-a1-profile.jsonl',",
            "    'ROUND16_A1_BINDINGS_PATH=/out/round16-a1-bindings.json',",
            "    'ROUND13_ATTRIBUTION_ENABLE=1',",
            "    'ROUND13_ATTRIBUTION_SPANS=/out/round13-spans.jsonl',",
            "    'ROUND13_ATTRIBUTION_BINDINGS=/out/round13-bindings.json',",
            "): args += ['-e', _r16_a1_env]",
        ),
    )


def _compile_python(path: Path) -> None:
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise SnapshotPreconditionError(f"patched Python does not compile: {path}") from exc


def _ignore_copy(_directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name == "__pycache__" or name.endswith(".pyc")}
    if ".git" in names:
        ignored.add(".git")
    return ignored


def _overlay_tree_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.relative_to(root) != Path("round9-candidate-overlay.json")
    }


def _overlay_tree_digest(root: Path) -> str:
    return canonical_json_sha256(_overlay_tree_files(root))


def _validated_parent_overlay_receipt(parent: Path) -> dict[str, Any]:
    overlay_root = parent / "overlay-python"
    receipt_path = parent / OVERLAY_RECEIPT_RELATIVE
    receipt = _read_json(receipt_path, error_type=SnapshotPreconditionError)
    if (
        receipt.get("hash_scope")
        != "PAYLOAD_EXCLUDING_OVERLAY_RECEIPT_AND_PYCACHE"
        or receipt.get("candidate_tree_sha256") != _overlay_tree_digest(overlay_root)
    ):
        raise SnapshotPreconditionError(
            "frozen parent overlay receipt does not bind its candidate tree"
        )
    return receipt


def _bind_round16_overlay(
    snapshot_root: Path,
    *,
    block: str,
    parent_candidate_tree_sha256: str,
    parent_overlay_files: Mapping[str, str],
) -> dict[str, Any]:
    overlay_root = snapshot_root / "overlay-python"
    receipt_path = snapshot_root / OVERLAY_RECEIPT_RELATIVE
    receipt = _read_json(receipt_path, error_type=SnapshotPreconditionError)
    changed = (
        Path(SCHEDULER_RELATIVE).relative_to("overlay-python").as_posix(),
        Path(KTEP_RELATIVE).relative_to("overlay-python").as_posix(),
        Path(OBSERVER_RELATIVE).relative_to("overlay-python").as_posix(),
    )
    receipt["parent_candidate_tree_sha256"] = parent_candidate_tree_sha256
    receipt["candidate_tree_sha256"] = _overlay_tree_digest(overlay_root)
    receipt["round16_a1_profile"] = {
        "schema": 1,
        "status": "INSTALLED_DEFAULT_OFF_PROFILE_ONLY",
        "runtime_version": RUNTIME_VERSION,
        "block_id": BLOCK_IDS[block],
        "parent_candidate_tree_sha256": parent_candidate_tree_sha256,
        "parent_overlay_files": {
            relative: parent_overlay_files.get(relative) for relative in changed
        },
        "modified_overlay_files": {
            relative: _sha256(overlay_root / relative) for relative in changed
        },
        "default_enabled": False,
        "compute_semantics": "UNCHANGED_OBSERVER_ONLY",
    }
    _write_json(receipt_path, receipt)
    if receipt["candidate_tree_sha256"] != _overlay_tree_digest(overlay_root):
        raise SnapshotPreconditionError(
            "Round16 overlay receipt does not bind its candidate tree"
        )
    return receipt


def _validate_preconditions(parent: Path, preconditions: Mapping[str, str]) -> None:
    if set(preconditions) != set(V295_PRECONDITIONS):
        raise SnapshotPreconditionError("precondition path set must match frozen v295 inputs")
    for relative in V295_PRECONDITIONS:
        path = parent / relative
        if not path.is_file():
            raise SnapshotPreconditionError(f"missing frozen input {relative}")
        actual = _sha256(path)
        expected = preconditions[relative]
        if actual != expected:
            raise SnapshotPreconditionError(
                f"SHA drift for {relative}: expected {expected}, observed {actual}"
            )


def _runtime_observer_source() -> Path:
    return Path(__file__).with_name("fresh_a1_runtime_observer.py")


def _repository_source(relative: str) -> Path:
    return Path(__file__).resolve().parents[2] / relative


def build_snapshot(
    parent: str | Path,
    output: str | Path,
    *,
    block: str,
    preconditions: Mapping[str, str] = V295_PRECONDITIONS,
) -> dict[str, Any]:
    """Verify frozen inputs, copy the tree, and patch only the new copy."""
    contract = block_contract(block)
    source_root = Path(parent).resolve()
    snapshot_root = Path(output).resolve()
    if source_root == snapshot_root or source_root in snapshot_root.parents:
        raise SnapshotPreconditionError(
            "snapshot output must be outside the frozen parent tree"
        )
    if snapshot_root.exists():
        raise SnapshotPreconditionError(f"fresh output required: {snapshot_root}")
    if not source_root.is_dir():
        raise SnapshotPreconditionError(f"parent snapshot is not a directory: {source_root}")
    _validate_preconditions(source_root, preconditions)
    parent_overlay = _validated_parent_overlay_receipt(source_root)
    parent_overlay_files = _overlay_tree_files(source_root / "overlay-python")

    observer_source = _runtime_observer_source()
    if not observer_source.is_file():
        raise SnapshotPreconditionError(f"runtime observer source missing: {observer_source}")

    shutil.copytree(source_root, snapshot_root, ignore=_ignore_copy)
    scheduler_path = snapshot_root / SCHEDULER_RELATIVE
    ktep_path = snapshot_root / KTEP_RELATIVE
    launcher_path = snapshot_root / LAUNCHER_RELATIVE
    monitor_path = snapshot_root / MONITOR_RELATIVE
    observer_path = snapshot_root / OBSERVER_RELATIVE

    scheduler_path.write_text(
        _patch_scheduler(scheduler_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    ktep_path.write_text(
        _patch_ktep(ktep_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    launcher_path.write_text(
        _patch_launcher(
            launcher_path.read_text(encoding="utf-8"),
            block_id=contract.block_id,
        ),
        encoding="utf-8",
    )
    monitor_path.write_text(
        _patch_monitor(monitor_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    observer_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(observer_source, observer_path)
    for relative in CONTROL_RELATIVES:
        source_path = _repository_source(relative)
        if not source_path.is_file():
            raise SnapshotPreconditionError(f"Round16 A1 control source missing: {relative}")
        target_path = snapshot_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
    overlay_receipt = _bind_round16_overlay(
        snapshot_root,
        block=contract.block,
        parent_candidate_tree_sha256=parent_overlay["candidate_tree_sha256"],
        parent_overlay_files=parent_overlay_files,
    )

    _write_json(
        snapshot_root / EXECUTION_BUDGET_RELATIVE,
        {
            "schema": 1,
            "stage": STAGE,
            "block_id": contract.block_id,
            "model_starts_limit": 1,
            "model_starts_used": 0,
            "starts": [],
            "generation_limit": GENERATIONS_PER_BLOCK,
            "abort_limit": 0,
            "automatic_retries_limit": 0,
        },
    )

    schedule = manifest_payload()["schedule"]
    schedule_sha256 = canonical_json_sha256(schedule)
    block_payload = {
        "schema": 1,
        "stage": STAGE,
        **asdict(contract),
        "default_enabled": False,
        "source_parent": "ROUND14A_V295_PROMOTED_GDN_S0",
        "generation_limit_total": GENERATION_LIMIT_TOTAL,
        "schedule_sha256": schedule_sha256,
        "raw_receipt": RAW_RECEIPT_RELATIVE,
        "execution": "DEVICE_REQUIRED_NOT_RUN",
    }
    _write_json(snapshot_root / BLOCK_CONTRACT_RELATIVE, block_payload)

    immutable_parent_files = tuple(
        relative
        for relative in V295_PRECONDITIONS
        if relative != EXECUTION_BUDGET_RELATIVE
    )
    tracked = immutable_parent_files + (
        OBSERVER_RELATIVE,
        BLOCK_CONTRACT_RELATIVE,
        OVERLAY_RECEIPT_RELATIVE,
        *CONTROL_RELATIVES,
    )
    for relative in tracked:
        if relative.endswith(".py"):
            _compile_python(snapshot_root / relative)
    files = {relative: _sha256(snapshot_root / relative) for relative in tracked}
    manifest = {
        "schema": 1,
        "stage": STAGE,
        "snapshot_id": contract.snapshot_id,
        "block_id": contract.block_id,
        "default_enabled": False,
        "parent_preconditions": dict(preconditions),
        "files": files,
        "execution_budget_contract": {
            "path": EXECUTION_BUDGET_RELATIVE,
            "initial_sha256": _sha256(snapshot_root / EXECUTION_BUDGET_RELATIVE),
            "allowed_transition": "INITIAL_UNSPENT_TO_SINGLE_MATCHING_START",
        },
        "source_manifest_scope": "EXACT_TRACKED_FILES_ONLY",
        "overlay_binding": {
            "path": OVERLAY_RECEIPT_RELATIVE,
            "runtime_version": RUNTIME_VERSION,
            "parent_candidate_tree_sha256": parent_overlay[
                "candidate_tree_sha256"
            ],
            "candidate_tree_sha256": overlay_receipt["candidate_tree_sha256"],
        },
        "device_execution": "NOT_RUN",
    }
    _write_json(snapshot_root / SOURCE_MANIFEST_RELATIVE, manifest)

    # Detect a concurrent or accidental change to the frozen parent during build.
    _validate_preconditions(source_root, preconditions)
    return manifest


def initial_ledger_payload() -> dict[str, Any]:
    return {
        "schema": 1,
        "stage": STAGE,
        "immutable": True,
        "generation_limit_total": GENERATION_LIMIT_TOTAL,
        "generation_limit_per_block": GENERATIONS_PER_BLOCK,
        "used_by_block": {block_id: 0 for block_id in BLOCK_IDS.values()},
        "request_ids": [],
        "update_policy": "COPY_ON_SPEND_NEW_SHA256",
        "automatic_retries": 0,
        "abort_controls": 0,
    }


def expected_request_ids(block_id: str) -> list[str]:
    """Return the only ordered request identities allowed for a frozen block."""
    if block_id not in BLOCK_IDS.values():
        raise ValueError("unknown immutable block identity")
    return [
        f"{block_id}:{item['ordinal']}:{item['role']}"
        for item in manifest_payload()["schedule"]
    ]


def controller_input_template(block: str) -> dict[str, Any]:
    """Return a frozen, disabled input template for later device review."""
    contract = block_contract(block)
    schedule = manifest_payload()["schedule"]
    return {
        "schema": 1,
        "stage": STAGE,
        "status": "FROZEN_OFFLINE_NOT_RUN",
        "enabled": False,
        "executable": False,
        "block": contract.block,
        "block_id": contract.block_id,
        "snapshot_id": contract.snapshot_id,
        "snapshot_root": None,
        "source_manifest": SOURCE_MANIFEST_RELATIVE,
        "source_manifest_sha256": None,
        "ledger_path": None,
        "ledger_sha256": None,
        "ledger_required_state": (
            "INITIAL_EMPTY" if block == "A" else "A_EXACTLY_SIX_B_EMPTY"
        ),
        "output_dir": None,
        "generation_limit": GENERATIONS_PER_BLOCK,
        "generation_limit_total": GENERATION_LIMIT_TOTAL,
        "schedule": schedule,
        "schedule_sha256": canonical_json_sha256(schedule),
        "automatic_retries": 0,
        "abort_controls": 0,
        "environment": {"ROUND16_A1_PROFILE_ENABLE": "0"},
        "required_transition": (
            "prepare_controller_input(..., enabled=True) after a fresh snapshot, "
            "exact source manifest, exact ledger SHA, and fresh output path"
        ),
        "execution": "DEVICE_REQUIRED_NOT_RUN",
    }


def _validate_ledger(
    ledger: Mapping[str, Any],
    *,
    block_id: str,
    error_type: type[RuntimeError],
) -> None:
    if ledger.get("stage") != STAGE or ledger.get("immutable") is not True:
        raise error_type("ledger must declare the exact stage and immutable=true")
    if ledger.get("generation_limit_total") != GENERATION_LIMIT_TOTAL:
        raise error_type("ledger total generation limit must be exactly 12")
    if ledger.get("generation_limit_per_block") != GENERATIONS_PER_BLOCK:
        raise error_type("ledger per-block generation limit must be exactly six")
    if ledger.get("automatic_retries") != 0 or ledger.get("abort_controls") != 0:
        raise error_type("ledger retries and abort controls must remain zero")
    counts = ledger.get("used_by_block")
    if not isinstance(counts, dict) or set(counts) != set(BLOCK_IDS.values()):
        raise error_type("ledger must contain the two immutable block identities")
    if any(type(value) is not int or not 0 <= value <= 6 for value in counts.values()):
        raise error_type("ledger block counts must be exact integers in [0, 6]")
    request_ids = ledger.get("request_ids")
    if (
        not isinstance(request_ids, list)
        or any(not isinstance(item, str) or not item for item in request_ids)
        or len(request_ids) != len(set(request_ids))
        or len(request_ids) != sum(counts.values())
    ):
        raise error_type("ledger request identities must be unique and match spent counts")
    if block_id not in counts:
        raise error_type("ledger does not contain requested block identity")
    if counts[block_id] != 0:
        raise error_type("controller requires a fresh current-block ledger allocation")
    if block_id == BLOCK_IDS["A"] and sum(counts.values()) != 0:
        raise error_type("block A controller requires a fresh shared ledger")
    if block_id == BLOCK_IDS["B"] and counts[BLOCK_IDS["A"]] != 6:
        raise error_type("block B requires block A to be exactly complete")
    if (
        block_id == BLOCK_IDS["B"]
        and request_ids != expected_request_ids(BLOCK_IDS["A"])
    ):
        raise error_type(
            "block B requires the ordered frozen block A request identities"
        )


def offline_preflight(
    snapshot_root: str | Path,
    *,
    block_id: str,
    ledger_path: str | Path,
) -> dict[str, Any]:
    """Validate files and budget without importing or probing a runtime."""
    root = Path(snapshot_root).resolve()
    contract = _read_json(root / BLOCK_CONTRACT_RELATIVE, error_type=PreflightError)
    manifest_path = root / SOURCE_MANIFEST_RELATIVE
    manifest = _read_json(manifest_path, error_type=PreflightError)
    if contract.get("block_id") != block_id or manifest.get("block_id") != block_id:
        raise PreflightError("block identity differs across controller inputs")
    if contract.get("enabled") is not False or contract.get("default_enabled") is not False:
        raise PreflightError("snapshot contract must remain default off")
    if manifest.get("default_enabled") is not False:
        raise PreflightError("source manifest must remain default off")
    schedule_sha256 = contract.get("schedule_sha256")
    if not isinstance(schedule_sha256, str) or len(schedule_sha256) != 64:
        raise PreflightError("snapshot block contract lacks a frozen schedule SHA")
    if (root / RAW_RECEIPT_RELATIVE).exists():
        raise PreflightError("fresh snapshot must not contain a raw receipt")
    files = manifest.get("files")
    immutable_expected = set(V295_PRECONDITIONS) - {EXECUTION_BUDGET_RELATIVE}
    if not isinstance(files, dict) or set(files) != immutable_expected | {
        OBSERVER_RELATIVE,
        BLOCK_CONTRACT_RELATIVE,
        OVERLAY_RECEIPT_RELATIVE,
        *CONTROL_RELATIVES,
    }:
        raise PreflightError("source manifest tracked-file set is not exact")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise PreflightError(f"source manifest mismatch for {relative}")
    overlay_binding = manifest.get("overlay_binding")
    overlay_receipt = _read_json(
        root / OVERLAY_RECEIPT_RELATIVE, error_type=PreflightError
    )
    actual_overlay_tree_sha256 = _overlay_tree_digest(root / "overlay-python")
    if (
        not isinstance(overlay_binding, dict)
        or overlay_binding.get("path") != OVERLAY_RECEIPT_RELATIVE
        or overlay_binding.get("runtime_version") != RUNTIME_VERSION
        or overlay_binding.get("candidate_tree_sha256")
        != actual_overlay_tree_sha256
        or overlay_receipt.get("candidate_tree_sha256")
        != actual_overlay_tree_sha256
        or overlay_receipt.get("round16_a1_profile", {}).get(
            "runtime_version"
        )
        != RUNTIME_VERSION
    ):
        raise PreflightError("Round16 overlay binding differs from installed tree")

    budget_contract = manifest.get("execution_budget_contract")
    if not isinstance(budget_contract, dict) or budget_contract.get("path") != EXECUTION_BUDGET_RELATIVE:
        raise PreflightError("execution budget contract is missing")
    initial_budget_sha = budget_contract.get("initial_sha256")
    if not isinstance(initial_budget_sha, str) or len(initial_budget_sha) != 64:
        raise PreflightError("execution budget initial SHA is missing")
    budget_path = root / EXECUTION_BUDGET_RELATIVE
    budget = _read_json(budget_path, error_type=PreflightError)
    expected_budget_keys = {
        "schema",
        "stage",
        "block_id",
        "model_starts_limit",
        "model_starts_used",
        "starts",
        "generation_limit",
        "abort_limit",
        "automatic_retries_limit",
    }
    if set(budget) != expected_budget_keys:
        raise PreflightError("execution budget schema drift")
    if budget.get("schema") != 1 or budget.get("stage") != STAGE:
        raise PreflightError("execution budget stage drift")
    if budget.get("block_id") != block_id:
        raise PreflightError("execution budget block identity drift")
    if budget.get("model_starts_limit") != 1 or budget.get("generation_limit") != GENERATIONS_PER_BLOCK:
        raise PreflightError("execution budget limits drift")
    if budget.get("automatic_retries_limit") != 0:
        raise PreflightError("execution budget retries must remain zero")
    if budget.get("abort_limit") != 0:
        raise PreflightError("execution budget abort controls must remain zero")
    used = budget.get("model_starts_used")
    starts = budget.get("starts")
    current_budget_sha = _sha256(budget_path)
    if type(used) is int and used == 0 and starts == []:
        if current_budget_sha != initial_budget_sha:
            raise PreflightError("initial execution budget SHA drift")
        budget_state = "INITIAL_UNSPENT"
        start_identity = None
    elif type(used) is int and used == 1 and isinstance(starts, list) and len(starts) == 1:
        start = starts[0]
        if not isinstance(start, dict) or set(start) != {
            "mode", "name", "container_id", "time", "output"
        }:
            raise PreflightError("execution budget single start schema drift")
        block = "A" if block_id == BLOCK_IDS["A"] else "B"
        expected_name = runtime_container_name(block)
        expected_output = (
            f"/mnt/data1/kt-round9/evidence/{runtime_evidence_name(block)}"
        )
        start_time = start.get("time")
        if (
            start.get("mode") != "OFF"
            or start.get("name") != expected_name
            or re.fullmatch(r"[0-9a-f]{64}", str(start.get("container_id"))) is None
            or isinstance(start_time, bool)
            or not isinstance(start_time, (int, float))
            or not math.isfinite(float(start_time))
            or start.get("output") != expected_output
        ):
            raise PreflightError("execution budget single start container identity drift")
        budget_state = "SINGLE_MATCHING_START"
        start_identity = dict(start)
    else:
        raise PreflightError("execution budget allows exactly zero or one model start")

    ledger = _read_json(Path(ledger_path).resolve(), error_type=PreflightError)
    _validate_ledger(ledger, block_id=block_id, error_type=PreflightError)
    return {
        "schema": 1,
        "stage": STAGE,
        "status": "PASS_OFFLINE_PREFLIGHT",
        "block_id": block_id,
        "snapshot_id": manifest["snapshot_id"],
        "enabled": False,
        "generation_limit": GENERATIONS_PER_BLOCK,
        "total_generation_limit": GENERATION_LIMIT_TOTAL,
        "source_manifest_status": "EXACT",
        "source_manifest_sha256": _sha256(manifest_path),
        "execution_budget_state": budget_state,
        "execution_budget_initial_sha256": initial_budget_sha,
        "execution_budget_current_sha256": current_budget_sha,
        "execution_budget_start": start_identity,
        "schedule_sha256": schedule_sha256,
        "ledger_sha256": _sha256(Path(ledger_path).resolve()),
        "device_probe_count": 0,
        "execution": "DEVICE_REQUIRED_NOT_RUN",
    }


def prepare_controller_input(
    snapshot_root: str | Path,
    *,
    block_id: str,
    ledger_path: str | Path,
    expected_ledger_sha256: str,
    expected_source_manifest_sha256: str | None = None,
    output_dir: str | Path,
    observer_path: str | Path | None = None,
    enabled: bool,
    required_budget_state: str | None = None,
) -> dict[str, Any]:
    """Create a reviewable plan only after every execution gate is exact."""
    if enabled is not True:
        raise ControllerInputError("controller must be explicitly enabled")
    if block_id not in BLOCK_IDS.values():
        raise ControllerInputError("unknown immutable block identity")
    ledger_file = Path(ledger_path).resolve()
    if _sha256(ledger_file) != expected_ledger_sha256:
        raise ControllerInputError("ledger SHA does not match reviewed input")
    output = Path(output_dir).resolve()
    if output.exists():
        raise ControllerInputError("fresh output directory is required")
    try:
        report = offline_preflight(
            snapshot_root,
            block_id=block_id,
            ledger_path=ledger_file,
        )
    except PreflightError as exc:
        message = str(exc)
        if "block identity" in message:
            raise ControllerInputError(message) from exc
        if "ledger" in message or "fresh" in message:
            raise ControllerInputError(message) from exc
        raise ControllerInputError(f"offline preflight failed: {message}") from exc
    schedule = manifest_payload()["schedule"]
    schedule_sha256 = canonical_json_sha256(schedule)
    if schedule_sha256 != report["schedule_sha256"]:
        raise ControllerInputError("controller schedule differs from snapshot contract")
    if (
        expected_source_manifest_sha256 is not None
        and report["source_manifest_sha256"] != expected_source_manifest_sha256
    ):
        raise ControllerInputError("controller reviewed source manifest SHA drift")
    if (
        required_budget_state is not None
        and report["execution_budget_state"] != required_budget_state
    ):
        raise ControllerInputError("controller execution budget state is not authorized")
    return {
        "schema": 1,
        "stage": STAGE,
        "executable": True,
        "block_id": block_id,
        "snapshot_id": report["snapshot_id"],
        "snapshot_root": str(Path(snapshot_root).resolve()),
        "source_manifest_sha256": report["source_manifest_sha256"],
        "execution_budget_state": report["execution_budget_state"],
        "execution_budget_required_state": required_budget_state,
        "execution_budget_initial_sha256": report[
            "execution_budget_initial_sha256"
        ],
        "execution_budget_current_sha256": report[
            "execution_budget_current_sha256"
        ],
        "ledger_path": str(ledger_file),
        "ledger_sha256": expected_ledger_sha256,
        "output_dir": str(output),
        "observer_path": str(
            Path(observer_path).resolve()
            if observer_path is not None
            else output / "round16-a1-profile.jsonl"
        ),
        "generation_limit": GENERATIONS_PER_BLOCK,
        "schedule": schedule,
        "schedule_sha256": schedule_sha256,
        "automatic_retries": 0,
        "abort_controls": 0,
        "environment": {
            "ROUND16_A1_PROFILE_ENABLE": "1",
            "ROUND16_A1_BLOCK_ID": block_id,
        },
        "execution_status": "AUTHORIZED_INPUT_ONLY_NOT_RUN",
    }


def raw_receipt(
    controller: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, Any]],
    *,
    observer_path: str,
) -> dict[str, Any]:
    """Form an unreviewed raw device receipt without selecting a candidate."""
    if controller.get("executable") is not True:
        raise ControllerInputError("raw receipt requires an executable controller input")
    if len(generation_rows) != GENERATIONS_PER_BLOCK:
        raise ControllerInputError("raw receipt requires exactly six generation rows")
    ordinals = [row.get("ordinal") for row in generation_rows]
    request_ids = [row.get("request_id") for row in generation_rows]
    if ordinals != list(range(1, GENERATIONS_PER_BLOCK + 1)):
        raise ControllerInputError("raw receipt ordinals must be exactly 1 through six")
    if (
        any(not isinstance(item, str) or not item for item in request_ids)
        or len(set(request_ids)) != GENERATIONS_PER_BLOCK
    ):
        raise ControllerInputError("raw receipt request identities must be unique")
    if not isinstance(observer_path, str) or not observer_path:
        raise ControllerInputError("observer path must be a non-empty string")
    return {
        "schema": 1,
        "stage": STAGE,
        "status": "RAW_DEVICE_RESULT_UNREVIEWED",
        "block_id": controller["block_id"],
        "snapshot_id": controller["snapshot_id"],
        "source_manifest_sha256": controller["source_manifest_sha256"],
        "execution_budget_initial_sha256": controller[
            "execution_budget_initial_sha256"
        ],
        "execution_budget_after_model_start_sha256": controller[
            "execution_budget_current_sha256"
        ],
        "schedule_sha256": controller["schedule_sha256"],
        "ledger_sha256_before": controller["ledger_sha256"],
        "generation_attempts": GENERATIONS_PER_BLOCK,
        "generations": [dict(row) for row in generation_rows],
        "observer_path": observer_path,
        "observer_status": "UNVALIDATED_RAW_INPUT",
        "candidate_selected": None,
        "execution_review": "REQUIRED_BEFORE_CANDIDATE_SELECTION",
    }


def _spend_ledger_payload(
    ledger: Mapping[str, Any],
    *,
    block_id: str,
    request_id: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger, allow_nan=False))
    counts = updated["used_by_block"]
    if counts[block_id] >= GENERATIONS_PER_BLOCK:
        raise ControllerInputError("current block generation budget is exhausted")
    if sum(counts.values()) >= GENERATION_LIMIT_TOTAL:
        raise ControllerInputError("shared twelve-generation budget is exhausted")
    if request_id in updated["request_ids"]:
        raise ControllerInputError("duplicate controller request identity")
    counts[block_id] += 1
    updated["request_ids"].append(request_id)
    return updated


def execute_controller(
    controller: Mapping[str, Any],
    driver: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute the fixed six-item queue through an injected device driver.

    This module supplies no device driver.  The function revalidates every
    frozen input immediately before creating the output directory.  Each
    attempted request is written to a new copy-on-spend ledger before the
    injected driver is called, so no automatic retry can hide budget use.
    """
    if controller.get("stage") != STAGE or controller.get("executable") is not True:
        raise ControllerInputError("controller input is not executable")
    environment = controller.get("environment")
    if not isinstance(environment, dict) or environment.get(
        "ROUND16_A1_PROFILE_ENABLE"
    ) != "1":
        raise ControllerInputError("controller input is not explicitly enabled")
    block_id = controller.get("block_id")
    if block_id not in BLOCK_IDS.values():
        raise ControllerInputError("unknown immutable block identity")
    if not callable(driver):
        raise ControllerInputError("controller requires an injected callable driver")
    schedule = controller.get("schedule")
    schedule_sha256 = controller.get("schedule_sha256")
    if (
        not isinstance(schedule, list)
        or not isinstance(schedule_sha256, str)
        or canonical_json_sha256(schedule) != schedule_sha256
    ):
        raise ControllerInputError("controller schedule SHA does not match its queue")

    ledger_path = Path(str(controller.get("ledger_path"))).resolve()
    expected_ledger_sha256 = controller.get("ledger_sha256")
    if not isinstance(expected_ledger_sha256, str) or _sha256(
        ledger_path
    ) != expected_ledger_sha256:
        raise ControllerInputError("ledger SHA changed after controller preparation")
    snapshot_root = Path(str(controller.get("snapshot_root"))).resolve()
    try:
        preflight = offline_preflight(
            snapshot_root,
            block_id=block_id,
            ledger_path=ledger_path,
        )
    except PreflightError as exc:
        raise ControllerInputError(f"controller preflight failed: {exc}") from exc
    if preflight["source_manifest_sha256"] != controller.get(
        "source_manifest_sha256"
    ):
        raise ControllerInputError("source manifest changed after controller preparation")
    if preflight["execution_budget_current_sha256"] != controller.get(
        "execution_budget_current_sha256"
    ):
        raise ControllerInputError("execution budget changed after controller preparation")
    required_budget_state = controller.get("execution_budget_required_state")
    if (
        required_budget_state is not None
        and preflight["execution_budget_state"] != required_budget_state
    ):
        raise ControllerInputError("controller execution budget state changed")
    if preflight["schedule_sha256"] != schedule_sha256:
        raise ControllerInputError("controller schedule SHA differs from snapshot contract")
    output_dir = Path(str(controller.get("output_dir"))).resolve()
    if output_dir.exists():
        raise ControllerInputError("fresh output directory is required")

    ledger = _read_json(ledger_path, error_type=ControllerInputError)
    _validate_ledger(ledger, block_id=block_id, error_type=ControllerInputError)
    if len(schedule) != GENERATIONS_PER_BLOCK:
        raise ControllerInputError("controller schedule must contain exactly six items")

    output_dir.mkdir(parents=False, exist_ok=False)
    ledger_after_path = output_dir / "round16-a1-ledger-after.json"
    rows: list[dict[str, Any]] = []
    for item in schedule:
        request_id = f"{block_id}:{item['ordinal']}:{item['role']}"
        ledger = _spend_ledger_payload(
            ledger,
            block_id=block_id,
            request_id=request_id,
        )
        _write_json(ledger_after_path, ledger)
        result = driver(
            dict(item),
            request_id=request_id,
            output_dir=output_dir,
            snapshot_root=snapshot_root,
            environment=dict(environment),
        )
        if not isinstance(result, Mapping):
            raise ControllerInputError("driver must return a mapping receipt")
        rows.append(
            {
                "ordinal": item["ordinal"],
                "role": item["role"],
                "request_id": request_id,
                "status": result.get("status", "UNKNOWN"),
                "schedule": dict(item),
                "driver": dict(result),
            }
        )

    receipt = raw_receipt(
        controller,
        rows,
        observer_path=str(controller.get("observer_path")),
    )
    receipt["ledger_sha256_after"] = _sha256(ledger_after_path)
    receipt_path = output_dir / RAW_RECEIPT_RELATIVE
    _write_json(receipt_path, receipt)
    return receipt


__all__ = (
    "ControllerInputError",
    "PreflightError",
    "SnapshotPreconditionError",
    "V295_PRECONDITIONS",
    "build_snapshot",
    "canonical_json_sha256",
    "controller_input_template",
    "execute_controller",
    "expected_request_ids",
    "initial_ledger_payload",
    "offline_preflight",
    "prepare_controller_input",
    "raw_receipt",
)
