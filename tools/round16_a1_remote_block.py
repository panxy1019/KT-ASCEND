#!/usr/bin/env python3
"""Launch, drive, supervise, retire, and archive one fresh A1 block.

This is a device-stage script.  Importing it is inert.  One invocation owns
one distinct container, one evidence directory, one model start, and exactly
six HTTP generation attempts.  It has no retry path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from round16.a.fresh_a1_profile import BLOCK_IDS, runtime_evidence_name
from round16.a.fresh_a1_analysis import (
    analysis_payload,
    load_evidence_bundle,
    write_analysis_outputs,
)
from round16.a.fresh_a1_evidence import write_analysis_input
from round16.a.fresh_a1_snapshot import (
    PreflightError,
    expected_request_ids,
    offline_preflight,
)
from round16.a.fresh_a1_remote import (
    REMOTE_BASE,
    block_runtime_contract,
    launch_command,
    validate_admission,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _container_path(path: Path) -> str:
    return "/round9/" + str(path.resolve().relative_to(REMOTE_BASE.resolve()))


def _inspect(container: str) -> dict:
    values = json.loads(
        subprocess.check_output(["docker", "inspect", container], text=True, timeout=30)
    )
    if len(values) != 1:
        raise RuntimeError("owned container did not resolve exactly once")
    return values[0]


def archive_admission_evidence(
    admission_path: Path,
    evidence_root: Path,
    *,
    expected_sha256: str,
) -> dict:
    """Copy the reviewed wrapper and its two SHA-bound raw rounds into evidence."""
    output = evidence_root / "launcher-admission"
    output.mkdir()
    result_copy = output / "result.json"
    with admission_path.open("rb") as reader, result_copy.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    result_sha = _sha256(result_copy)
    if result_sha != expected_sha256:
        raise RuntimeError("launcher admission result SHA changed before archive")
    result = json.loads(result_copy.read_text(encoding="utf-8"))
    names = result.get("rounds")
    if (
        not isinstance(names, list)
        or len(names) != 2
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            for name in names
        )
        or len(set(names)) != 2
    ):
        raise RuntimeError("validated admission round order is unavailable for archive")
    reviewed_raw = result.get("raw_sha256")
    if not isinstance(reviewed_raw, dict) or set(reviewed_raw) != set(names):
        raise RuntimeError("validated admission raw round SHA map is unavailable for archive")
    sources = [admission_path, *(admission_path.parent / name for name in names)]
    files = {}
    for index, source in enumerate(sources):
        archived_name = "result.json" if index == 0 else source.name
        destination = output / archived_name
        if index != 0:
            with source.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            if _sha256(destination) != reviewed_raw[archived_name]:
                raise RuntimeError(
                    f"launcher admission raw round SHA changed before archive: {archived_name}"
                )
        identity = {
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
        if index != 0:
            raw = json.loads(destination.read_text(encoding="utf-8"))
            identity.update(started_at=raw["started_at"], ended_at=raw["ended_at"])
        else:
            identity.update(
                started_at=result["started_at"],
                ended_at=result["ended_at"],
                validated_at_epoch=result["validated_at_epoch"],
            )
        files[archived_name] = identity
    manifest = {
        "schema": 1,
        "status": "FROZEN_LAUNCHER_ADMISSION",
        "target": result["target"],
        "rounds": list(names),
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _terminate_bound_guard(
    pid: int,
    container: str,
    monitor_path: Path,
    *,
    _process=None,
) -> str:
    """Terminate only the guard process whose command line binds this container."""
    process = Path(f"/proc/{pid}") if _process is None else _process
    if not process.exists():
        return "ALREADY_EXITED_BEFORE_CLEANUP"
    try:
        arguments = [
            item.decode(errors="surrogateescape")
            for item in (process / "cmdline").read_bytes().split(b"\0")
            if item
        ]
    except OSError as exc:
        if not process.exists():
            return "ALREADY_EXITED_BEFORE_CLEANUP"
        raise RuntimeError("recorded guard process is unavailable") from exc
    expected_monitor = str(monitor_path.resolve())
    if expected_monitor not in arguments:
        raise RuntimeError("recorded guard PID command does not match monitor path")
    try:
        index = arguments.index("--container")
    except ValueError as exc:
        raise RuntimeError("recorded guard PID lacks container binding") from exc
    if index + 1 >= len(arguments) or arguments[index + 1] != container:
        raise RuntimeError("recorded guard PID container binding differs")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while process.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("recorded guard did not terminate after SIGTERM")
        time.sleep(0.05)
    return "TERMINATED_BEFORE_REMOVE"


def cleanup_owned_container(
    *,
    container: str,
    container_id: str,
    monitor_path: Path,
    guard_process_path: Path | None = None,
    supervisor_started_at: float,
    inspect_fn,
    run_fn,
    wait_guard_fn,
    terminate_guard_fn=_terminate_bound_guard,
    clock=time.time,
) -> dict:
    """Stop if live, bind the terminal ACK to the full ID, then always remove."""
    current = inspect_fn(container)
    if current.get("Id") != container_id:
        raise RuntimeError("cleanup container identity changed")
    state = current.get("State", {})
    running = state.get("Running") is True
    status = state.get("Status")
    if not running and status == "created":
        if state.get("Pid") != 0:
            raise RuntimeError("created container unexpectedly has a process PID")
        termination = "NOT_STARTED_NO_GUARD_PROCESS_RECEIPT"
        if guard_process_path is not None and guard_process_path.is_file():
            try:
                guard = json.loads(guard_process_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("cannot read pre-start guard process receipt") from exc
            if guard.get("container_id") != container_id:
                raise RuntimeError("pre-start guard process receipt container identity drift")
            guard_pid = guard.get("pid")
            if type(guard_pid) is not int or guard_pid <= 1:
                raise RuntimeError("pre-start guard process receipt PID is invalid")
            outcome = terminate_guard_fn(guard_pid, container, monitor_path)
            termination = outcome or "TERMINATED_BEFORE_REMOVE"
        run_fn(["docker", "rm", container_id], check=True, timeout=30)
        return {
            "guard_termination": termination,
            "owned_container_removed": True,
            "was_running_at_cleanup": False,
            "container_state": "created",
            "verified_container_pid": 0,
        }
    not_before = supervisor_started_at
    if running:
        not_before = float(clock())
        run_fn(["docker", "stop", "-t", "10", container_id], check=True, timeout=30)
    elif status not in {"exited", "dead"}:
        raise RuntimeError("owned container is neither running nor terminal")
    event = wait_guard_fn(monitor_path, not_before=not_before)
    if event.get("container_id") != container_id:
        raise RuntimeError("guard terminal ACK is not bound to the owned container ID")
    run_fn(["docker", "rm", container_id], check=True, timeout=30)
    return {
        "guard_exit_event": event,
        "owned_container_removed": True,
        "was_running_at_cleanup": running,
        "container_state": status,
    }


def record_cleanup_failure(report: dict, error: BaseException) -> None:
    """Attach cleanup evidence without erasing an earlier launch/driver failure."""
    report["cleanup_error"] = repr(error)
    if report.get("status") != "FAIL":
        report["status"] = "FAIL_CLEANUP"


def record_nonblocking_post_cleanup_probe_failure(
    report: dict, error: BaseException
) -> None:
    """Record an optional resource snapshot without invalidating completed work.

    The next model start always performs a fresh two-round admission.  Once the
    owned container is removed and its guard ACK is bound, a conservative
    post-exit NUMA estimate is therefore diagnostic rather than a second
    teardown gate.  In particular, project-owned read-only file cache may still
    be resident until the later cache-retirement hook runs.
    """
    report["post_cleanup_resource_probe_error"] = repr(error)
    report["physical12_after"] = {
        "target": 12,
        "status": "DEFERRED_TO_NEXT_FRESH_ADMISSION",
        "blocking": False,
    }


def validate_snapshot_prelaunch(
    snapshot: Path,
    *,
    block_id: str,
    source_manifest_sha256: str,
    ledger: Path,
) -> dict:
    """Recheck reviewed immutable source and the unspent model-start budget."""
    manifest = snapshot / "round16-a1-source-sha256.json"
    if _sha256(manifest) != source_manifest_sha256:
        raise RuntimeError("source manifest SHA differs from reviewed prelaunch input")
    try:
        report = offline_preflight(snapshot, block_id=block_id, ledger_path=ledger)
    except PreflightError as exc:
        raise RuntimeError(f"source prelaunch gate failed: {exc}") from exc
    if report.get("source_manifest_sha256") != source_manifest_sha256:
        raise RuntimeError("source prelaunch report manifest SHA drift")
    if report.get("execution_budget_state") != "INITIAL_UNSPENT":
        raise RuntimeError("source prelaunch gate requires an unspent model-start budget")
    return report


def validate_snapshot_after_model_start(
    snapshot: Path,
    *,
    block_id: str,
    source_manifest_sha256: str,
    ledger: Path,
) -> dict:
    """Require the only allowed mutable-budget transition after launcher return."""
    try:
        report = offline_preflight(snapshot, block_id=block_id, ledger_path=ledger)
    except PreflightError as exc:
        raise RuntimeError(f"post-launch source/budget gate failed: {exc}") from exc
    if report.get("source_manifest_sha256") != source_manifest_sha256:
        raise RuntimeError("post-launch source manifest SHA drift")
    if report.get("execution_budget_state") != "SINGLE_MATCHING_START":
        raise RuntimeError("post-launch budget lacks exactly one matching model start")
    return report


A_BLOCK_RAW_RECEIPT = (
    REMOTE_BASE
    / f"evidence/{runtime_evidence_name('A')}"
    / "round16-a1-controller/round16-a1-raw-receipt.json"
)


def validate_block_b_predecessor(
    receipt_path: Path,
    *,
    ledger_path: Path,
    ledger_sha256: str,
) -> dict:
    """Bind block B to the exact completed block A run and its ledger bytes."""
    receipt_path = receipt_path.resolve()
    evidence_root = receipt_path.parent.parent
    expected_ledger_path = (
        evidence_root / "round16-a1-controller/round16-a1-ledger-after.json"
    ).resolve()
    if ledger_path.resolve() != expected_ledger_path:
        raise RuntimeError("block B ledger is not the fixed block A evidence ledger")
    if _sha256(ledger_path) != ledger_sha256:
        raise RuntimeError("block B predecessor ledger SHA differs from supplied bytes")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("block B predecessor raw receipt is unavailable") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "RAW_DEVICE_RESULT_UNREVIEWED"
        or receipt.get("block_id") != BLOCK_IDS["A"]
        or receipt.get("generation_attempts") != 6
    ):
        raise RuntimeError("block B predecessor is not the exact completed block A receipt")
    expected_ids = expected_request_ids(BLOCK_IDS["A"])
    expected_schedule = [
        (ordinal, request_id.rsplit(":", 1)[1], request_id)
        for ordinal, request_id in enumerate(expected_ids, 1)
    ]
    generations = receipt.get("generations")
    observed = (
        [
            (
                row.get("ordinal"),
                row.get("role"),
                row.get("request_id"),
            )
            for row in generations
        ]
        if isinstance(generations, list)
        and all(isinstance(row, dict) for row in generations)
        else None
    )
    if observed != expected_schedule or any(
        row.get("status") != "COMPLETE" for row in generations or ()
    ):
        raise RuntimeError("block B predecessor lacks the exact six block A requests")
    if receipt.get("ledger_sha256_after") != ledger_sha256:
        raise RuntimeError("block B predecessor receipt ledger SHA does not bind the supplied ledger")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("request_ids") != expected_ids:
        raise RuntimeError("block B predecessor ledger lacks frozen block A request identities")
    supervisor_path = evidence_root / "round16-a1-supervisor.json"
    archive_path = evidence_root / "round16-a1-archive-sha256.json"
    try:
        supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("block B predecessor cleanup/archive evidence is unavailable") from exc
    if (
        not isinstance(supervisor, dict)
        or supervisor.get("block_id") != BLOCK_IDS["A"]
        or supervisor.get("status") != "PASS_FRESH_A1_BLOCK_RAW_UNREVIEWED"
        or supervisor.get("owned_container_removed") is not True
        or not isinstance(supervisor.get("physical12_after"), dict)
        or not supervisor["physical12_after"]
    ):
        raise RuntimeError("block A cleanup terminal state is not complete")
    files = archive.get("files") if isinstance(archive, dict) else None
    expected_archive_files = {
        "round16-a1-supervisor.json": _sha256(supervisor_path),
        "round16-a1-controller/round16-a1-raw-receipt.json": _sha256(
            receipt_path
        ),
        "round16-a1-controller/round16-a1-ledger-after.json": ledger_sha256,
    }
    if (
        archive.get("status")
        != "FROZEN_AFTER_GUARD_ACK_AND_CONTAINER_REMOVAL"
        or not isinstance(files, dict)
        or any(files.get(name) != sha for name, sha in expected_archive_files.items())
    ):
        raise RuntimeError("block A archive does not bind cleanup, receipt, and ledger")
    return {
        "status": "PASS_EXACT_BLOCK_A_PREDECESSOR",
        "raw_receipt_path": str(receipt_path.resolve()),
        "raw_receipt_sha256": _sha256(receipt_path),
        "supervisor_sha256": _sha256(supervisor_path),
        "archive_sha256": _sha256(archive_path),
        "ledger_sha256_after": ledger_sha256,
        "request_ids": expected_ids,
    }


def _archive_manifest(out: Path) -> dict:
    admission_manifest_path = out / "launcher-admission/manifest.json"
    try:
        admission_manifest = json.loads(
            admission_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("archive launcher admission manifest is unavailable") from exc
    admission_rounds = admission_manifest.get("rounds")
    if (
        not isinstance(admission_rounds, list)
        or len(admission_rounds) != 2
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            for name in admission_rounds
        )
        or len(set(admission_rounds)) != 2
    ):
        raise RuntimeError("archive launcher admission round identities are invalid")
    required = (
        "admission.json",
        "container-created.json",
        "guard-process.json",
        "guard-ready.json",
        "model-start.json",
        "monitor.jsonl",
        "round16-a1-profile.jsonl",
        "round16-a1-bindings.json",
        "round13-spans.jsonl",
        "round13-bindings.json",
        "round16-a1-observer.jsonl",
        "round16-a1-observer-metadata.json",
        "round16-a1-controller/round16-a1-ledger-after.json",
        "round16-a1-controller/round16-a1-raw-receipt.json",
        "round16-a1-supervisor.json",
        "launcher-admission/result.json",
        *(f"launcher-admission/{name}" for name in admission_rounds),
        "launcher-admission/manifest.json",
    )
    files = {}
    for relative in required:
        path = out / relative
        if not path.is_file():
            raise RuntimeError(f"archive required output is missing: {relative}")
        files[relative] = _sha256(path)
    analysis_root = out / "round16-a1-analysis"
    if analysis_root.is_dir():
        for name in (
            "A1_CRITICAL_PATH.json",
            "A1_CRITICAL_PATH.md",
            "A_REGION_DEPENDENCY_DAG.json",
            "A_CANDIDATE_DECISION.json",
        ):
            path = analysis_root / name
            if not path.is_file():
                raise RuntimeError(f"two-block analysis output is missing: {name}")
            files[f"round16-a1-analysis/{name}"] = _sha256(path)
    large_logs = {}
    for name in ("server.log", "guard.log", "round16-a1-driver.log"):
        path = out / name
        if not path.is_file():
            raise RuntimeError(f"archive required log is missing: {name}")
        large_logs[name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema": 1,
        "status": "FROZEN_AFTER_GUARD_ACK_AND_CONTAINER_REMOVAL",
        "files": files,
        "large_logs": large_logs,
    }


def run_block(
    *,
    block: str,
    snapshot: Path,
    source_manifest_sha256: str,
    admission: Path,
    admission_sha256: str,
    ledger: Path,
    ledger_sha256: str,
) -> dict:
    contract = block_runtime_contract(block)
    validate_admission(admission, expected_sha256=admission_sha256)
    if _sha256(ledger) != ledger_sha256:
        raise RuntimeError("reviewed generation ledger SHA mismatch")
    prelaunch = validate_snapshot_prelaunch(
        snapshot,
        block_id=contract["block_id"],
        source_manifest_sha256=source_manifest_sha256,
        ledger=ledger,
    )
    predecessor = None
    if block == "B":
        predecessor = validate_block_b_predecessor(
            A_BLOCK_RAW_RECEIPT,
            ledger_path=ledger,
            ledger_sha256=ledger_sha256,
        )
    block_doc = json.loads((snapshot / "round16-a1-block.json").read_text())
    if block_doc.get("block_id") != contract["block_id"]:
        raise RuntimeError("snapshot/block runtime identity mismatch")
    out = REMOTE_BASE / "evidence" / contract["evidence"]
    if out.exists():
        raise FileExistsError(f"fresh evidence directory required: {out}")
    report = {
        "schema": 1,
        "stage": contract["stage"],
        "block_id": contract["block_id"],
        "status": "LAUNCHING",
        "model_start_attempts": 0,
        "driver_process_starts": 0,
        "generation_limit": 6,
        "automatic_retries": 0,
        "started_at": time.time(),
        "admission_sha256": admission_sha256,
        "ledger_sha256_before": ledger_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "execution_budget_initial_sha256": prelaunch[
            "execution_budget_initial_sha256"
        ],
    }
    if predecessor is not None:
        report["block_a_predecessor"] = predecessor
    container_id = None
    report_path = out / "round16-a1-supervisor.json"
    launch = launch_command(snapshot, admission)
    launch_environment = dict(os.environ)
    launch_environment["ROUND16_A1_ADMISSION_SHA256"] = admission_sha256
    launch_environment["ROUND16_A1_PROFILE_ENABLE"] = "1"
    try:
        report["model_start_attempts"] = 1
        subprocess.run(launch, check=True, timeout=300, env=launch_environment)
        postlaunch = validate_snapshot_after_model_start(
            snapshot,
            block_id=contract["block_id"],
            source_manifest_sha256=source_manifest_sha256,
            ledger=ledger,
        )
        report["execution_budget_after_model_start_sha256"] = postlaunch[
            "execution_budget_current_sha256"
        ]
        report["launcher_admission_evidence"] = archive_admission_evidence(
            admission, out, expected_sha256=admission_sha256
        )
        start = json.loads((out / "model-start.json").read_text())
        if start.get("name") != contract["container"]:
            raise RuntimeError("launcher created a different container identity")
        container_id = start.get("container_id")
        if not isinstance(container_id, str) or len(container_id) != 64:
            raise RuntimeError("launcher model-start lacks full container ID")
        budget_start = postlaunch.get("execution_budget_start")
        if (
            not isinstance(budget_start, dict)
            or budget_start.get("container_id") != container_id
            or budget_start.get("name") != contract["container"]
        ):
            raise RuntimeError("model-start receipt differs from spent execution budget")
        report.update(status="WAITING_FOR_SERVER", container_id=container_id)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

        deadline = time.monotonic() + 21600
        while time.monotonic() < deadline:
            current = _inspect(contract["container"])
            if current["Id"] != container_id or not current["State"]["Running"]:
                raise RuntimeError("owned server exited or changed identity before readiness")
            log = out / "server.log"
            if log.exists():
                if log.stat().st_size > 512 * 1024**2:
                    raise RuntimeError("server log exceeded the reviewed ceiling")
                with log.open("rb") as source:
                    source.seek(max(0, log.stat().st_size - 65536))
                    tail = source.read().decode(errors="replace")
                if (
                    "Uvicorn running on http://127.0.0.1:30091" in tail
                    or "The server is fired up and ready to roll!" in tail
                ):
                    break
            time.sleep(5)
        else:
            raise TimeoutError("fresh A1 server readiness deadline exceeded")

        snapshot_in = _container_path(snapshot)
        ledger_in = _container_path(ledger)
        command = [
            "docker", "exec",
            "-e", f"PYTHONPATH={snapshot_in}",
            "-e", "ROUND16_A1_PROFILE_ENABLE=1",
            "-e", f"ROUND16_A1_BLOCK_ID={contract['block_id']}",
            container_id,
            "python", f"{snapshot_in}/tools/round16_a1_container_driver.py",
            "--snapshot", snapshot_in,
            "--block-id", contract["block_id"],
            "--ledger", ledger_in,
            "--ledger-sha256", ledger_sha256,
            "--source-manifest-sha256", source_manifest_sha256,
            "--output", "/out/round16-a1-controller",
            "--observer-receipt-path", str(out / "round16-a1-observer.jsonl"),
        ]
        report.update(status="RUNNING_SIX_REQUEST_DRIVER", driver_process_starts=1, driver_command=command)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        with (out / "round16-a1-driver.log").open("x") as stream:
            subprocess.run(command, check=True, timeout=10860, stdout=stream, stderr=subprocess.STDOUT)
        raw = json.loads((out / "round16-a1-controller/round16-a1-raw-receipt.json").read_text())
        if (
            raw.get("generation_attempts") != 6
            or [row.get("status") for row in raw.get("generations", [])] != ["COMPLETE"] * 6
        ):
            raise RuntimeError("six-request driver raw receipt is incomplete")
        report["status"] = "DRIVER_COMPLETE_WAITING_CLEANUP"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["cleanup_started_at"] = time.time()
        if out.exists() and not (out / "launcher-admission/manifest.json").is_file():
            try:
                report["launcher_admission_evidence"] = archive_admission_evidence(
                    admission, out, expected_sha256=admission_sha256
                )
            except BaseException as archive_exc:
                report["admission_evidence_error"] = repr(archive_exc)
                if report.get("status") != "FAIL":
                    report["status"] = "FAIL_ADMISSION_EVIDENCE"
        try:
            if container_id is None and (out / "container-created.json").is_file():
                container_id = json.loads((out / "container-created.json").read_text())["Id"]
            if container_id is not None:
                cleanup = _load(snapshot / "m3_cleanup.py", "_r16_m3_cleanup")
                report.update(
                    cleanup_owned_container(
                        container=contract["container"],
                        container_id=container_id,
                        monitor_path=out / "monitor.jsonl",
                        guard_process_path=out / "guard-process.json",
                        supervisor_started_at=report["started_at"],
                        inspect_fn=_inspect,
                        run_fn=subprocess.run,
                        wait_guard_fn=cleanup.wait_for_guard_exit,
                    )
                )
        except BaseException as cleanup_exc:
            record_cleanup_failure(report, cleanup_exc)
        if report.get("owned_container_removed") is True:
            try:
                resources = _load(
                    snapshot / "source/tools/candidate_resources.py",
                    "_r16_resources",
                )
                post = resources.sample(
                    "MXFP4",
                    "shared",
                    "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef",
                )
                report["physical12_after"] = {
                    "training_pids": post["training_pids"],
                    "baseline_hbm_mib": post["baseline_hbm_mib"],
                    "target": 12,
                }
            except BaseException as resource_exc:
                record_nonblocking_post_cleanup_probe_failure(report, resource_exc)
        report["ended_at"] = time.time()
        if out.exists():
            report_path.write_text(json.dumps(report, indent=2) + "\n")

    if report["status"] != "DRIVER_COMPLETE_WAITING_CLEANUP" or not report.get("owned_container_removed"):
        raise RuntimeError("fresh A1 block did not complete cleanup")
    try:
        report["status"] = "NORMALIZING_ANALYSIS_INPUT"
        raw_receipt = out / "round16-a1-controller/round16-a1-raw-receipt.json"
        observer_rows = out / "round16-a1-observer.jsonl"
        observer_metadata = out / "round16-a1-observer-metadata.json"
        write_analysis_input(
            receipt_path=raw_receipt,
            runtime_rows_path=out / "round16-a1-profile.jsonl",
            runtime_bindings_path=out / "round16-a1-bindings.json",
            round13_rows_path=out / "round13-spans.jsonl",
            round13_bindings_path=out / "round13-bindings.json",
            output_rows_path=observer_rows,
            output_metadata_path=observer_metadata,
        )
        bundle = load_evidence_bundle(raw_receipt, observer_metadata)
        preflight = analysis_payload(
            bundle if block == "A" else None,
            bundle if block == "B" else None,
        )
        if preflight["decision"].get("status") != "DEVICE_EVIDENCE_PENDING":
            raise RuntimeError("single-block analysis preflight did not remain pending")
        report["single_block_analysis_preflight"] = "PASS_DEVICE_EVIDENCE_PENDING"
        if block == "B":
            a_out = REMOTE_BASE / f"evidence/{runtime_evidence_name('A')}"
            a_bundle = load_evidence_bundle(
                a_out / "round16-a1-controller/round16-a1-raw-receipt.json",
                a_out / "round16-a1-observer-metadata.json",
            )
            paths = write_analysis_outputs(
                out / "round16-a1-analysis", a_bundle, bundle
            )
            decision = json.loads(paths["candidate_decision_json"].read_text())
            if (
                decision.get("status") != "A1_INSUFFICIENT_ATTRIBUTION"
                or decision.get("selected_candidate") is not None
            ):
                raise RuntimeError(
                    "incomplete direct attribution did not preserve the evidence gap"
                )
            report["two_block_candidate_decision"] = "A1_INSUFFICIENT_ATTRIBUTION"
        report["status"] = "PASS_FRESH_A1_BLOCK_RAW_UNREVIEWED"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as normalization_error:
        report.update(
            status="FAIL_ANALYSIS_INPUT_NORMALIZATION",
            normalization_error=repr(normalization_error),
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    archive = _archive_manifest(out)
    (out / "round16-a1-archive-sha256.json").write_text(json.dumps(archive, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    args = parser.parse_args()
    run_block(
        block=args.block,
        snapshot=args.snapshot.resolve(),
        source_manifest_sha256=args.source_manifest_sha256,
        admission=args.admission.resolve(),
        admission_sha256=args.admission_sha256,
        ledger=args.ledger.resolve(),
        ledger_sha256=args.ledger_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
