#!/usr/bin/env python3
"""Recompute the Round11 final evidence, budget, test, and cleanup gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts" / "round11"
ARCHIVE_SHA256 = {
    147: "0834b7139b8480fbf083f16bdfb9491964371aa93e6efe2945e5cc33dc24131c",
    148: "26d65cf039b6082c5975be30905bf71470aea8480a18577e5df618f1457604c5",
    149: "ec80137516febc416db5fbebaf1daf5faa56fa9216aa53b2626e1f4da18cf130",
    150: "257e7bbc4348fb0fbd80c5df573d6c2dc223ffeace883ae71aac2254f94f2754",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def run(command, environment=None):
    process = subprocess.run(
        command, cwd=REPO, text=True, capture_output=True, env=environment
    )
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def verify():
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    checks = {
        "round11_tests": run([
            sys.executable, "-m", "unittest", "discover", "-s",
            "artifacts/round11/tests", "-p", "test_*.py",
        ], environment),
        "core_tests": run([
            sys.executable, "-m", "unittest", "discover", "-s", "tests",
            "-p", "test_*.py",
        ], environment),
    }
    qsa_environment = dict(environment)
    qsa_environment["PYTHONPATH"] = "/tmp/round11-torch-cpu"
    qsa_python = (
        "/home/admin/.cache/codex-runtimes/codex-primary-runtime/"
        "dependencies/python/bin/python3"
    )
    checks["qsa_tests"] = run([
        qsa_python, "-m", "unittest",
        "migration_bridge.round9.tests.qsa_module_tests",
        "migration_bridge.round9.tests.round11_t5_qsa_component_probe_tests",
    ], qsa_environment)
    compile_environment = dict(environment)
    compile_environment["PYTHONPYCACHEPREFIX"] = "/tmp/round11-final-pycache"
    checks["py_compile"] = run([
        sys.executable, "-m", "py_compile",
        "artifacts/round11/analyze_round11_t6.py",
        "artifacts/round11/round11_t6_driver.py",
        "artifacts/round11/build_round11_t6_snapshot.py",
        "artifacts/round11/analyze_round11_t5.py",
        "artifacts/round11/analyze_round11_t5_qsa_component.py",
        "artifacts/round11/round11_t5_qsa_model_driver.py",
        "artifacts/round11/build_round11_t5_qsa_model.py",
        "artifacts/round11/verify_round11_final.py",
    ], compile_environment)

    with tempfile.NamedTemporaryFile(
            prefix="round11-t6-", suffix=".json", delete=False) as handle:
        replay_path = Path(handle.name)
    try:
        command = [sys.executable, "artifacts/round11/analyze_round11_t6.py"]
        for version in ARCHIVE_SHA256:
            command += [
                "--archive",
                (f"artifacts/round11/remote-evidence/"
                 f"20260926-round11-t6-v{version}-evidence.tar.gz"),
            ]
        command += ["--output", str(replay_path)]
        checks["t6_replay"] = run(command, environment)
        checks["t6_replay"]["byte_exact"] = (
            replay_path.read_bytes()
            == (ROOT / "ROUND11_T6_FINAL_ANALYSIS.json").read_bytes()
        )
    finally:
        replay_path.unlink(missing_ok=True)

    for name, row in checks.items():
        if row["exit_code"] != 0:
            raise RuntimeError(f"{name} failed: {row}")
    if not checks["t6_replay"]["byte_exact"]:
        raise RuntimeError("T6 replay drift")

    status = load(ROOT / "ROUND11_FINAL_STATUS.json")
    budget = load(ROOT / "ROUND11_EXECUTION_BUDGET.json")
    t5 = load(ROOT / "ROUND11_T5_FINAL_DISPOSITION.json")
    t6 = load(ROOT / "ROUND11_T6_FINAL_ANALYSIS.json")
    retirement = load(ROOT / "ROUND11_RESOURCE_RETIREMENT_RECEIPT.json")
    remote = load(ROOT / "ROUND11_FINAL_REMOTE_STATE.json")
    if status["status"] != "ROUND11_HYBRID_PERFORMANCE_AND_BACKEND_QUALIFIED":
        raise RuntimeError("Round11 final status mismatch")
    if budget["status"] != "COMPLETE_ROUND11_HYBRID_PERFORMANCE_AND_BACKEND_QUALIFIED":
        raise RuntimeError("Round11 budget status mismatch")
    if t5["status"] != "COMPLETE_QSA_COMPONENT_ONLY_NO_PROMOTION":
        raise RuntimeError("T5 disposition mismatch")
    if t6["status"] != "PASS_ROUND11_T6_FINAL_INTEGRATION":
        raise RuntimeError("T6 status mismatch")
    if retirement["status"] != "PASS_ROUND11_RESOURCE_RETIREMENT":
        raise RuntimeError("retirement status mismatch")
    if remote["status"] != "PASS_FINAL_REMOTE_CLEANUP":
        raise RuntimeError("remote cleanup status mismatch")

    stages = budget["stages"]
    global_budget = budget["global"]
    model_starts = sum(stages[name]["model_starts_used"]
                       for name in ("T1", "T3", "T4", "T5", "T6"))
    if model_starts != global_budget["model_starts_used"] or model_starts != 19:
        raise RuntimeError("model-start budget does not add up")
    generations = sum(stages[name]["generation_used"]
                      for name in ("T1", "T3", "T4", "T5", "T6"))
    if generations != global_budget["generation_used"] or generations != 80:
        raise RuntimeError("generation budget does not add up")
    aborts = sum(stages[name]["abort_controls_used"]
                 for name in ("T1", "T4", "T5", "T6"))
    if aborts != global_budget["abort_controls_used"] or aborts != 4:
        raise RuntimeError("abort budget does not add up")
    if global_budget["automatic_retries_used"] != 0:
        raise RuntimeError("unexpected automatic retry")

    for evidence in status["evidence"].values():
        path = REPO / evidence["path"]
        if not path.is_file() or sha256(path) != evidence["sha256"]:
            raise RuntimeError(f"final evidence hash mismatch: {path}")
    for version, expected in ARCHIVE_SHA256.items():
        path = (ROOT / "remote-evidence"
                / f"20260926-round11-t6-v{version}-evidence.tar.gz")
        if sha256(path) != expected:
            raise RuntimeError(f"T6 archive hash mismatch: {version}")
    for row in retirement["blocks"]:
        if not (row["status"] == "PASS_CACHE_RETIRED"
                and row["resident_pages_after"] == 0
                and row["active_reference_count"] == 0
                and row["stat_identity_unchanged"]
                and not row["global_drop_caches"]):
            raise RuntimeError(f"cache retirement mismatch: {row['block']}")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if head != "b41e901c059222b8c1e4367abe193d7f58cc1d83":
        raise RuntimeError("repository HEAD drift")
    if branch != "codex/round11-hybrid-performance-backend":
        raise RuntimeError("repository branch drift")
    report = (REPO / "docs/reports/2026-09-26-round11-final-zh.md").read_text()
    current = (ROOT / "ROUND11_CURRENT_EXECUTION.md").read_text()
    terminal = "ROUND11_HYBRID_PERFORMANCE_AND_BACKEND_QUALIFIED"
    if terminal not in report or terminal not in current:
        raise RuntimeError("final prose status mismatch")

    for row in checks.values():
        text = row.pop("stdout") + "\n" + row.pop("stderr")
        match = re.search(r"Ran (\d+) tests", text)
        row["test_count"] = int(match.group(1)) if match else None
    return {
        "schema": 1,
        "status": "PASS_ROUND11_FINAL_VERIFICATION",
        "verified_at": time.time(),
        "repository": {"head": head, "branch": branch},
        "checks": checks,
        "budget": global_budget,
        "artifact_statuses": {
            "final": status["status"],
            "t5": t5["status"],
            "t6": t6["status"],
            "retirement": retirement["status"],
            "remote_cleanup": remote["status"],
        },
        "archive_sha256": {
            str(version): value for version, value in ARCHIVE_SHA256.items()
        },
        "evidence_hashes_recomputed": len(status["evidence"]),
        "dirty_worktree_preserved": True,
    }


def main():
    result = verify()
    output = ROOT / "ROUND11_FINAL_VERIFICATION.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    counts = {name: row.get("test_count") for name, row in result["checks"].items()}
    print(result["status"], counts)


if __name__ == "__main__":
    main()
