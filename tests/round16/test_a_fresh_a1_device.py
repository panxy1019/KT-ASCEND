from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from round16.a.fresh_a1_device import (
    DeviceDriverError,
    SOURCE_IDENTITY_KEYS,
    build_request,
    drive_schedule_item,
)
from round16.a.fresh_a1_profile import BLOCK_IDS, manifest_payload
from round16.a.fresh_a1_remote import (
    RemoteControlError,
    V295_REMOTE,
    block_runtime_contract,
    validate_admission,
)
from tools.round16_a1_remote_block import (
    _archive_manifest,
    _terminate_bound_guard,
    archive_admission_evidence,
    cleanup_owned_container,
    record_nonblocking_post_cleanup_probe_failure,
    record_cleanup_failure,
    validate_block_b_predecessor,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeResponse:
    def __init__(self, rows=None, body=None):
        self.rows = rows or []
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.rows)

    def read(self):
        return json.dumps(self.body).encode()


class FreshA1DeviceTests(unittest.TestCase):
    def _admission(self, root: Path) -> tuple[Path, dict]:
        names = (
            "RESUME02_ADMISSION_ROUND1.json",
            "RESUME02_ADMISSION_ROUND2.json",
        )
        raw = (
            {"started_at": 10.0, "ended_at": 70.0},
            {"started_at": 71.0, "ended_at": 131.0},
        )
        hashes = {}
        for name, payload in zip(names, raw, strict=True):
            path = root / name
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            hashes[name] = _sha(path)
        payload = {
            "schema": 1,
            "status": "PASS",
            "started_at": 10.0,
            "ended_at": 131.0,
            "validated_at_epoch": 132.0,
            "target": {
                "physical_id": 12,
                "card_id": 6,
                "chip_id": 0,
                "pci_bdf": "0000:85:00.0",
                "numa_nodes": [2, 3, 4, 7],
            },
            "rounds": list(names),
            "raw_sha256": hashes,
        }
        path = root / "admission.json"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        return path, payload

    def _fixture(self, root: Path) -> Path:
        cases = {
            "edge128": {"input_ids": list(range(128)), "input_tokens": 128},
            "canary_math391": {
                "input_ids": list(range(27)),
                "input_tokens": 27,
                "input_ids_sha256": "4d2584778859d8b34fa388d6fb09925c1da1a0197945de1298f09d199dce489d",
                "messages_sha256": "e5a27eaad7efa96e5ff9020ca99f901cf968238ac7ae7b662c7369923c5e6615",
                "validator": {"kind": "exact_text", "expected": "391"},
            },
        }
        path = root / "round10-t1-fixture.json"
        path.write_text(json.dumps({"model_revision": "a" * 40, "cases": cases}))
        return path

    def test_build_request_matches_strict_six_item_schedule(self):
        with tempfile.TemporaryDirectory() as item:
            fixture = json.loads(self._fixture(Path(item)).read_text())
            schedule = manifest_payload()["schedule"]
            requests = [
                build_request(row, request_id=f"r{i}", fixture=fixture)
                for i, row in enumerate(schedule, 1)
            ]
            self.assertEqual([len(row["input_ids"]) for row in requests], [128, 128, 128, 128, 27, 27])
            self.assertEqual([row["sampling_params"]["max_new_tokens"] for row in requests], [16, 128, 16, 128, 4, 32])
            self.assertEqual([row["sampling_params"]["ignore_eos"] for row in requests], [True, True, True, True, True, False])
            self.assertEqual([row["stream"] for row in requests], [True, True, True, True, True, False])

    def test_driver_uses_one_post_and_returns_exact_identity_without_retry(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            fixture = self._fixture(root)
            row = manifest_payload()["schedule"][0]
            calls = []

            def post(endpoint, body, timeout):
                calls.append((endpoint, body, timeout))
                return FakeResponse()

            receipt = drive_schedule_item(
                row,
                request_id=f"{BLOCK_IDS['A']}:1:W1_WARMUP",
                fixture_path=fixture,
                block_id=BLOCK_IDS["A"],
                post_fn=post,
                consume_stream_fn=lambda response, count: {
                    "completion_tokens": count,
                    "finite_logprobs": True,
                    "finish_reason": "length",
                    "output_ids": list(range(count)),
                },
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipt["status"], "COMPLETE")
            self.assertEqual(receipt["observed_request"]["input_tokens"], 128)
            self.assertEqual(set(receipt["source_identity"]), set(SOURCE_IDENTITY_KEYS))

    def test_driver_rejects_schedule_drift_before_post(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            fixture = self._fixture(root)
            row = dict(manifest_payload()["schedule"][0])
            row["output_tokens"] = 17
            calls = []
            with self.assertRaisesRegex(DeviceDriverError, "schedule"):
                drive_schedule_item(
                    row,
                    request_id="forged",
                    fixture_path=fixture,
                    block_id=BLOCK_IDS["A"],
                    post_fn=lambda *args: calls.append(args),
                    consume_stream_fn=lambda *args: {},
                )
            self.assertEqual(calls, [])

    def test_runtime_contracts_are_distinct_and_zero_retry(self):
        a = block_runtime_contract("A")
        b = block_runtime_contract("B")
        self.assertNotEqual(a["container"], b["container"])
        self.assertNotEqual(a["evidence"], b["evidence"])
        self.assertEqual(a["generation_limit"], 6)
        self.assertEqual(a["automatic_retries"], 0)
        self.assertEqual(a["profile_environment"]["ROUND16_A1_PROFILE_ENABLE"], "1")
        self.assertEqual(
            str(V295_REMOTE),
            "/mnt/data1/kt-round9/round14a-a-gdn-model-20260927-v2/v295",
        )

    def test_admission_requires_exact_hash_pass_and_physical12(self):
        with tempfile.TemporaryDirectory() as item:
            path, _ = self._admission(Path(item))
            with self.assertRaisesRegex(RemoteControlError, "SHA"):
                validate_admission(path, expected_sha256="0" * 64)
            self.assertEqual(
                validate_admission(path, expected_sha256=_sha(path), now=133.0)["status"],
                "PASS",
            )

    def test_admission_recomputes_raw_round_hash_duration_order_and_wrapper_times(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            path, payload = self._admission(root)
            raw1 = root / payload["rounds"][0]
            raw1.write_text(json.dumps({"started_at": 10.0, "ended_at": 69.0}) + "\n")
            payload["raw_sha256"][raw1.name] = _sha(raw1)
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RemoteControlError, "60 seconds"):
                validate_admission(path, expected_sha256=_sha(path), now=133.0)

            path, payload = self._admission(root)
            payload["started_at"] = 100.0
            payload["ended_at"] = 160.0
            payload["validated_at_epoch"] = 161.0
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RemoteControlError, "wrap.*raw rounds"):
                validate_admission(path, expected_sha256=_sha(path), now=162.0)

            path, payload = self._admission(root)
            payload["rounds"].reverse()
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RemoteControlError, "round order"):
                validate_admission(path, expected_sha256=_sha(path), now=133.0)

            path, payload = self._admission(root)
            payload["validated_at_epoch"] = 10_000.0
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RemoteControlError, "raw rounds.*stale"):
                validate_admission(path, expected_sha256=_sha(path), now=10_001.0)

    def test_admission_evidence_copies_result_and_both_sha_bound_raw_rounds(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            admission, payload = self._admission(root)
            evidence = root / "evidence"
            evidence.mkdir()

            manifest = archive_admission_evidence(
                admission, evidence, expected_sha256=_sha(admission)
            )

            self.assertEqual(manifest["status"], "FROZEN_LAUNCHER_ADMISSION")
            self.assertEqual(
                set(manifest["files"]),
                {"result.json", *payload["rounds"]},
            )
            for name, identity in manifest["files"].items():
                copied = evidence / "launcher-admission" / name
                self.assertEqual(identity["sha256"], _sha(copied))
                self.assertEqual(identity["size_bytes"], copied.stat().st_size)

    def test_admission_archive_rejects_raw_drift_after_wrapper_review(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            admission, payload = self._admission(root)
            reviewed_sha = _sha(admission)
            raw = root / payload["rounds"][0]
            raw.write_text(json.dumps({"started_at": 10.0, "ended_at": 80.0}) + "\n")
            evidence = root / "evidence"
            evidence.mkdir()

            with self.assertRaisesRegex(RuntimeError, "raw round.*SHA"):
                archive_admission_evidence(
                    admission, evidence, expected_sha256=reviewed_sha
                )

    def test_archive_manifest_hashes_large_logs_and_admission_evidence(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
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
                "launcher-admission/RESUME02_ADMISSION_ROUND1.json",
                "launcher-admission/RESUME02_ADMISSION_ROUND2.json",
                "launcher-admission/manifest.json",
            )
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative + "\n")
            (root / "launcher-admission/manifest.json").write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "status": "FROZEN_LAUNCHER_ADMISSION",
                        "rounds": [
                            "RESUME02_ADMISSION_ROUND1.json",
                            "RESUME02_ADMISSION_ROUND2.json",
                        ],
                    }
                )
                + "\n"
            )
            for name, content in (
                ("server.log", "server\n"),
                ("guard.log", "guard\n"),
                ("round16-a1-driver.log", "driver\n"),
            ):
                (root / name).write_text(content)

            manifest = _archive_manifest(root)

            self.assertNotIn("excluded_large_logs", manifest)
            self.assertEqual(
                set(manifest["large_logs"]),
                {"server.log", "guard.log", "round16-a1-driver.log"},
            )
            for name, identity in manifest["large_logs"].items():
                self.assertEqual(identity["sha256"], _sha(root / name))
                self.assertEqual(identity["size_bytes"], (root / name).stat().st_size)

    def test_block_b_predecessor_binds_fixed_a_receipt_to_exact_ledger(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            schedule = manifest_payload()["schedule"]
            request_ids = [
                f"{BLOCK_IDS['A']}:{row['ordinal']}:{row['role']}"
                for row in schedule
            ]
            ledger = {
                "used_by_block": {BLOCK_IDS["A"]: 6, BLOCK_IDS["B"]: 0},
                "request_ids": request_ids,
            }
            controller = root / "round16-a1-controller"
            controller.mkdir()
            ledger_path = controller / "round16-a1-ledger-after.json"
            ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n")
            receipt = {
                "status": "RAW_DEVICE_RESULT_UNREVIEWED",
                "block_id": BLOCK_IDS["A"],
                "generation_attempts": 6,
                "generations": [
                    {
                        "ordinal": row["ordinal"],
                        "role": row["role"],
                        "request_id": request_id,
                        "status": "COMPLETE",
                    }
                    for row, request_id in zip(schedule, request_ids, strict=True)
                ],
                "ledger_sha256_after": _sha(ledger_path),
            }
            receipt_path = controller / "round16-a1-raw-receipt.json"
            receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            supervisor_path = root / "round16-a1-supervisor.json"
            supervisor_path.write_text(
                json.dumps(
                    {
                        "block_id": BLOCK_IDS["A"],
                        "status": "PASS_FRESH_A1_BLOCK_RAW_UNREVIEWED",
                        "owned_container_removed": True,
                        "physical12_after": {"target": 12},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            archive_path = root / "round16-a1-archive-sha256.json"
            archive_path.write_text(
                json.dumps(
                    {
                        "status": "FROZEN_AFTER_GUARD_ACK_AND_CONTAINER_REMOVAL",
                        "files": {
                            "round16-a1-supervisor.json": _sha(supervisor_path),
                            "round16-a1-controller/round16-a1-raw-receipt.json": _sha(receipt_path),
                            "round16-a1-controller/round16-a1-ledger-after.json": _sha(ledger_path),
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            report = validate_block_b_predecessor(
                receipt_path, ledger_path=ledger_path, ledger_sha256=_sha(ledger_path)
            )
            self.assertEqual(report["status"], "PASS_EXACT_BLOCK_A_PREDECESSOR")
            self.assertEqual(report["supervisor_sha256"], _sha(supervisor_path))
            self.assertEqual(report["archive_sha256"], _sha(archive_path))

            receipt["ledger_sha256_after"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RuntimeError, "ledger SHA"):
                validate_block_b_predecessor(
                    receipt_path,
                    ledger_path=ledger_path,
                    ledger_sha256=_sha(ledger_path),
                )

    def test_block_b_predecessor_rejects_incomplete_a_cleanup_even_if_archive_rehashed(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            controller = root / "round16-a1-controller"
            controller.mkdir()
            schedule = manifest_payload()["schedule"]
            request_ids = [
                f"{BLOCK_IDS['A']}:{row['ordinal']}:{row['role']}"
                for row in schedule
            ]
            ledger = controller / "round16-a1-ledger-after.json"
            ledger.write_text(
                json.dumps({"request_ids": request_ids}, sort_keys=True) + "\n"
            )
            raw = controller / "round16-a1-raw-receipt.json"
            raw.write_text(
                json.dumps(
                    {
                        "status": "RAW_DEVICE_RESULT_UNREVIEWED",
                        "block_id": BLOCK_IDS["A"],
                        "generation_attempts": 6,
                        "generations": [
                            {
                                "ordinal": row["ordinal"],
                                "role": row["role"],
                                "request_id": request_id,
                                "status": "COMPLETE",
                            }
                            for row, request_id in zip(schedule, request_ids, strict=True)
                        ],
                        "ledger_sha256_after": _sha(ledger),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            supervisor = root / "round16-a1-supervisor.json"
            supervisor.write_text(
                json.dumps(
                    {
                        "block_id": BLOCK_IDS["A"],
                        "status": "FAIL_CLEANUP",
                        "owned_container_removed": False,
                        "physical12_after": {"target": 12},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            (root / "round16-a1-archive-sha256.json").write_text(
                json.dumps(
                    {
                        "status": "FROZEN_AFTER_GUARD_ACK_AND_CONTAINER_REMOVAL",
                        "files": {
                            "round16-a1-supervisor.json": _sha(supervisor),
                            "round16-a1-controller/round16-a1-raw-receipt.json": _sha(raw),
                            "round16-a1-controller/round16-a1-ledger-after.json": _sha(ledger),
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            with self.assertRaisesRegex(RuntimeError, "cleanup terminal"):
                validate_block_b_predecessor(
                    raw, ledger_path=ledger, ledger_sha256=_sha(ledger)
                )

    def test_terminal_cleanup_uses_original_start_for_existing_guard_ack_then_removes(self):
        events = []
        full_id = "a" * 64

        def run(command, **kwargs):
            events.append(tuple(command))

        def wait(path, *, not_before):
            events.append(("wait", not_before))
            return {"action": "OWNED_EXITED", "container_id": full_id, "time": 15.0}

        result = cleanup_owned_container(
            container="kt-round16-a1-a-v003",
            container_id=full_id,
            monitor_path=Path("monitor.jsonl"),
            supervisor_started_at=10.0,
            inspect_fn=lambda name: {"Id": full_id, "State": {"Running": False, "Status": "exited"}},
            run_fn=run,
            wait_guard_fn=wait,
            clock=lambda: 20.0,
        )

        self.assertEqual(events[0], ("wait", 10.0))
        self.assertEqual(events[1], ("docker", "rm", full_id))
        self.assertFalse(result["was_running_at_cleanup"])

    def test_running_cleanup_stops_waits_for_id_bound_ack_then_removes(self):
        events = []
        full_id = "b" * 64

        def run(command, **kwargs):
            events.append(tuple(command))

        result = cleanup_owned_container(
            container="kt-round16-a1-b-v003",
            container_id=full_id,
            monitor_path=Path("monitor.jsonl"),
            supervisor_started_at=10.0,
            inspect_fn=lambda name: {"Id": full_id, "State": {"Running": True, "Status": "running"}},
            run_fn=run,
            wait_guard_fn=lambda path, *, not_before: {
                "action": "OWNED_EXITED",
                "container_id": full_id,
                "time": not_before,
            },
            clock=lambda: 20.0,
        )

        self.assertEqual(events, [("docker", "stop", "-t", "10", full_id), ("docker", "rm", full_id)])
        self.assertTrue(result["was_running_at_cleanup"])

    def test_created_cleanup_terminates_bound_guard_before_safe_remove(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            events = []
            full_id = "e" * 64
            guard_process = root / "guard-process.json"
            guard_process.write_text(
                json.dumps({"pid": 4321, "container_id": full_id}) + "\n"
            )

            result = cleanup_owned_container(
                container="kt-round16-a1-a-v003",
                container_id=full_id,
                monitor_path=root / "monitor.jsonl",
                guard_process_path=guard_process,
                supervisor_started_at=10.0,
                inspect_fn=lambda name: {
                    "Id": full_id,
                    "State": {"Running": False, "Status": "created", "Pid": 0},
                },
                run_fn=lambda command, **kwargs: events.append(tuple(command)),
                wait_guard_fn=lambda *args, **kwargs: self.fail(
                    "a never-started container cannot produce a terminal ACK"
                ),
                terminate_guard_fn=lambda pid, container_name, monitor: events.append(
                    ("terminate-guard", pid, container_name, str(monitor))
                ),
            )

            self.assertEqual(events[0][0], "terminate-guard")
            self.assertEqual(events[1], ("docker", "rm", full_id))
            self.assertEqual(result["guard_termination"], "TERMINATED_BEFORE_REMOVE")
            self.assertEqual(result["container_state"], "created")

    def test_created_cleanup_removes_after_bound_guard_already_exited(self):
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            events = []
            full_id = "f" * 64
            guard_process = root / "guard-process.json"
            guard_process.write_text(
                json.dumps({"pid": 9876, "container_id": full_id}) + "\n"
            )

            result = cleanup_owned_container(
                container="kt-round16-a1-b-v003",
                container_id=full_id,
                monitor_path=root / "monitor.jsonl",
                guard_process_path=guard_process,
                supervisor_started_at=10.0,
                inspect_fn=lambda name: {
                    "Id": full_id,
                    "State": {"Running": False, "Status": "created", "Pid": 0},
                },
                run_fn=lambda command, **kwargs: events.append(tuple(command)),
                wait_guard_fn=lambda *args, **kwargs: self.fail("no terminal ACK"),
                terminate_guard_fn=lambda pid, container_name, monitor: (
                    "ALREADY_EXITED_BEFORE_CLEANUP"
                ),
            )

            self.assertEqual(events, [("docker", "rm", full_id)])
            self.assertEqual(
                result["guard_termination"], "ALREADY_EXITED_BEFORE_CLEANUP"
            )

    def test_guard_exit_between_exists_and_cmdline_read_is_already_exited(self):
        class MissingCmdline:
            def read_bytes(self):
                raise FileNotFoundError("guard exited")

        class ExitingProcess:
            def __init__(self):
                self.checks = 0

            def exists(self):
                self.checks += 1
                return self.checks == 1

            def __truediv__(self, name):
                self.assert_name = name
                return MissingCmdline()

        process = ExitingProcess()

        result = _terminate_bound_guard(
            1234,
            "kt-round16-a1-a-v003",
            Path("monitor.py"),
            _process=process,
        )

        self.assertEqual(result, "ALREADY_EXITED_BEFORE_CLEANUP")
        self.assertEqual(process.assert_name, "cmdline")

    def test_cleanup_rejects_unbound_ack_and_preserves_primary_failure(self):
        full_id = "c" * 64
        calls = []
        with self.assertRaisesRegex(RuntimeError, "ACK.*owned container ID"):
            cleanup_owned_container(
                container="kt-round16-a1-a-v003",
                container_id=full_id,
                monitor_path=Path("monitor.jsonl"),
                supervisor_started_at=10.0,
                inspect_fn=lambda name: {"Id": full_id, "State": {"Running": False, "Status": "dead"}},
                run_fn=lambda command, **kwargs: calls.append(command),
                wait_guard_fn=lambda path, *, not_before: {"action": "OWNED_EXITED", "container_id": "d" * 64},
            )
        self.assertEqual(calls, [])
        report = {"status": "FAIL", "error": "primary"}
        record_cleanup_failure(report, RuntimeError("cleanup"))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"], "primary")
        self.assertIn("cleanup", report["cleanup_error"])

    def test_post_cleanup_resource_probe_is_diagnostic_after_owned_exit(self):
        report = {
            "status": "DRIVER_COMPLETE_WAITING_CLEANUP",
            "owned_container_removed": True,
        }

        record_nonblocking_post_cleanup_probe_failure(
            report, RuntimeError("NUMA reclaim estimate remains conservative")
        )

        self.assertEqual(report["status"], "DRIVER_COMPLETE_WAITING_CLEANUP")
        self.assertTrue(report["owned_container_removed"])
        self.assertEqual(report["physical12_after"]["target"], 12)
        self.assertFalse(report["physical12_after"]["blocking"])
        self.assertIn("NUMA reclaim", report["post_cleanup_resource_probe_error"])


if __name__ == "__main__":
    unittest.main()
