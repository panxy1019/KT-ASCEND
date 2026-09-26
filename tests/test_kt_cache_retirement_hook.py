import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "round11" / "kt_cache_retirement_hook.py"
SPEC = importlib.util.spec_from_file_location("kt_cache_retirement_hook", MODULE_PATH)
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


class FakeDependencies:
    def __init__(
        self,
        resident_before=None,
        resident_after=None,
        references=None,
        complete=True,
        activity=None,
    ):
        self.resident_before = resident_before or {}
        self.resident_after = resident_after or {}
        self.references = references or []
        self.complete = complete
        self.fadvise_paths = []
        self.resident_calls = 0
        self.numa_calls = 0
        self.activity_calls = 0
        self.audit_calls = 0
        self.activity = activity or hook.ActivityResult(True, [], [], [])

    def project_activity(self, _contract):
        self.activity_calls += 1
        return self.activity

    def audit(self, records):
        self.audit_calls += 1
        return hook.AuditResult(self.complete, self.references, [])

    def resident_pages(self, path):
        values = self.resident_before if self.resident_calls < len(self.resident_before) else self.resident_after
        self.resident_calls += 1
        return values[str(path)]

    def fadvise(self, record):
        self.fadvise_paths.append(str(record.resolved_path))

    def numa_memfree(self):
        self.numa_calls += 1
        return {"2": 1000 + self.numa_calls}

    def settle(self):
        pass


class CacheRetirementHookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.bank_root = self.base / "banks"
        self.checkpoint_root = self.base / "checkpoint"
        self.bank_root.mkdir()
        self.checkpoint_root.mkdir()
        self.bank = self.bank_root / "bank.gguf"
        self.checkpoint = self.checkpoint_root / "model.safetensors"
        self.bank.write_bytes(b"bank")
        self.checkpoint.write_bytes(b"checkpoint")
        self.bank.chmod(0o644)
        self.checkpoint.chmod(0o644)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, path, scope):
        value = path.stat()
        return {
            "scope": scope,
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "sha256": "a" * 64,
            "full_file_sha256_verified": True,
            "read_only_registered": True,
            "stat": {
                "device": value.st_dev,
                "inode": value.st_ino,
                "size": value.st_size,
                "mtime_ns": value.st_mtime_ns,
            },
        }

    def manifest(self, files=None):
        return {
            "schema": 1,
            "experiment_block_complete": True,
            "project_containers_running": 0,
            "project_processes_running": 0,
            "read_only_use": True,
            "hash_verification": {"status": "PASS", "complete": True, "full_file": True},
            "bank_root": str(self.bank_root),
            "checkpoint": {"identity": "checkpoint-v1", "root": str(self.checkpoint_root)},
            "project_activity": {
                "container_names": ["kt-round11-owned"],
                "container_id_prefixes": ["abc123"],
                "process_markers": ["kt-round11-owned-marker"],
            },
            "files": files or [self.record(self.bank, "bank"), self.record(self.checkpoint, "checkpoint")],
        }

    def run_hook(self, manifest=None, deps=None, dry_run=False):
        deps = deps or FakeDependencies(
            {str(self.bank): 2, str(self.checkpoint): 3},
            {str(self.bank): 0, str(self.checkpoint): 0},
        )
        return hook.retire_project_cache(
            manifest or self.manifest(),
            deps=deps,
            expected_bank_root=self.bank_root,
            expected_checkpoint_root=self.checkpoint_root,
            expected_checkpoint_identity="checkpoint-v1",
            dry_run=dry_run,
        )

    def test_exact_registered_file_set_retires_and_preserves_identity(self):
        deps = FakeDependencies(
            {str(self.bank): 2, str(self.checkpoint): 3},
            {str(self.bank): 0, str(self.checkpoint): 0},
        )
        result = self.run_hook(deps=deps)
        self.assertEqual(result["status"], "PASS_CACHE_RETIRED")
        self.assertEqual(result["file_count"], 2)
        self.assertTrue(result["stat_identity_unchanged"])
        self.assertEqual(result["resident_pages_after"], {str(self.bank): 0, str(self.checkpoint): 0})
        identities = {row["alias_path"]: row for row in result["registered_stat_identity"]}
        self.assertEqual(identities[str(self.bank)]["inode"], self.bank.stat().st_ino)
        self.assertEqual(deps.activity_calls, 3)
        self.assertEqual(deps.audit_calls, 3)

    def test_rejects_extension_and_root_escape(self):
        wrong_extension = self.bank_root / "bank.bin"
        wrong_extension.write_bytes(b"x")
        wrong_extension.chmod(0o444)
        with self.assertRaisesRegex(hook.RetirementRefused, "extension"):
            self.run_hook(self.manifest([self.record(wrong_extension, "bank")]))
        outside = self.base / "outside.gguf"
        outside.write_bytes(b"x")
        outside.chmod(0o444)
        with self.assertRaisesRegex(hook.RetirementRefused, "root"):
            self.run_hook(self.manifest([self.record(outside, "bank")]))

    def test_registered_bank_symlink_alias_inside_root_is_allowed_and_reported(self):
        backing = self.bank_root / "objects" / "bank.gguf"
        backing.parent.mkdir()
        backing.write_bytes(b"backing")
        link = self.bank_root / "alias.gguf"
        link.symlink_to(backing)
        deps = FakeDependencies(
            {str(backing): 2, str(self.checkpoint): 3},
            {str(backing): 0, str(self.checkpoint): 0},
        )
        result = self.run_hook(
            self.manifest([self.record(link, "bank"), self.record(self.checkpoint, "checkpoint")]),
            deps=deps,
        )
        bank_row = next(row for row in result["registered_stat_identity"] if row["scope"] == "bank")
        self.assertEqual(bank_row["alias_path"], str(link))
        self.assertEqual(bank_row["resolved_path"], str(backing))
        self.assertIn(str(backing), deps.fadvise_paths)

    def test_rejects_bank_alias_escape_or_unregistered_resolved_target(self):
        outside = self.base / "outside.gguf"
        outside.write_bytes(b"outside")
        link = self.bank_root / "alias.gguf"
        link.symlink_to(outside)
        with self.assertRaisesRegex(hook.RetirementRefused, "root"):
            self.run_hook(self.manifest([self.record(link, "bank"), self.record(self.checkpoint, "checkpoint")]))
        link.unlink()
        link.symlink_to(self.bank)
        row = self.record(link, "bank")
        row.pop("resolved_path")
        with self.assertRaisesRegex(hook.RetirementRefused, "resolved_path"):
            self.run_hook(self.manifest([row, self.record(self.checkpoint, "checkpoint")]))

    def test_checkpoint_symlink_remains_rejected(self):
        alias = self.checkpoint_root / "alias.safetensors"
        alias.symlink_to(self.checkpoint)
        with self.assertRaisesRegex(hook.RetirementRefused, "checkpoint symlink"):
            self.run_hook(self.manifest([self.record(self.bank, "bank"), self.record(alias, "checkpoint")]))

    def test_read_only_contract_flags_are_required_but_mode_0644_is_allowed(self):
        self.assertEqual(self.bank.stat().st_mode & 0o777, 0o644)
        self.assertEqual(self.run_hook()["status"], "PASS_CACHE_RETIRED")
        manifest = self.manifest()
        manifest["read_only_use"] = False
        with self.assertRaisesRegex(hook.RetirementRefused, "read-only use"):
            self.run_hook(manifest)
        manifest = self.manifest()
        manifest["files"][0]["read_only_registered"] = False
        with self.assertRaisesRegex(hook.RetirementRefused, "read-only registration"):
            self.run_hook(manifest)

    def test_checkpoint_root_and_identity_must_match_expected_values(self):
        with self.assertRaisesRegex(hook.RetirementRefused, "checkpoint root"):
            hook.validate_manifest(
                self.manifest(),
                expected_bank_root=self.bank_root,
                expected_checkpoint_root=self.base,
                expected_checkpoint_identity="checkpoint-v1",
            )
        with self.assertRaisesRegex(hook.RetirementRefused, "checkpoint identity"):
            hook.validate_manifest(
                self.manifest(),
                expected_bank_root=self.bank_root,
                expected_checkpoint_root=self.checkpoint_root,
                expected_checkpoint_identity="wrong",
            )

    def test_live_project_activity_or_activity_audit_gap_refuses_retirement(self):
        for activity, message in (
            (hook.ActivityResult(True, [{"name": "kt-round11-owned"}], [], []), "container"),
            (hook.ActivityResult(True, [], [{"pid": 42}], []), "process"),
            (hook.ActivityResult(False, [], [], [{"kind": "docker"}]), "incomplete"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(hook.RetirementRefused, message):
                self.run_hook(deps=FakeDependencies(activity=activity))

    def test_rejects_active_fd_or_map_files_reference(self):
        for kind in ("fd", "map_file"):
            deps = FakeDependencies(references=[{"pid": 7, "kind": kind, "device": 1, "inode": 2}])
            with self.subTest(kind=kind), self.assertRaisesRegex(hook.RetirementRefused, "active reference"):
                self.run_hook(deps=deps)

    def test_rejects_incomplete_reference_audit(self):
        deps = FakeDependencies(complete=False)
        with self.assertRaisesRegex(hook.RetirementRefused, "incomplete"):
            self.run_hook(deps=deps)

    def test_rejects_incomplete_hash_or_running_project_resources(self):
        for field, value, message in (
            ("hash_verification", {"status": "PASS", "complete": False, "full_file": True}, "hash"),
            ("project_processes_running", 1, "process"),
            ("project_containers_running", 1, "container"),
        ):
            manifest = self.manifest()
            manifest[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(hook.RetirementRefused, message):
                self.run_hook(manifest)

    def test_fails_when_any_resident_pages_remain(self):
        deps = FakeDependencies(
            {str(self.bank): 2, str(self.checkpoint): 3},
            {str(self.bank): 0, str(self.checkpoint): 1},
        )
        with self.assertRaisesRegex(hook.RetirementFailed, "resident"):
            self.run_hook(deps=deps)
        self.assertEqual(deps.numa_calls, 2)

    def test_zero_residency_is_a_noop_and_does_not_call_fadvise(self):
        deps = FakeDependencies(
            {str(self.bank): 0, str(self.checkpoint): 0},
            {str(self.bank): 0, str(self.checkpoint): 0},
        )
        result = self.run_hook(deps=deps)
        self.assertEqual(result["status"], "NOOP_ZERO_RESIDENCY")
        self.assertEqual(deps.fadvise_paths, [])

    def test_dry_run_audits_without_pageout(self):
        deps = FakeDependencies(
            {str(self.bank): 2, str(self.checkpoint): 3},
            {str(self.bank): 0, str(self.checkpoint): 0},
        )
        result = self.run_hook(deps=deps, dry_run=True)
        self.assertEqual(result["status"], "DRY_RUN_READY")
        self.assertEqual(deps.fadvise_paths, [])

    def test_admission_retry_uses_one_new_identity_and_never_loops(self):
        admissions = []

        def admit(identity):
            admissions.append(identity)
            return {"status": "RESOURCE_INSUFFICIENT", "identity": identity}

        result = hook.retry_admission_after_cache_retirement(
            {"status": "RESOURCE_INSUFFICIENT", "identity": "old", "fresh_admission": True, "project_file_cache_cause": True},
            lambda: {"status": "PASS_CACHE_RETIRED"},
            admit,
            lambda: "new",
        )
        self.assertEqual(admissions, ["new"])
        self.assertEqual(result["retry_kind"], "admission")
        self.assertEqual(result["model_component_retry_increment"], 0)

    def test_admission_is_not_retried_after_zero_residency_noop(self):
        called = []
        result = hook.retry_admission_after_cache_retirement(
            {"status": "RESOURCE_INSUFFICIENT", "identity": "old", "fresh_admission": True, "project_file_cache_cause": True},
            lambda: {"status": "NOOP_ZERO_RESIDENCY"},
            lambda identity: called.append(identity),
            lambda: "new",
        )
        self.assertEqual(called, [])
        self.assertEqual(result["status"], "NO_RETRY_ZERO_RESIDENCY")

    def test_stale_admission_cannot_trigger_retirement_or_retry(self):
        called = []
        result = hook.retry_admission_after_cache_retirement(
            {"status": "RESOURCE_INSUFFICIENT", "identity": "old", "project_file_cache_cause": True},
            lambda: called.append("retire"),
            lambda identity: called.append(identity),
            lambda: "new",
        )
        self.assertEqual(called, [])
        self.assertEqual(result["status"], "NO_RETRY_NOT_FRESH")


class ProcAuditTests(unittest.TestCase):
    def test_default_activity_audit_matches_exact_container_selectors_and_process_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            process = proc_root / "42"
            process.mkdir()
            (process / "cmdline").write_bytes(b"python\0kt-round11-owned-marker\0")

            def docker_ps(_args, **_kwargs):
                return "abc123ffff\tother-name\nffffffff\tkt-round11-owned\n"

            result = hook.audit_project_activity(
                {
                    "container_names": ["kt-round11-owned"],
                    "container_id_prefixes": ["abc123"],
                    "process_markers": ["kt-round11-owned-marker"],
                },
                proc_root=proc_root,
                command=docker_ps,
            )
            self.assertTrue(result.complete)
            self.assertEqual({row["id"] for row in result.containers}, {"abc123ffff", "ffffffff"})
            self.assertEqual([row["pid"] for row in result.processes], [42])

    def test_fadvise_refuses_an_opened_inode_that_differs_from_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.gguf"
            path.write_bytes(b"old")
            value = path.stat()
            record = hook.TargetRecord("bank", path, path, value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
            path.unlink()
            path.write_bytes(b"new")
            with self.assertRaisesRegex(hook.RetirementRefused, "opened file identity"):
                hook.advise_dontneed(record)

    def test_real_proc_scanner_finds_fd_by_device_and_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "weight.gguf"
            target.write_bytes(b"weight")
            value = target.stat()
            record = hook.TargetRecord("bank", target, target.resolve(), value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
            with target.open("rb"):
                result = hook.audit_proc_references([record], pids=[os.getpid()])
            self.assertTrue(any(row["kind"] == "fd" for row in result.references))

    def test_proc_scanner_finds_map_files_identity_in_fake_proc(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "weight.gguf"
            target.write_bytes(b"weight")
            value = target.stat()
            proc = base / "proc" / "42" / "map_files"
            (base / "proc" / "42" / "fd").mkdir(parents=True)
            proc.mkdir()
            (proc / "100-200").symlink_to(target)
            record = hook.TargetRecord("bank", target, target.resolve(), value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
            result = hook.audit_proc_references([record], proc_root=base / "proc")
            self.assertTrue(any(row["kind"] == "map_file" for row in result.references))

    def test_proc_scanner_marks_permission_gap_incomplete(self):
        target = hook.TargetRecord("bank", Path("/x.gguf"), Path("/x.gguf"), 1, 2, 3, 4)

        def denied(_path):
            raise PermissionError("denied")

        result = hook.audit_proc_references([target], pids=[99], scandir=denied)
        self.assertFalse(result.complete)
        self.assertEqual({row["kind"] for row in result.errors}, {"fd", "map_file"})


if __name__ == "__main__":
    unittest.main()
