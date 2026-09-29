from __future__ import annotations

import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from round16.a.fresh_a1_profile import BLOCK_IDS
from round16.a.fresh_a1_snapshot import (
    ControllerInputError,
    PreflightError,
    SnapshotPreconditionError,
    build_snapshot,
    canonical_json_sha256,
    controller_input_template,
    execute_controller,
    initial_ledger_payload,
    offline_preflight,
    prepare_controller_input,
    raw_receipt,
)
from tools.round16_a1_remote_block import run_block


SCHEDULER = """\
def event_loop(self, batch):
    from sglang.srt.layers.round9.round13_attribution import (
        before as round13_before, after as round13_after)
    round13_before(self, batch)
    result = self.run_batch(batch)
    round13_after(self)
    self.process_batch_result(batch, result)
"""

KTEP = """\
def shared(ctx, dispatch_output):
    result = ctx.gpu_method.apply(ctx.gpu_layer, dispatch_output)
    return result

def resident(self, layer, masked_dispatch_output):
    gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
    return gpu_combine_input

def merge(self, output, cpu_output, x):
    if not self._no_cpu_stream:
        device_module.current_stream(x.device).wait_event(self._sync_done_event)
    output = output + cpu_output
    if getattr(self, \"_output_cast_to_input_dtype\", False):
        output = output.to(x.dtype)
    return output
"""

LAUNCHER = """\
import argparse,hashlib,json,pathlib,re,shutil,subprocess,time
continuous=json.loads(a.cost_admission.read_text())
if continuous.get('status')!='PASS' or time.time()-continuous['ended_at']>600:
 raise RuntimeError('continuous admission missing/stale')
for rel,sha in backend_source.items():
 if rel.startswith(('sglang/srt/layers/moe/','sglang/srt/layers/quantization/')):
  if digest(sglang/rel)!=sha:raise RuntimeError('layerwise MoE backend differs from device gate')

name='kt-round14a-a-gdn-abba-candidate-a-v295';out=root/'evidence'/'20260927-round14a-a-gdn-abba-candidate-a-v295'

def start_guard(guard_args, log):
 guard=subprocess.Popen(guard_args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 return guard

def launch(args):
    args+=['-e','ROUND14_GDN_BACKEND_ENABLE=1']
    if budget.get('stage')!='ROUND14A_A_GDN_FULL_MODEL':raise RuntimeError('wrong Round14a GDN model budget')
    return args
"""

ROUND13 = """\
def before(runner, batch):
    return None

def after(runner):
    return None
"""

FIXTURE = """{"schema":1,"cases":{"canary_math391":{"input_tokens":27}}}\n"""
EXECUTION_BUDGET = """{"schema":1,"stage":"ROUND14A_A_GDN_FULL_MODEL"}\n"""
MONITOR = """\
import re
def owned_container_name(name):
    return re.fullmatch(
        r'(?:kt-round9-full-model-v[0-9]+'
        r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297))', name) is not None
if terminal:
    with a.output.open('a') as f:
        f.write(json.dumps({'time':time.time(),'action':'OWNED_EXITED','state':state['State']})+'\\n')
if sample:
    with a.output.open('a') as f:
        f.write(json.dumps({'time':time.time(),'action':'OWNED_EXITED_DURING_SAMPLE'})+'\\n')
if failed:
    with a.output.open('a') as f:f.write(json.dumps(dict(time=time.time(),action='OWNED_EXITED_DURING_SAMPLE',sampling_error=repr(exc)))+'\\n')
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _overlay_tree_digest(root: Path) -> str:
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.relative_to(root) != Path("round9-candidate-overlay.json")
    }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class FreshA1SnapshotTests(unittest.TestCase):
    def _parent(self, root: Path) -> tuple[Path, dict[str, str]]:
        parent = root / "v295"
        files = {
            "overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py": KTEP,
            "overlay-python/sglang/srt/layers/round9/round13_attribution.py": ROUND13,
            "overlay-python/sglang/srt/managers/scheduler.py": SCHEDULER,
            "source/tools/launch_hybrid.py": LAUNCHER,
            "round10-t1-fixture.json": FIXTURE,
            "execution-budget.json": EXECUTION_BUDGET,
            "round11_t4_driver.py": "# frozen test driver\n",
            "round10_t1_protocol.py": "# frozen test protocol\n",
            "m3_cleanup.py": "# frozen test cleanup\n",
            "source/tools/candidate_resources.py": "# frozen test resource gate\n",
            "source/tools/monitor_owned_server.py": MONITOR,
        }
        preconditions: dict[str, str] = {}
        for relative, text in files.items():
            path = parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            preconditions[relative] = _sha(text.encode())
        overlay_root = parent / "overlay-python"
        (overlay_root / "round9-candidate-overlay.json").write_text(
            json.dumps(
                {
                    "hash_scope": "PAYLOAD_EXCLUDING_OVERLAY_RECEIPT_AND_PYCACHE",
                    "baseline_tree_sha256": "f" * 64,
                    "candidate_tree_sha256": _overlay_tree_digest(overlay_root),
                    "modules": {},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return parent, preconditions

    def _build(self, root: Path, block: str = "A") -> tuple[Path, Path]:
        parent, preconditions = self._parent(root)
        output = root / f"snapshot-{block.lower()}"
        build_snapshot(
            parent,
            output,
            block=block,
            preconditions=preconditions,
        )
        return parent, output

    def _write_ledger(self, root: Path) -> Path:
        path = root / "round16-a1-ledger.json"
        path.write_text(
            json.dumps(initial_ledger_payload(), indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def test_builder_copies_then_patches_only_child_under_exact_preconditions(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            parent, output = self._build(root)

            self.assertEqual((parent / "source/tools/launch_hybrid.py").read_text(), LAUNCHER)
            scheduler = (
                output / "overlay-python/sglang/srt/managers/scheduler.py"
            ).read_text()
            ktep = (
                output / "overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py"
            ).read_text()
            launcher = (output / "source/tools/launch_hybrid.py").read_text()
            observer = output / (
                "overlay-python/sglang/srt/layers/round9/round16_a1_observer.py"
            )

            self.assertIn("round16_a1_before(self, batch)", scheduler)
            self.assertIn("round16_a1_after(self)", scheduler)
            self.assertNotIn("observe_npu_call", ktep)
            self.assertIn("observe_host_merge(", ktep)
            self.assertIn("ROUND13_ATTRIBUTION_ENABLE=1", launcher)
            self.assertIn("ROUND13_ATTRIBUTION_SPANS=/out/round13-spans.jsonl", launcher)
            self.assertIn("guard-process.json", launcher)
            self.assertIn("guard.terminate()", launcher)
            monitor = (output / "source/tools/monitor_owned_server.py").read_text()
            tree = ast.parse(monitor)
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "owned_container_name"
            )
            scope = {"re": __import__("re")}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "monitor.py", "exec"), scope)
            self.assertTrue(scope["owned_container_name"]("kt-round16-a1-a-v003"))
            self.assertTrue(scope["owned_container_name"]("kt-round16-a1-b-v003"))
            self.assertFalse(scope["owned_container_name"]("kt-round16-a1-c-v003"))
            self.assertIn("'container_id':state['Id']", monitor)
            self.assertIn("import os", launcher)
            self.assertIn("ROUND16_A1_PROFILE_ENABLE", launcher)
            self.assertIn(BLOCK_IDS["A"], launcher)
            self.assertIn("kt-round16-a1-a-v003", launcher)
            self.assertIn("20260929-round16-a1-fresh-profile-a-v003", launcher)
            self.assertIn("ROUND16_A1_FRESH_PROFILE", launcher)
            self.assertIn("ROUND16_A1_ADMISSION_SHA256", launcher)
            self.assertIn("validated_at_epoch", launcher)
            self.assertIn("require_parent_or_bound_delta", launcher)
            self.assertTrue(observer.is_file())
            self.assertTrue((output / "round16-a1-source-sha256.json").is_file())
            budget = json.loads((output / "execution-budget.json").read_text())
            self.assertEqual(budget["stage"], "ROUND16_A1_FRESH_PROFILE")
            self.assertEqual(budget["model_starts_limit"], 1)
            self.assertEqual(budget["generation_limit"], 6)
            self.assertEqual(budget["automatic_retries_limit"], 0)
            source_manifest = json.loads(
                (output / "round16-a1-source-sha256.json").read_text()
            )
            self.assertNotIn("execution-budget.json", source_manifest["files"])
            self.assertEqual(
                source_manifest["execution_budget_contract"]["initial_sha256"],
                _sha((output / "execution-budget.json").read_bytes()),
            )
            overlay_root = output / "overlay-python"
            overlay_receipt = json.loads(
                (overlay_root / "round9-candidate-overlay.json").read_text()
            )
            self.assertEqual(
                overlay_receipt["candidate_tree_sha256"],
                _overlay_tree_digest(overlay_root),
            )
            self.assertEqual(
                overlay_receipt["round16_a1_profile"]["runtime_version"],
                "v003",
            )
            delta = overlay_receipt["round16_a1_profile"]
            self.assertEqual(
                set(delta["parent_overlay_files"]),
                set(delta["modified_overlay_files"]),
            )
            self.assertEqual(
                delta["parent_overlay_files"][
                    "sglang/srt/layers/moe/kt_ep_wrapper.py"
                ],
                _sha(
                    (
                        parent
                        / "overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py"
                    ).read_bytes()
                ),
            )

            compile(scheduler, "scheduler.py", "exec")
            compile(ktep, "kt_ep_wrapper.py", "exec")
            compile(launcher, "launch_hybrid.py", "exec")
            compile(observer.read_text(), "round16_a1_observer.py", "exec")

    def test_built_snapshot_control_package_imports_in_isolated_remote_shape(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(output),
                "PYTHONNOUSERSITE": "1",
            }

            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "from round16.a.fresh_a1_snapshot import offline_preflight; "
                        "assert 'torch' not in sys.modules; "
                        "assert 'torch_npu' not in sys.modules"
                    ),
                ],
                cwd=root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_builder_rejects_hash_drift_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            parent, preconditions = self._parent(root)
            target = parent / "source/tools/launch_hybrid.py"
            target.write_text(LAUNCHER + "# drift\n", encoding="utf-8")
            output = root / "must-not-exist"

            with self.assertRaisesRegex(SnapshotPreconditionError, "launch_hybrid.py"):
                build_snapshot(
                    parent,
                    output,
                    block="A",
                    preconditions=preconditions,
                )

            self.assertFalse(output.exists())

    def test_remote_block_rejects_source_drift_before_any_launch_call(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            manifest = output / "round16-a1-source-sha256.json"
            reviewed_manifest_sha = _sha(manifest.read_bytes())
            tracked = output / "round16/a/fresh_a1_device.py"
            tracked.write_text(tracked.read_text() + "# drift\n")
            launch = mock.Mock()

            with mock.patch(
                "tools.round16_a1_remote_block.validate_admission",
                return_value={"status": "PASS"},
            ), mock.patch(
                "tools.round16_a1_remote_block.subprocess.run", launch
            ):
                with self.assertRaisesRegex(RuntimeError, "source.*prelaunch"):
                    run_block(
                        block="A",
                        snapshot=output,
                        source_manifest_sha256=reviewed_manifest_sha,
                        admission=root / "admission.json",
                        admission_sha256="a" * 64,
                        ledger=ledger,
                        ledger_sha256=_sha(ledger.read_bytes()),
                    )

            launch.assert_not_called()

    def test_remote_block_b_rejects_mismatched_a_receipt_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root, block="B")
            schedule = controller_input_template("A")["schedule"]
            request_ids = [
                f"{BLOCK_IDS['A']}:{row['ordinal']}:{row['role']}"
                for row in schedule
            ]
            ledger_payload = initial_ledger_payload()
            ledger_payload["used_by_block"][BLOCK_IDS["A"]] = 6
            ledger_payload["request_ids"] = request_ids
            a_controller = root / "a-evidence/round16-a1-controller"
            a_controller.mkdir(parents=True)
            ledger = a_controller / "round16-a1-ledger-after.json"
            ledger.write_text(json.dumps(ledger_payload, indent=2) + "\n")
            a_receipt = a_controller / "round16-a1-raw-receipt.json"
            a_receipt.write_text(
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
                            for row, request_id in zip(
                                schedule, request_ids, strict=True
                            )
                        ],
                        "ledger_sha256_after": "0" * 64,
                    }
                )
                + "\n"
            )
            manifest = output / "round16-a1-source-sha256.json"
            launch = mock.Mock()

            with mock.patch(
                "tools.round16_a1_remote_block.validate_admission",
                return_value={"status": "PASS"},
            ), mock.patch(
                "tools.round16_a1_remote_block.A_BLOCK_RAW_RECEIPT", a_receipt
            ), mock.patch(
                "tools.round16_a1_remote_block.subprocess.run", launch
            ):
                with self.assertRaisesRegex(RuntimeError, "ledger SHA"):
                    run_block(
                        block="B",
                        snapshot=output,
                        source_manifest_sha256=_sha(manifest.read_bytes()),
                        admission=root / "admission.json",
                        admission_sha256="a" * 64,
                        ledger=ledger,
                        ledger_sha256=_sha(ledger.read_bytes()),
                    )

            launch.assert_not_called()

    def test_builder_rejects_output_inside_frozen_parent(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            parent, preconditions = self._parent(root)
            output = parent / "forbidden-child"

            with self.assertRaisesRegex(SnapshotPreconditionError, "outside"):
                build_snapshot(
                    parent,
                    output,
                    block="A",
                    preconditions=preconditions,
                )

            self.assertFalse(output.exists())

    def test_offline_preflight_verifies_manifest_default_off_and_immutable_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)

            report = offline_preflight(output, block_id=BLOCK_IDS["A"], ledger_path=ledger)

            self.assertEqual(report["status"], "PASS_OFFLINE_PREFLIGHT")
            self.assertFalse(report["enabled"])
            self.assertEqual(report["generation_limit"], 6)
            self.assertEqual(report["total_generation_limit"], 12)
            self.assertEqual(report["device_probe_count"], 0)
            self.assertEqual(report["source_manifest_status"], "EXACT")
            self.assertEqual(report["execution_budget_state"], "INITIAL_UNSPENT")
            self.assertEqual(
                report["execution_budget_initial_sha256"],
                report["execution_budget_current_sha256"],
            )

    def test_controller_preflight_accepts_one_matching_model_start_and_rejects_illegal_budget_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            budget_path = output / "execution-budget.json"
            initial_sha = _sha(budget_path.read_bytes())
            base = json.loads(budget_path.read_text())
            legal_start = {
                "mode": "OFF",
                "name": "kt-round16-a1-a-v003",
                "container_id": "a" * 64,
                "time": 123.0,
                "output": "/mnt/data1/kt-round9/evidence/20260929-round16-a1-fresh-profile-a-v003",
            }
            base.update(model_starts_used=1, starts=[legal_start])
            budget_path.write_text(json.dumps(base, indent=2) + "\n")

            plan = prepare_controller_input(
                output,
                block_id=BLOCK_IDS["A"],
                ledger_path=ledger,
                expected_ledger_sha256=_sha(ledger.read_bytes()),
                output_dir=root / "result",
                enabled=True,
            )

            self.assertEqual(plan["execution_budget_state"], "SINGLE_MATCHING_START")
            self.assertEqual(plan["execution_budget_initial_sha256"], initial_sha)
            self.assertNotEqual(plan["execution_budget_current_sha256"], initial_sha)

            illegal_updates = (
                ({"stage": "WRONG"}, "stage"),
                ({"model_starts_used": 2, "starts": [legal_start, legal_start]}, "one"),
                ({"automatic_retries_limit": 1}, "retries"),
                ({"abort_limit": 1}, "abort"),
                (
                    {
                        "model_starts_used": 1,
                        "starts": [{**legal_start, "container_id": "short"}],
                    },
                    "container",
                ),
            )
            for index, (updates, message) in enumerate(illegal_updates):
                with self.subTest(message=message):
                    _, bad_output = self._build(root / f"bad-{index}")
                    bad_budget = bad_output / "execution-budget.json"
                    payload = json.loads(bad_budget.read_text())
                    payload.update(updates)
                    bad_budget.write_text(json.dumps(payload, indent=2) + "\n")
                    with self.assertRaisesRegex(ControllerInputError, message):
                        prepare_controller_input(
                            bad_output,
                            block_id=BLOCK_IDS["A"],
                            ledger_path=ledger,
                            expected_ledger_sha256=_sha(ledger.read_bytes()),
                            output_dir=root / f"bad-result-{index}",
                            enabled=True,
                        )

    def test_preflight_rejects_source_drift_and_existing_raw_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            observer = output / (
                "overlay-python/sglang/srt/layers/round9/round16_a1_observer.py"
            )
            observer.write_text(observer.read_text() + "# drift\n")

            with self.assertRaisesRegex(PreflightError, "source manifest"):
                offline_preflight(output, block_id=BLOCK_IDS["A"], ledger_path=ledger)

            _, clean = self._build(root / "second")
            (clean / "round16-a1-raw-receipt.json").write_text("{}\n")
            with self.assertRaisesRegex(PreflightError, "raw receipt"):
                offline_preflight(clean, block_id=BLOCK_IDS["A"], ledger_path=ledger)

    def test_preflight_rejects_overlay_tree_drift_outside_tracked_file_list(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            untracked = output / "overlay-python/untracked-runtime-file.py"
            untracked.write_text("# source-gate drift\n", encoding="utf-8")

            with self.assertRaisesRegex(PreflightError, "overlay binding"):
                offline_preflight(
                    output,
                    block_id=BLOCK_IDS["A"],
                    ledger_path=ledger,
                )

    def test_controller_requires_explicit_enable_exact_identity_manifest_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            ledger_hash = _sha(ledger.read_bytes())
            result_dir = root / "fresh-result"

            with self.assertRaisesRegex(ControllerInputError, "explicitly enabled"):
                prepare_controller_input(
                    output,
                    block_id=BLOCK_IDS["A"],
                    ledger_path=ledger,
                    expected_ledger_sha256=ledger_hash,
                    output_dir=result_dir,
                    enabled=False,
                )
            with self.assertRaisesRegex(ControllerInputError, "block identity"):
                prepare_controller_input(
                    output,
                    block_id=BLOCK_IDS["B"],
                    ledger_path=ledger,
                    expected_ledger_sha256=ledger_hash,
                    output_dir=result_dir,
                    enabled=True,
                )
            with self.assertRaisesRegex(ControllerInputError, "ledger SHA"):
                prepare_controller_input(
                    output,
                    block_id=BLOCK_IDS["A"],
                    ledger_path=ledger,
                    expected_ledger_sha256="0" * 64,
                    output_dir=result_dir,
                    enabled=True,
                )
            with self.assertRaisesRegex(ControllerInputError, "reviewed source manifest"):
                prepare_controller_input(
                    output,
                    block_id=BLOCK_IDS["A"],
                    ledger_path=ledger,
                    expected_ledger_sha256=ledger_hash,
                    expected_source_manifest_sha256="0" * 64,
                    output_dir=result_dir,
                    enabled=True,
                )

            plan = prepare_controller_input(
                output,
                block_id=BLOCK_IDS["A"],
                ledger_path=ledger,
                expected_ledger_sha256=ledger_hash,
                output_dir=result_dir,
                enabled=True,
            )

            self.assertTrue(plan["executable"])
            self.assertEqual(plan["environment"]["ROUND16_A1_PROFILE_ENABLE"], "1")
            self.assertEqual(plan["block_id"], BLOCK_IDS["A"])
            self.assertEqual(plan["generation_limit"], 6)
            self.assertEqual(plan["ledger_sha256"], ledger_hash)
            self.assertRegex(plan["schedule_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                [item["role"] for item in plan["schedule"]],
                [
                    "W1_WARMUP",
                    "W2_WARMUP",
                    "W1_MEASUREMENT",
                    "W2_MEASUREMENT",
                    "SHORT_DRAIN",
                    "FIXED_CANARY",
                ],
            )
            self.assertFalse(result_dir.exists())
            self.assertNotIn("command", plan)

    def test_controller_rejects_mutable_or_spent_ledger_and_occupied_output(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            result_dir = root / "fresh-result"

            for update, message in (
                ({"immutable": False}, "immutable"),
                (
                    {
                        "used_by_block": {BLOCK_IDS["A"]: 1, BLOCK_IDS["B"]: 0},
                        "request_ids": ["a-1"],
                    },
                    "fresh",
                ),
            ):
                ledger_payload = initial_ledger_payload()
                ledger_payload.update(update)
                ledger = root / f"ledger-{message}.json"
                ledger.write_text(json.dumps(ledger_payload) + "\n")
                with self.assertRaisesRegex(ControllerInputError, message):
                    prepare_controller_input(
                        output,
                        block_id=BLOCK_IDS["A"],
                        ledger_path=ledger,
                        expected_ledger_sha256=_sha(ledger.read_bytes()),
                        output_dir=result_dir,
                        enabled=True,
                    )

            ledger = self._write_ledger(root)
            result_dir.mkdir()
            with self.assertRaisesRegex(ControllerInputError, "output"):
                prepare_controller_input(
                    output,
                    block_id=BLOCK_IDS["A"],
                    ledger_path=ledger,
                    expected_ledger_sha256=_sha(ledger.read_bytes()),
                    output_dir=result_dir,
                    enabled=True,
                )

    def test_raw_receipt_requires_exact_six_unique_generation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            plan = prepare_controller_input(
                output,
                block_id=BLOCK_IDS["A"],
                ledger_path=ledger,
                expected_ledger_sha256=_sha(ledger.read_bytes()),
                output_dir=root / "result",
                enabled=True,
            )
            rows = [
                {"ordinal": ordinal, "request_id": f"a-{ordinal}", "status": "COMPLETE"}
                for ordinal in range(1, 7)
            ]

            receipt = raw_receipt(plan, rows, observer_path="round16-a1-profile.jsonl")

            self.assertEqual(receipt["status"], "RAW_DEVICE_RESULT_UNREVIEWED")
            self.assertEqual(receipt["generation_attempts"], 6)
            self.assertEqual(receipt["block_id"], BLOCK_IDS["A"])
            self.assertEqual(receipt["schedule_sha256"], plan["schedule_sha256"])
            self.assertEqual(receipt["observer_status"], "UNVALIDATED_RAW_INPUT")
            self.assertEqual(receipt["execution_review"], "REQUIRED_BEFORE_CANDIDATE_SELECTION")
            with self.assertRaisesRegex(ControllerInputError, "six"):
                raw_receipt(plan, rows[:-1], observer_path="profile.jsonl")

    def test_controller_executes_fixed_six_item_queue_and_writes_raw_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            result_dir = root / "fresh-result"
            plan = prepare_controller_input(
                output,
                block_id=BLOCK_IDS["A"],
                ledger_path=ledger,
                expected_ledger_sha256=_sha(ledger.read_bytes()),
                output_dir=result_dir,
                enabled=True,
            )
            seen: list[tuple[str, str]] = []

            def driver(item, *, request_id, output_dir, snapshot_root, environment):
                self.assertEqual(output_dir, result_dir)
                self.assertEqual(snapshot_root, output)
                self.assertEqual(environment["ROUND16_A1_PROFILE_ENABLE"], "1")
                self.assertEqual(environment["ROUND16_A1_BLOCK_ID"], BLOCK_IDS["A"])
                seen.append((item["role"], request_id))
                return {"status": "COMPLETE", "driver_receipt": f"raw-{item['ordinal']}"}

            receipt = execute_controller(plan, driver)

            self.assertEqual(
                [role for role, _ in seen],
                [
                    "W1_WARMUP",
                    "W2_WARMUP",
                    "W1_MEASUREMENT",
                    "W2_MEASUREMENT",
                    "SHORT_DRAIN",
                    "FIXED_CANARY",
                ],
            )
            self.assertEqual(len({request_id for _, request_id in seen}), 6)
            self.assertEqual(receipt["generation_attempts"], 6)
            self.assertTrue((result_dir / "round16-a1-raw-receipt.json").is_file())
            ledger_after = json.loads(
                (result_dir / "round16-a1-ledger-after.json").read_text()
            )
            self.assertEqual(ledger_after["used_by_block"][BLOCK_IDS["A"]], 6)
            self.assertEqual(ledger_after["used_by_block"][BLOCK_IDS["B"]], 0)

    def test_controller_rechecks_manifest_and_ledger_before_driver_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            result_dir = root / "fresh-result"
            plan = prepare_controller_input(
                output,
                block_id=BLOCK_IDS["A"],
                ledger_path=ledger,
                expected_ledger_sha256=_sha(ledger.read_bytes()),
                output_dir=result_dir,
                enabled=True,
            )
            ledger.write_text(ledger.read_text() + "\n")
            events: list[str] = []

            with self.assertRaisesRegex(ControllerInputError, "ledger SHA"):
                execute_controller(plan, lambda *args, **kwargs: events.append("called"))

            self.assertEqual(events, [])
            self.assertFalse(result_dir.exists())

    def test_controller_rejects_schedule_mutation_before_driver_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, output = self._build(root)
            ledger = self._write_ledger(root)
            result_dir = root / "fresh-result"
            plan = prepare_controller_input(
                output,
                block_id=BLOCK_IDS["A"],
                ledger_path=ledger,
                expected_ledger_sha256=_sha(ledger.read_bytes()),
                output_dir=result_dir,
                enabled=True,
            )
            plan["schedule"][0]["role"] = "FORGED"
            events: list[str] = []

            with self.assertRaisesRegex(ControllerInputError, "schedule SHA"):
                execute_controller(plan, lambda *args, **kwargs: events.append("called"))

            self.assertEqual(events, [])
            self.assertFalse(result_dir.exists())

    def test_two_sequential_blocks_consume_exactly_six_each_and_twelve_total(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, snapshot_a = self._build(root / "a-build", block="A")
            _, snapshot_b = self._build(root / "b-build", block="B")
            ledger = self._write_ledger(root)

            def complete_driver(
                item, *, request_id, output_dir, snapshot_root, environment
            ):
                self.assertEqual(environment["ROUND16_A1_PROFILE_ENABLE"], "1")
                return {"status": "COMPLETE", "role": item["role"]}

            plan_a = prepare_controller_input(
                snapshot_a,
                block_id=BLOCK_IDS["A"],
                ledger_path=ledger,
                expected_ledger_sha256=_sha(ledger.read_bytes()),
                output_dir=root / "result-a",
                enabled=True,
            )
            execute_controller(plan_a, complete_driver)
            ledger_after_a = root / "result-a/round16-a1-ledger-after.json"
            plan_b = prepare_controller_input(
                snapshot_b,
                block_id=BLOCK_IDS["B"],
                ledger_path=ledger_after_a,
                expected_ledger_sha256=_sha(ledger_after_a.read_bytes()),
                output_dir=root / "result-b",
                enabled=True,
            )
            execute_controller(plan_b, complete_driver)
            final_ledger = json.loads(
                (root / "result-b/round16-a1-ledger-after.json").read_text()
            )

            self.assertEqual(
                final_ledger["used_by_block"],
                {BLOCK_IDS["A"]: 6, BLOCK_IDS["B"]: 6},
            )
            self.assertEqual(len(final_ledger["request_ids"]), 12)

    def test_block_b_requires_block_a_exactly_complete(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, snapshot_b = self._build(root, block="B")
            ledger = self._write_ledger(root)

            with self.assertRaisesRegex(ControllerInputError, "block A.*complete"):
                prepare_controller_input(
                    snapshot_b,
                    block_id=BLOCK_IDS["B"],
                    ledger_path=ledger,
                    expected_ledger_sha256=_sha(ledger.read_bytes()),
                    output_dir=root / "result-b",
                    enabled=True,
                )

    def test_block_b_rejects_forged_six_request_id_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            _, snapshot_b = self._build(root, block="B")
            forged = initial_ledger_payload()
            forged["used_by_block"][BLOCK_IDS["A"]] = 6
            forged["request_ids"] = [f"stale-a-request-{ordinal}" for ordinal in range(1, 7)]
            ledger = root / "forged-ledger.json"
            ledger.write_text(json.dumps(forged, indent=2) + "\n")

            with self.assertRaisesRegex(ControllerInputError, "frozen block A request"):
                prepare_controller_input(
                    snapshot_b,
                    block_id=BLOCK_IDS["B"],
                    ledger_path=ledger,
                    expected_ledger_sha256=_sha(ledger.read_bytes()),
                    output_dir=root / "result-b",
                    enabled=True,
                )

    def test_initial_ledger_is_canonical_two_block_twelve_generation_contract(self) -> None:
        ledger = initial_ledger_payload()

        self.assertTrue(ledger["immutable"])
        self.assertEqual(ledger["generation_limit_total"], 12)
        self.assertEqual(ledger["generation_limit_per_block"], 6)
        self.assertEqual(
            ledger["used_by_block"],
            {BLOCK_IDS["A"]: 0, BLOCK_IDS["B"]: 0},
        )
        self.assertEqual(ledger["request_ids"], [])
        self.assertRegex(canonical_json_sha256(ledger), r"^[0-9a-f]{64}$")

    def test_frozen_ledger_and_controller_input_artifacts_match_runtime_payloads(self) -> None:
        artifact_root = Path(__file__).resolve().parents[2] / "artifacts/round16-a"
        ledger = json.loads(
            (artifact_root / "FRESH_A1_LEDGER_INITIAL.json").read_text()
        )
        self.assertEqual(ledger, initial_ledger_payload())
        for block in ("A", "B"):
            controller = json.loads(
                (
                    artifact_root
                    / f"FRESH_A1_CONTROLLER_INPUT_{block}.json"
                ).read_text()
            )
            self.assertEqual(controller, controller_input_template(block))
            self.assertFalse(controller["enabled"])
            self.assertFalse(controller["executable"])
            self.assertEqual(controller["block_id"], BLOCK_IDS[block])
            self.assertIsNone(controller["source_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
