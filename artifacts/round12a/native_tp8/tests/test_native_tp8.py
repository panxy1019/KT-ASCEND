import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuilderContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load("native_tp8_builder")

    def test_roles_are_isolated_four_process_abba_replicates(self):
        self.assertEqual(
            self.builder.ROLES,
            {180: "A", 181: "B", 182: "B", 183: "A"},
        )
        self.assertEqual(
            self.builder.OLD_QSA_SHA256,
            "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c",
        )
        self.assertNotEqual(
            self.builder.OLD_QSA_SHA256, self.builder.PROMOTED_QSA_SHA256
        )

    def test_self_contained_frozen_inputs_match_builder_contract(self):
        frozen = ROOT / "frozen-inputs"
        for name, expected in self.builder.FROZEN_INPUT_SHA256.items():
            self.assertEqual(self.builder.sha256(frozen / name), expected)
        self.assertEqual(
            self.builder.sha256(ROOT / "numa_policy.py"),
            self.builder.FROZEN_INPUT_SHA256["numa_policy.py"],
        )

    def test_qwen_patch_adds_tp_reduce_and_keeps_explicit_gate(self):
        source = (ROOT.parents[1] / "round10/h0-v90-source/overlay-python/sglang/srt/models/qwen4_exp.py").read_text()
        patched = self.builder.patch_qwen_tp8(source)
        self.assertIn("attn_tp_all_reduce", patched)
        self.assertIn('ROUND12A_NATIVE_TP8_MODE', patched)
        self.assertIn('tp_size not in (1, 8)', patched)
        self.assertIn('hidden = attn_tp_all_reduce(hidden)', patched)
        self.assertIn('num_key_heads=self.num_k_heads // self.attn_tp_size', patched)
        self.assertIn('self.reference_config.num_value_heads', patched)
        self.assertIn('NATIVE_TP8_OLD_QSA_HIT', patched)
        self.assertIn('NATIVE_TP8_GDN_HIT', patched)
        self.assertIn('NATIVE_TP8_LAYER_PHYSICAL', patched)
        self.assertIn('"num_attention_heads": 24', patched)
        self.assertIn('qsa_layers != list(range(3, 48, 4))', patched)
        compile(patched, "qwen4_exp.py", "exec")

    def test_ple_and_framework_patches_are_explicit_and_tp8_only(self):
        source_root = ROOT.parents[1] / "round10/h0-v90-source/overlay-python/sglang/srt/layers/round9"
        ple = self.builder.patch_ple_binding((source_root / "ple_binding.py").read_text())
        self.assertIn("args.tp_size not in (1, 8)", ple)
        self.assertIn("bind_native_tp8_source_identity", ple)
        self.assertIn("ple-async-events-rank", ple)
        compile(ple, "ple_binding.py", "exec")

        hooks = self.builder.patch_framework_hooks(
            (ROOT / "frozen-inputs/framework_hooks.py").read_text()
        )
        self.assertIn("ROUND12A_NATIVE_TP8_MODE", hooks)
        native_branch = hooks.index("if os.environ.get('ROUND12A_NATIVE_TP8_MODE') == '1':")
        nonkt_branch = hooks.index("elif os.environ.get('ROUND12_NONKT_MODE') == '1':")
        self.assertLess(native_branch, nonkt_branch)
        compile(hooks, "framework_hooks.py", "exec")

    def test_qwen_patch_rejects_promoted_parent(self):
        promoted = (ROOT.parent / "fixtures/v154/qwen4_exp.py").read_text()
        with self.assertRaisesRegex(ValueError, "old Qwen"):
            self.builder.patch_qwen_tp8(promoted)

    def test_self_contained_builder_produces_frozen_native_snapshot(self):
        parent = ROOT.parents[1] / "round10/h0-v90-source"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v180"
            self.builder.build(
                parent, output, ROOT, version=180,
                queue=ROOT / "frozen-inputs/ROUND12A_ABBA_QUEUE.json",
            )
            contract = json.loads(
                (output / "ROUND12A_NATIVE_TP8_CONTRACT.json").read_text()
            )
            identity = json.loads(
                (output / "ROUND12A_NATIVE_TP8_IDENTITY.json").read_text()
            )
            self.assertEqual(contract["arm_semantics"], "IDENTICAL_NATIVE_REPEAT_COHORT")
            self.assertEqual(identity["qsa_sha256"], self.builder.OLD_QSA_SHA256)
            self.assertEqual(identity["tp8_geometry"], self.builder.TP8_GEOMETRY)
            self.assertTrue((output / "native_tp8_image_preflight.py").is_file())

    def test_success_receipt_is_written_only_after_container_removal(self):
        source = (ROOT / "native_tp8_supervisor.py").read_text()
        removal = source.index('subprocess.run(["docker", "rm", args.container_id], check=True')
        final_write = source.index('(args.out / "native-tp8-final.json").write_text')
        self.assertLess(removal, final_write)


class DriverContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load("native_tp8_driver")

    def test_schedule_is_same_64x2_warmup_plus_three_measurements(self):
        fixture_path = ROOT.parents[1] / "round10/round10-t1-fixture.json"
        fixture = json.loads(fixture_path.read_text())
        rows = self.driver.build_schedule(fixture, version=180, block="A")
        self.assertEqual([row["phase"] for row in rows], ["warmup"] + ["measurement"] * 3)
        self.assertTrue(all(row["input_tokens"] == 64 for row in rows))
        self.assertTrue(all(row["decode_tokens"] == 2 for row in rows))
        self.assertTrue(all(row["request"]["sampling_params"]["temperature"] == 0 for row in rows))
        self.assertTrue(all(row["request"]["sampling_params"]["sampling_seed"] == 20260920 for row in rows))

    def test_rank_evidence_requires_all_eight_ranks_and_layers(self):
        events = {}
        for rank in range(8):
            rows = []
            for layer in range(48):
                rows.append({
                    "status": "NATIVE_TP8_LAYER_PHYSICAL",
                    "rank": rank,
                    "layer": layer,
                    "w13_device": "npu",
                    "w2_device": "npu",
                    "dtype": "torch.bfloat16",
                    "w13_bytes": 1,
                    "w2_bytes": 1,
                })
            for layer in range(3, 48, 4):
                rows.append({
                    "status": "NATIVE_TP8_OLD_QSA_HIT",
                    "rank": rank,
                    "layer": layer,
                    "query_heads": 3,
                    "kv_heads": 1,
                    "query_tokens": 32,
                })
            for layer in self.driver.GDN_LAYERS:
                rows.append({
                    "status": "NATIVE_TP8_GDN_HIT",
                    "rank": rank,
                    "layer": layer,
                    "key_heads": 2,
                    "value_heads": 6,
                    "z_heads": 6,
                    "conv_state_shape": [2560, 3],
                    "recurrent_state_shape": [6, 128, 128],
                    "conv_state_device": "npu",
                    "recurrent_state_device": "npu",
                })
            events[rank] = rows
        receipt = self.driver.validate_rank_evidence(events)
        self.assertEqual(receipt["ranks"], list(range(8)))
        events[7] = [
            row for row in events[7]
            if not (row["status"] == "NATIVE_TP8_OLD_QSA_HIT" and row["layer"] == 47)
        ]
        with self.assertRaisesRegex(RuntimeError, "old-QSA"):
            self.driver.validate_rank_evidence(events)

    def test_tp8_sparse_gqa_partition_matches_full_geometry(self):
        # Full geometry has 24 query heads and two KV heads, so each KV head
        # owns 12 adjacent query heads.  QKVParallelLinear replicates one KV
        # head across each group of four TP8 ranks; each rank owns three query
        # heads.  Concatenating the rank-local mapping must reproduce full GQA.
        full_kv_owner = [head // 12 for head in range(24)]
        rank_local_kv_owner = [rank // 4 for rank in range(8) for _ in range(3)]
        self.assertEqual(rank_local_kv_owner, full_kv_owner)


class AdmissionAndMonitorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admission = load("native_tp8_admission")
        cls.monitor = load("native_tp8_monitor")

    def test_npu_parser_maps_first_eight_physical_devices(self):
        raw = "\n".join(
            f"| {p // 2} Ascend910 | OK | - |\n| {p % 2} {p} | 0000:{0x9d - p:02X}:00.0 | 0 0 / 0 {3000+p} / 65536 |"
            for p in range(8)
        )
        memory = self.admission.parse_first8_memory(raw)
        self.assertEqual(set(memory), set(range(8)))
        self.assertTrue(all(row["total_mib"] == 65536 for row in memory.values()))

    def test_container_name_gate_only_allows_new_versions(self):
        self.assertTrue(self.monitor.owned_container_name("kt-round12a-native-tp8-a-v180"))
        self.assertTrue(self.monitor.owned_container_name("kt-round12a-native-tp8-b-v182"))
        self.assertFalse(self.monitor.owned_container_name("kt-round12a-nonkt-b-v176"))

    def test_json_roundtrip_numeric_device_keys_are_accepted(self):
        row = {device: {1234: 100 + device} for device in range(8)}
        restored = json.loads(json.dumps(row))
        self.assertEqual(self.admission.keyed(restored, 7), {"1234": 107})


class ImagePreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preflight = load("native_tp8_image_preflight")

    def test_no_device_import_shim_is_scoped_and_restores_probe(self):
        class FakeNPU:
            def is_available(self):
                return False

        torch_module = SimpleNamespace(npu=FakeNPU())
        with mock.patch.dict("os.environ", {"ASCEND_RT_VISIBLE_DEVICES": ""}):
            with self.preflight.no_device_npu_import_probe(torch_module) as receipt:
                self.assertTrue(torch_module.npu.is_available())
            self.assertTrue(receipt["restored"])
            self.assertFalse(torch_module.npu.is_available())

    def test_no_device_import_shim_rejects_visible_devices(self):
        torch_module = SimpleNamespace(
            npu=SimpleNamespace(is_available=lambda: False)
        )
        with mock.patch.dict(
            "os.environ", {"ASCEND_RT_VISIBLE_DEVICES": "0"}
        ):
            with self.assertRaisesRegex(RuntimeError, "must not expose"):
                with self.preflight.no_device_npu_import_probe(torch_module):
                    pass

    def test_no_device_preflight_defers_full_qwen_native_extension_import(self):
        source = (ROOT / "native_tp8_image_preflight.py").read_text()
        self.assertNotIn(
            "import sglang.srt.models.qwen4_exp as qwen4_exp", source
        )
        self.assertIn(
            '"QWEN_IMPORT_DEFERRED_TO_DEVICE_LOAD"', source
        )


class ComparatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparator = load("compare_native_tp8")

    def test_four_process_order_is_fixed(self):
        self.assertEqual(
            self.comparator.ORDER,
            [(180, "A"), (181, "B"), (182, "B"), (183, "A")],
        )

    def test_identical_native_repeats_require_resource_safety_and_stability(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for offset, (version, block) in enumerate(self.comparator.ORDER):
                rows = [
                    {
                        "phase": "measurement", "e2e_seconds": 10 + offset * 0.01,
                        "ttft_seconds": 9 + offset * 0.01, "tpot_seconds": 1.0,
                        "output_ids": [3897, 248046], "quality_pass": True,
                    }
                    for _ in range(3)
                ]
                document = {
                    "status": "PASS_ROUND12A_NATIVE_TP8_MODEL_PROCESS",
                    "version": version, "block": block, "mode": "NATIVE_TP8",
                    "runtime": {"qsa_sha256": self.comparator.OLD_QSA_SHA256, "tp_size": 8},
                    "results": rows,
                    "rank_evidence": {"ranks": list(range(8))},
                    "resource": {
                        "per_device_peak_hbm_mib": {str(i): 33000 + i for i in range(8)},
                        "host_cgroup_peak_bytes": 350 * 1024**3,
                        "min_host_mem_available_kib": 700 * 1024**2,
                        "min_numa_memfree_kib": {str(n): 32 * 1024**2 for n in (2, 3, 4, 7)},
                    },
                }
                path = Path(directory) / f"v{version}.json"
                path.write_text(json.dumps(document))
                paths.append(path)
            report = self.comparator.compare(paths)
            self.assertEqual(report["status"], "PASS_ROUND12A_NATIVE_TP8_BASELINE_ABBA")
            self.assertTrue(report["resource"]["runtime_memory_safety_floors_pass"])


if __name__ == "__main__":
    unittest.main()
