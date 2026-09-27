import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "artifacts/round14/build_round14_a_snapshot.py"


def load_module(path):
    spec = importlib.util.spec_from_file_location("round14_native_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeNpu:
    def __init__(self):
        self.original_is_available = self.is_available
        self.original_mem_get_info = self.mem_get_info
        self.original_get_device_name = self.get_device_name

    def is_available(self):
        return False

    def mem_get_info(self):
        raise RuntimeError("no device is exposed")

    def get_device_name(self, _device=0):
        raise RuntimeError("no device is exposed")


class FakeTorch:
    def __init__(self):
        self.npu = FakeNpu()


class Round14NativePreflightTest(unittest.TestCase):
    def test_no_device_scope_supplies_capacity_and_restores_both_probes(self):
        builder = load_module(BUILDER)
        parent = ROOT / "artifacts/round12a/native_tp8/offline-snapshots-r2/v180"
        rendered = builder.render_native_control_plane(parent, builder.contract(239))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native_tp8_image_preflight.py"
            path.write_text(rendered["native_tp8_image_preflight.py"])
            api = load_module(path)
            torch = FakeTorch()
            original_available = torch.npu.is_available
            original_mem = torch.npu.mem_get_info
            original_name = torch.npu.get_device_name
            with api.no_device_npu_import_probe(torch) as receipt:
                self.assertTrue(torch.npu.is_available())
                self.assertEqual(torch.npu.mem_get_info(), (65536 * 1024**2, 65536 * 1024**2))
                self.assertEqual(torch.npu.get_device_name(0), "Ascend910B")
                self.assertEqual(receipt["memory_capacity_for_serverargs_mib"], 65536)
            self.assertEqual(torch.npu.is_available.__func__, original_available.__func__)
            self.assertEqual(torch.npu.mem_get_info.__func__, original_mem.__func__)
            self.assertEqual(torch.npu.get_device_name.__func__, original_name.__func__)
            self.assertTrue(receipt["restored"])
            self.assertTrue(receipt["mem_get_info_restored"])
            self.assertTrue(receipt["get_device_name_restored"])

    def test_preflight_checks_every_driver_dependency_before_model_load(self):
        builder = load_module(BUILDER)
        parent = ROOT / "artifacts/round12a/native_tp8/offline-snapshots-r2/v180"
        rendered = builder.render_native_control_plane(parent, builder.contract(247))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "native_tp8_image_preflight.py"
            path.write_text(rendered["native_tp8_image_preflight.py"])
            (root / "driver.py").write_text("pass\n")
            (root / "fixture.json").write_text("{}\n")
            (root / "ROUND14_A_CONTRACT.json").write_text(__import__("json").dumps({
                "driver_command": [
                    "{DRIVER_ROOT}/driver.py", "--fixture",
                    "{DRIVER_ROOT}/fixture.json",
                ]
            }))
            api = load_module(path)
            self.assertEqual(
                api.validate_driver_dependencies(root),
                ["driver.py", "fixture.json"],
            )
            (root / "fixture.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "fixture.json"):
                api.validate_driver_dependencies(root)

    def test_preflight_checks_lazy_local_imports_before_model_load(self):
        builder = load_module(BUILDER)
        parent = ROOT / "artifacts/round12a/native_tp8/offline-snapshots-r2/v180"
        rendered = builder.render_native_control_plane(parent, builder.contract(251))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "native_tp8_image_preflight.py"
            path.write_text(rendered["native_tp8_image_preflight.py"])
            (root / "round14_a_driver.py").write_text(
                "def f():\n"
                "    from round11_t4_driver import consume_stream\n"
                "    from round10_t1_protocol import validate_response\n"
            )
            (root / "round11_t4_driver.py").write_text("def consume_stream(): pass\n")
            (root / "round10_t1_protocol.py").write_text("def validate_response(): pass\n")
            api = load_module(path)
            self.assertEqual(
                api.validate_local_import_dependencies(root, root / "round14_a_driver.py"),
                ["round10_t1_protocol.py", "round11_t4_driver.py"],
            )
            (root / "round10_t1_protocol.py").unlink()
            with self.assertRaisesRegex(RuntimeError, "round10_t1_protocol.py"):
                api.validate_local_import_dependencies(root, root / "round14_a_driver.py")


if __name__ == "__main__":
    unittest.main()
