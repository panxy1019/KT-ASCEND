from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OfflinePackagingTests(unittest.TestCase):
    def test_standard_library_install_and_fresh_import_have_no_runtime_side_effects(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        installer = repository / "round16" / "b" / "install.py"
        self.assertTrue(installer.is_file(), "offline installer must exist")
        with tempfile.TemporaryDirectory() as target:
            installed = subprocess.run(
                [sys.executable, str(installer), "--target", target],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)
            code = (
                "import sys; "
                "assert not {'vllm','torch','torch_npu'}.intersection(sys.modules); "
                "import round16.b; "
                "import round16_integration.p1_allgather_q32_probe; "
                "import round16_integration.b2_exact_worker_preflight; "
                "import round16_integration.b2_exact_worker_plugin; "
                "import round16_integration.b2_worker_overlay_entry; "
                "import round16_integration.b2_unresolved_bindings; "
                "assert not {'vllm','torch','torch_npu'}.intersection(sys.modules); "
                "print(round16.b.TARGET_ARCHITECTURE)"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = target
            imported = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=target,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            # -I intentionally ignores PYTHONPATH; use an explicit isolated path.
            if imported.returncode != 0:
                isolated_code = f"import sys; sys.path.insert(0, {target!r}); {code}"
                imported = subprocess.run(
                    [sys.executable, "-I", "-c", isolated_code],
                    cwd=target,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            self.assertEqual(imported.stdout.strip(), "Qwen4ExpForConditionalGeneration")


if __name__ == "__main__":
    unittest.main()
