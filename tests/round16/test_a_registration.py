from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from round16.a.provider_manifest import PROVIDER_NAME
from round16.a.registration import RegistrationConflictError, register_provider
from round16.a.sglang_adapter import KTAscendProvider


class RegistrationTests(unittest.TestCase):
    def test_mapping_registration_is_explicit_and_idempotent(self) -> None:
        registry: dict[str, object] = {}

        first = register_provider(registry)
        second = register_provider(registry)

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertIs(registry[PROVIDER_NAME], KTAscendProvider)

    def test_registration_refuses_to_overwrite_a_different_provider(self) -> None:
        registry = {PROVIDER_NAME: object()}

        with self.assertRaises(RegistrationConflictError):
            register_provider(registry)

    def test_fresh_import_does_not_import_runtime_dependencies_or_mutate_environment(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = """
import json, os, sys
before = dict(os.environ)
import round16.a
blocked = [name for name in ('torch', 'torch_npu', 'sglang', 'ktransformers') if name in sys.modules]
print(json.dumps({'blocked': blocked, 'environment_changed': before != dict(os.environ)}))
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)

        completed = subprocess.run(
            [sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r); %s" % (str(root), script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(
            completed.stdout.strip(),
            '{"blocked": [], "environment_changed": false}',
        )

    def test_offline_install_exposes_only_the_round16_packages(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="round16-a-install-") as target:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "tools" / "install_round16_a.py"),
                    "--target",
                    target,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    (
                        "import sys; sys.path.insert(0, %r); "
                        "import round16.a; "
                        "print(round16.a.PROVIDER_NAME)"
                    )
                    % target,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.stdout.strip(), PROVIDER_NAME)


if __name__ == "__main__":
    unittest.main()
