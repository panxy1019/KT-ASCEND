from __future__ import annotations

import importlib.util
import unittest


class Round16BPackageTests(unittest.TestCase):
    def test_round16_b_package_exists(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("round16.b"))


if __name__ == "__main__":
    unittest.main()
