"""Install Round16-B and its shared contracts using only the standard library."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    arguments = parser.parse_args()

    source = Path(__file__).resolve().parents[1]
    integration_source = source.parent / "round16_integration"
    target_root = arguments.target.resolve()
    destination = target_root / "round16"
    integration_destination = target_root / "round16_integration"
    target_root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or integration_destination.exists():
        raise FileExistsError(
            "refusing to overwrite an existing offline install"
        )
    if not integration_source.is_dir():
        raise FileNotFoundError(
            f"integration probe package is missing: {integration_source}"
        )
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copytree(
        integration_source,
        integration_destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
