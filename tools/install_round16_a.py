#!/usr/bin/env python3
"""Install the pure-Python Round16 component with standard-library tools."""

from __future__ import annotations

import argparse
import compileall
import json
from pathlib import Path
import shutil


def install(target: Path) -> dict[str, object]:
    source = Path(__file__).resolve().parents[1] / "round16"
    destination = target.resolve() / "round16"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing {destination}")
    destination.mkdir(parents=True)
    shutil.copy2(source / "__init__.py", destination / "__init__.py")
    for package in ("common", "a"):
        shutil.copytree(
            source / package,
            destination / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    if not compileall.compile_dir(destination, quiet=1, force=True):
        raise RuntimeError("installed package failed byte-compilation")
    return {
        "schema": 1,
        "installer": "stdlib-copytree-compileall",
        "target": str(destination),
        "package": "round16.a",
        "installed_packages": ["round16", "round16.common", "round16.a"],
        "network_used": False,
        "runtime_dependencies_imported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.target), sort_keys=True))


if __name__ == "__main__":
    main()
