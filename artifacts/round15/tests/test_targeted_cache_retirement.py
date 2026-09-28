from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "targeted_cache_retirement.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round15_cache_retirement", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_and_summarize_fincore():
    module = load_module()
    payload = json.dumps(
        {
            "fincore": [
                {"pages": 2, "size": 5000, "res": 8192, "file": "/x/a.gguf"},
                {"pages": 0, "size": 9000, "res": 0, "file": "/x/b.safetensors"},
            ]
        }
    )
    rows = module.parse_fincore_json(payload)
    assert rows[0]["resident_bytes"] == 8192
    assert rows[1]["resident_pages"] == 0
    summary = module.summarize_fincore(rows)
    assert summary == {
        "file_count": 2,
        "total_size_bytes": 14000,
        "resident_pages": 2,
        "resident_bytes": 8192,
        "resident_file_count": 1,
    }


def test_manifest_is_exact_and_identity_checked():
    module = load_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bank = root / "layer-00-mxfp4.gguf"
        model = root / "model-00001-of-00001.safetensors"
        bank.write_bytes(b"bank")
        model.write_bytes(b"model")
        rows = []
        for role, path in (("bank", bank), ("model", model)):
            stat = path.stat()
            rows.append(
                {
                    "role": role,
                    "path": str(path),
                    "size": stat.st_size,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        targets = module.validate_registered_targets(rows, expected_count=2)
        assert targets == [bank.resolve(), model.resolve()]
        rows[0]["size"] += 1
        try:
            module.validate_registered_targets(rows, expected_count=2)
        except RuntimeError as error:
            assert "identity mismatch" in str(error)
        else:
            raise AssertionError("identity mismatch must fail closed")


def test_retirement_threshold_requires_material_drop():
    module = load_module()
    assert module.retirement_sufficient(1000, 100, minimum_drop_bytes=800)
    assert not module.retirement_sufficient(1000, 900, minimum_drop_bytes=800)
    assert module.retirement_sufficient(0, 0, minimum_drop_bytes=800)


def test_build_manifest_uses_catalog_and_model_index_only():
    module = load_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bank_root = root / "banks"
        model_root = root / "model"
        bank_root.mkdir()
        model_root.mkdir()
        bank = bank_root / "layer-00.gguf"
        model = model_root / "model-00001.safetensors"
        bank.write_bytes(b"bank")
        model.write_bytes(b"model")
        catalog = root / "catalog.json"
        catalog.write_text(
            json.dumps(
                {
                    "banks": [
                        {
                            "bank_path": "/banks/layer-00.gguf",
                            "bytes": bank.stat().st_size,
                            "sha256": "trusted-bank-sha",
                        }
                    ]
                }
            )
        )
        index = model_root / "model.safetensors.index.json"
        index.write_text(
            json.dumps({"weight_map": {"x": "model-00001.safetensors"}})
        )
        manifest = module.build_registered_manifest(
            bank_catalog=catalog,
            bank_container_root=Path("/banks"),
            bank_host_root=bank_root,
            model_index=index,
            model_root=model_root,
            expected_bank_count=1,
            expected_model_count=1,
        )
        assert [row["role"] for row in manifest["targets"]] == ["bank", "model"]
        assert manifest["targets"][0]["trusted_sha256"] == "trusted-bank-sha"
        assert manifest["targets"][1]["path"] == str(model.resolve())


if __name__ == "__main__":
    test_parse_and_summarize_fincore()
    test_manifest_is_exact_and_identity_checked()
    test_retirement_threshold_requires_material_drop()
    test_build_manifest_uses_catalog_and_model_index_only()
    print("PASS_ROUND15_TARGETED_CACHE_RETIREMENT_TESTS")
