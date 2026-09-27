"""Checkpoint-only identity binding for native TP8; no KT bank is loaded."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MANIFEST_SHA256 = "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
RECEIPT_SHA256 = "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
_CACHE = {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_native_tp8_source_identity(model_path):
    identity_path = os.environ.get("ROUND12A_NATIVE_TP8_IDENTITY")
    expected = os.environ.get("ROUND12A_NATIVE_TP8_IDENTITY_SHA256")
    if not identity_path or not expected:
        raise RuntimeError("native TP8 identity path/hash pair is required")
    path = Path(identity_path)
    if _sha256(path) != expected:
        raise RuntimeError("native TP8 identity changed")
    document = json.loads(path.read_text())
    source = document.get("source_identity")
    if (
        document.get("status") != "FROZEN_ROUND12A_NATIVE_TP8_OLD_QSA"
        or document.get("tp_size") != 8
        or document.get("qsa_mode") != "QUALIFIED_ORIGINAL"
        or document.get("config_sha256") != CONFIG_SHA256
        or source is None
        or source.get("revision") != REVISION
        or source.get("manifest_sha256") != MANIFEST_SHA256
        or source.get("receipt_sha256") != RECEIPT_SHA256
    ):
        raise RuntimeError("native TP8 checkpoint identity contract changed")
    key = (str(Path(model_path).resolve()), expected)
    if key not in _CACHE:
        config = Path(model_path) / "config.json"
        if not config.is_file() or _sha256(config) != CONFIG_SHA256:
            raise RuntimeError("native TP8 live model config differs from frozen identity")
        from .probe_identity import verify_probe_identity

        actual = verify_probe_identity(
            model=model_path,
            receipt=document["source_receipt"],
            expected_model=Path("/models") / REVISION,
            expected_receipt_sha256=RECEIPT_SHA256,
            revision=REVISION,
            manifest_sha256=MANIFEST_SHA256,
        )
        if actual != source:
            raise RuntimeError("native TP8 live checkpoint differs from frozen identity")
        _CACHE[key] = actual
    return {"source_identity": _CACHE[key], "backend": "SGLANG_NATIVE_TP8_BF16"}
