# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

CPU_DETECT_PATH = Path(__file__).resolve().parents[2] / "python" / "_cpu_detect.py"
SPEC = importlib.util.spec_from_file_location(
    "kt_cpu_detect_metadata_under_test",
    CPU_DETECT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
cpu_detect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cpu_detect
SPEC.loader.exec_module(cpu_detect)


def test_detects_native_arm_before_x86_feature_matching(monkeypatch):
    monkeypatch.delenv("KT_KERNEL_CPU_VARIANT", raising=False)
    monkeypatch.setattr(cpu_detect.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        "builtins.open", lambda *_args, **_kwargs: io.StringIO("Features: fp asimd sve")
    )

    assert cpu_detect.detect_cpu_features() == "arm"


def test_detects_x86_avx2_without_regression(monkeypatch):
    monkeypatch.delenv("KT_KERNEL_CPU_VARIANT", raising=False)
    monkeypatch.setattr(cpu_detect.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        "builtins.open", lambda *_args, **_kwargs: io.StringIO("flags: sse4_2 avx avx2")
    )

    assert cpu_detect.detect_cpu_features() == "avx2"


def test_unknown_arch_preserves_x86_compatible_fallback(monkeypatch):
    monkeypatch.delenv("KT_KERNEL_CPU_VARIANT", raising=False)
    monkeypatch.setattr(cpu_detect.platform, "machine", lambda: "mystery64")
    monkeypatch.setattr(
        "builtins.open", lambda *_args, **_kwargs: io.StringIO("processor: unknown")
    )

    assert cpu_detect.detect_cpu_features() == "avx2"


def test_arm_extension_metadata_must_match_arm_host():
    cpu_detect._validate_loaded_variant("arm", "arm")

    try:
        cpu_detect._validate_loaded_variant("arm", "avx2")
    except RuntimeError as error:
        assert "different CPU architecture" in str(error)
    else:
        raise AssertionError("An x86 extension must be rejected on an ARM host")


def test_initialize_reports_loaded_extension_variant(monkeypatch):
    extension = SimpleNamespace(__cpu_variant__="avx512_bf16")
    monkeypatch.setattr(cpu_detect, "detect_cpu_features", lambda: "amx")
    monkeypatch.setattr(cpu_detect, "load_extension", lambda _variant: extension)

    loaded, variant = cpu_detect.initialize()

    assert loaded is extension
    assert variant == "avx512_bf16"


def test_initialize_rejects_extension_newer_than_host(monkeypatch):
    extension = SimpleNamespace(__cpu_variant__="amx")
    monkeypatch.setattr(cpu_detect, "detect_cpu_features", lambda: "avx512_bf16")
    monkeypatch.setattr(cpu_detect, "load_extension", lambda _variant: extension)

    try:
        cpu_detect.initialize()
    except RuntimeError as error:
        assert "requires a newer CPU ISA" in str(error)
    else:
        raise AssertionError("AMX extension must be rejected on an AVX512-BF16 host")
