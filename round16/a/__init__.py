"""Portable Round16 SGLang/KT provider integration.

Importing this module performs no registration and imports no runtime package.
"""

from round16.a.provider_manifest import (
    OPERATION_GDN,
    OPERATION_MOE,
    OPERATION_QSA,
    PROVIDER_NAME,
    dependency_tiers,
    manifest_payload,
)
from round16.a.registration import register_provider
from round16.a.sglang_adapter import KTAscendProvider

__all__ = [
    "KTAscendProvider",
    "OPERATION_GDN",
    "OPERATION_MOE",
    "OPERATION_QSA",
    "PROVIDER_NAME",
    "dependency_tiers",
    "manifest_payload",
    "register_provider",
]
