"""Fail-closed source and residency proof for Round12a non-KT OffloaderV1."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path


REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MODEL_ROOT = Path("/models") / REVISION
RECEIPT = Path("/models") / (
    REVISION
    + ".round9-support/destination-verification/"
    + "verification-1789886190714568317.json"
)
RECEIPT_SHA256 = "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
MANIFEST_SHA256 = "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
OFFLOADER_SHA256 = "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006"
CPU_OFFLOAD_GIB = 233
NUMA_NODES = (2, 3, 4, 7)
_SOURCE_CACHE = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_interleave_policy() -> dict:
    """Interleave anonymous CPU allocations over the admitted NUMA nodes."""
    library = ctypes.CDLL("libnuma.so.1")
    library.numa_available.restype = ctypes.c_int
    if library.numa_available() < 0:
        raise RuntimeError("libnuma reports NUMA unavailable")
    library.numa_parse_nodestring.argtypes = [ctypes.c_char_p]
    library.numa_parse_nodestring.restype = ctypes.c_void_p
    library.numa_set_interleave_mask.argtypes = [ctypes.c_void_p]
    library.numa_bitmask_isbitset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    library.numa_bitmask_isbitset.restype = ctypes.c_int
    library.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    mask = library.numa_parse_nodestring(b"2,3,4,7")
    if not mask:
        raise RuntimeError("libnuma could not parse the admitted node mask")
    try:
        actual = [
            node
            for node in range(8)
            if library.numa_bitmask_isbitset(mask, node)
        ]
        if actual != list(NUMA_NODES):
            raise RuntimeError(f"libnuma parsed unexpected nodes: {actual}")
        library.numa_set_interleave_mask(mask)
    finally:
        library.numa_bitmask_free(mask)
    return {"policy": "INTERLEAVE", "nodes": list(NUMA_NODES)}


def _verify_source(model: str | Path | None = None) -> dict:
    global _SOURCE_CACHE
    requested = Path(model or MODEL_ROOT).resolve(strict=True)
    root = MODEL_ROOT.resolve(strict=True)
    if requested != root:
        raise ValueError("non-KT model path differs from the frozen revision")
    if _SOURCE_CACHE is not None:
        return dict(_SOURCE_CACHE)
    raw = RECEIPT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RECEIPT_SHA256:
        raise ValueError("non-KT model receipt hash mismatch")
    data = json.loads(raw)
    if (
        data.get("status") != "PASS"
        or data.get("revision") != REVISION
        or data.get("manifest_sha256") != MANIFEST_SHA256
    ):
        raise ValueError("non-KT model receipt differs from the frozen source")
    seen = set()
    for record in data["files"]:
        name = record["path"]
        path = (root / name).resolve(strict=True)
        if (
            name in seen
            or not path.is_relative_to(root)
            or record.get("status") != "PASS"
            or path.stat().st_size != record["size"]
        ):
            raise ValueError(f"non-KT source identity failed for {name}")
        seen.add(name)
        if name in ("config.json", "model.safetensors.index.json"):
            if _sha(path) != record["sha256"]:
                raise ValueError(f"non-KT metadata changed: {name}")
    if not {"config.json", "model.safetensors.index.json"} <= seen:
        raise ValueError("non-KT receipt lacks required model metadata")
    _SOURCE_CACHE = {
        "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "receipt_sha256": digest,
        "model_path": str(root),
        "file_sizes_checked": len(seen),
        "verification_scope": (
            "TRUSTED_FULL_RECEIPT_PLUS_LIVE_SIZES_AND_METADATA_HASHES"
        ),
    }
    return dict(_SOURCE_CACHE)


def _load_contract() -> tuple[dict, str]:
    path = os.environ.get("ROUND12A_NONKT_IDENTITY")
    expected = os.environ.get("ROUND12A_NONKT_IDENTITY_SHA256")
    if not path or not expected or not Path(path).is_absolute():
        raise RuntimeError("Round12a non-KT identity path and SHA are required")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise RuntimeError("Round12a non-KT identity document changed")
    document = json.loads(raw)
    required = {
        "schema",
        "status",
        "model_revision",
        "source_identity",
        "cpu_offload_gib",
        "loaded_main_layer_bytes",
        "largest_layer_stage_bytes",
        "offloader_sha256",
        "qwen4_exp_sha256",
        "model_runner_sha256",
        "qsa_sha256",
        "context_reference_runtime_sha256",
        "context_reference_binding_sha256",
    }
    if (
        set(document) != required
        or document.get("schema") != 1
        or document.get("status") != "FROZEN_ROUND12A_NONKT_IDENTITY"
        or document.get("model_revision") != REVISION
        or document.get("cpu_offload_gib") != CPU_OFFLOAD_GIB
        or document.get("offloader_sha256") != OFFLOADER_SHA256
    ):
        raise ValueError("invalid Round12a non-KT identity document")
    return document, digest


def bind_round12a_nonkt_source_identity(model) -> dict:
    """Supply only the verified checkpoint identity to unchanged KT gates."""
    document, unused_digest = _load_contract()
    identity = _verify_source(model)
    if identity != document["source_identity"]:
        raise RuntimeError("non-KT live source identity differs from the contract")
    return {
        "receipt_sha256": identity["receipt_sha256"],
        "catalog_sha256": None,
        "source_identity": identity,
        "layer0_sha256": None,
        "backend": "SGLANG_OFFLOADERV1_FULL_BF16",
    }


def bind_round12a_nonkt_identity(
    runner, *, receipt_path: Path = Path("/out/nonkt-offloader-binding.json")
) -> dict:
    """Prove exact sources and all-layer CPU residency after model loading."""
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior.get("status") != "PASS_ROUND12A_NONKT_OFFLOADER_BINDING":
            raise RuntimeError("existing Round12a non-KT binding receipt is not PASS")
        return prior

    document, document_sha = _load_contract()
    args = runner.server_args
    if (
        args.kt_weight_path is not None
        or args.cpu_offload_gb != CPU_OFFLOAD_GIB
        or args.device != "npu"
        or args.tp_size != 1
        or not args.disable_cuda_graph
        or not args.disable_radix_cache
        or not args.disable_overlap_schedule
        or args.max_running_requests != 1
    ):
        raise RuntimeError("non-KT server arguments differ from the frozen profile")
    source_identity = _verify_source(args.model_path)
    if source_identity != document["source_identity"]:
        raise RuntimeError("non-KT live source identity differs from the contract")

    import sglang.srt.layers.round9.context_reference_binding as binding_source
    import sglang.srt.layers.round9.context_reference_runtime as runtime_source
    import sglang.srt.layers.round9.qsa as qsa_source
    import sglang.srt.model_executor.model_runner as model_runner
    import sglang.srt.models.qwen4_exp as qwen4_exp
    import sglang.srt.utils.offloader as offloader_source
    from sglang.srt.utils.offloader import OffloaderV1, get_offloader

    observed = {
        "qwen4_exp_sha256": _sha(Path(qwen4_exp.__file__).resolve()),
        "model_runner_sha256": _sha(Path(model_runner.__file__).resolve()),
        "offloader_sha256": _sha(Path(offloader_source.__file__).resolve()),
        "qsa_sha256": _sha(Path(qsa_source.__file__).resolve()),
        "context_reference_runtime_sha256": _sha(
            Path(runtime_source.__file__).resolve()
        ),
        "context_reference_binding_sha256": _sha(
            Path(binding_source.__file__).resolve()
        ),
    }
    if any(document[key] != value for key, value in observed.items()):
        raise RuntimeError("non-KT loaded source identity differs from the contract")
    offloader = get_offloader()
    if not isinstance(offloader, OffloaderV1):
        raise RuntimeError("non-KT model did not install OffloaderV1")
    if offloader._cpu_offload_max_bytes != CPU_OFFLOAD_GIB * 1024**3:
        raise RuntimeError("non-KT OffloaderV1 budget changed")

    text_model = getattr(runner.model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None or len(layers) != 48:
        raise RuntimeError("non-KT model does not expose all 48 decoder layers")
    layer_parameter_ids = set()
    cpu_parameter_bytes = 0
    per_layer = []
    for index, layer in enumerate(layers):
        parameters = list(layer.parameters())
        devices = {parameter.device.type for parameter in parameters}
        if not parameters or devices != {"cpu"}:
            raise RuntimeError(
                f"non-KT decoder layer {index} is not fully CPU resident: {devices}"
            )
        size = sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        )
        per_layer.append(size)
        cpu_parameter_bytes += size
        layer_parameter_ids.update(map(id, parameters))
    stray_cpu = [
        name
        for name, parameter in runner.model.named_parameters()
        if parameter.device.type == "cpu" and id(parameter) not in layer_parameter_ids
    ]
    if stray_cpu:
        raise RuntimeError(
            f"non-layer parameters were unexpectedly offloaded: {stray_cpu[:8]}"
        )
    if offloader._cpu_offload_bytes != cpu_parameter_bytes:
        raise RuntimeError(
            "OffloaderV1 byte counter differs from live layer parameters: "
            f"{offloader._cpu_offload_bytes} != {cpu_parameter_bytes}"
        )
    expected = int(document["loaded_main_layer_bytes"])
    if abs(cpu_parameter_bytes - expected) > 64 * 1024**2:
        raise RuntimeError(
            f"live CPU layer bytes differ from metadata audit: "
            f"{cpu_parameter_bytes} vs {expected}"
        )
    result = {
        "schema": 1,
        "status": "PASS_ROUND12A_NONKT_OFFLOADER_BINDING",
        "source_identity": source_identity,
        "identity_document_sha256": document_sha,
        **observed,
        "cpu_offload_gib": CPU_OFFLOAD_GIB,
        "decoder_layers": len(layers),
        "cpu_parameter_bytes": cpu_parameter_bytes,
        "offloader_accounted_bytes": offloader._cpu_offload_bytes,
        "per_layer_min_bytes": min(per_layer),
        "per_layer_max_bytes": max(per_layer),
        "metadata_audit_layer_bytes": expected,
        "non_layer_cpu_parameters": [],
    }
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
