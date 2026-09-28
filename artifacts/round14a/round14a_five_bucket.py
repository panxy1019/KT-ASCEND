"""Round14a five-bucket storage census for one frozen S0 model process.

The census runs after model and request-state pools are constructed.  It uses
underlying storage identity, so views and shared parameters contribute bytes
once.  Allocator slack and whole-card residuals stay outside the five logical
buckets instead of being attributed to an arbitrary owner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import types


BUCKETS = ("Expert", "PLE", "Dense backbone", "KV+GDN state", "Workspace")


def classify_model_name(name: str) -> str:
    lower = "." + name.lower() + "."
    if ".ple." in lower or "position_learning" in lower:
        return "PLE"
    if ".experts." in lower or ".routed_expert" in lower:
        return "Expert"
    return "Dense backbone"


def _storage(tensor):
    storage = tensor.untyped_storage()
    size = storage.nbytes()
    if callable(size):
        size = size()
    pointer = storage.data_ptr()
    return (str(tensor.device), int(pointer), int(size)), int(size)


def _device_family(tensor) -> str:
    value = str(tensor.device).lower()
    if value.startswith("npu"):
        return "npu"
    if value.startswith("cpu"):
        return "host"
    return "other"


class StorageLedger:
    def __init__(self):
        self._owners = {}
        self._records = []
        self._aliases = []

    def add(self, bucket: str, name: str, tensor) -> bool:
        if bucket not in BUCKETS:
            raise ValueError(f"unknown Round14a bucket: {bucket}")
        key, storage_bytes = _storage(tensor)
        existing = self._owners.get(key)
        if existing is not None:
            self._aliases.append({
                "owner_bucket": existing["bucket"],
                "owner": existing["name"],
                "alias_bucket": bucket,
                "alias": name,
                "device": key[0],
                "storage_bytes": storage_bytes,
            })
            return False
        record = {
            "bucket": bucket,
            "name": name,
            "device": key[0],
            "device_family": _device_family(tensor),
            "storage_pointer": key[1],
            "storage_bytes": storage_bytes,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        self._owners[key] = record
        self._records.append(record)
        return True

    def document(self):
        summary = {
            bucket: {"npu_bytes": 0, "host_bytes": 0, "other_bytes": 0,
                     "unique_storages": 0}
            for bucket in BUCKETS
        }
        for row in self._records:
            field = ("npu_bytes" if row["device_family"] == "npu" else
                     "host_bytes" if row["device_family"] == "host" else
                     "other_bytes")
            summary[row["bucket"]][field] += row["storage_bytes"]
            summary[row["bucket"]]["unique_storages"] += 1
        return {
            "buckets": summary,
            "unique_storage_count": len(self._records),
            "unique_storage_bytes": sum(row["storage_bytes"] for row in self._records),
            "records": self._records,
            "aliases": self._aliases,
        }


def _is_tensor(value) -> bool:
    try:
        import torch
        return isinstance(value, torch.Tensor)
    except Exception:
        return False


def _walk_tensors(value, path, bucket, ledger, *, depth=0, visited=None):
    if value is None or depth > 8:
        return
    if visited is None:
        visited = set()
    if _is_tensor(value):
        ledger.add(bucket, path, value)
        return
    if isinstance(value, (str, bytes, int, float, bool, Path, types.FunctionType,
                          types.MethodType, type)):
        return
    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_tensors(child, f"{path}[{key!r}]", bucket, ledger,
                          depth=depth + 1, visited=visited)
    elif isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _walk_tensors(child, f"{path}[{index}]", bucket, ledger,
                          depth=depth + 1, visited=visited)
    elif hasattr(value, "__dict__"):
        for key, child in vars(value).items():
            if key in {"model", "module", "runner", "model_runner"}:
                continue
            _walk_tensors(child, f"{path}.{key}", bucket, ledger,
                          depth=depth + 1, visited=visited)


def summarize_allocators(*, allocated: int, reserved: int,
                         persistent_npu_bytes: int, board_used: int | None,
                         owned_used: int | None):
    owned = allocated if owned_used is None else owned_used
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "persistent_classified_npu_bytes": persistent_npu_bytes,
        "workspace_unclassified_allocated_bytes": max(0, allocated - persistent_npu_bytes),
        "allocator_reserved_unallocated_bytes": max(0, reserved - allocated),
        "owned_used_bytes": owned_used,
        "board_used_bytes": board_used,
        "board_driver_or_external_residual_bytes": (
            None if board_used is None else max(0, board_used - owned)
        ),
    }


def _named_model_tensors(model):
    for method_name in ("named_parameters", "named_buffers"):
        method = getattr(model, method_name)
        try:
            rows = method(recurse=True, remove_duplicate=False)
        except TypeError:
            rows = method(recurse=True)
        for name, tensor in rows:
            yield name, tensor


def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _bank_document():
    path = os.environ.get("ROUND9_BANK_RECEIPT")
    if not path:
        return {"status": "UNKNOWN_NO_BANK_RECEIPT_ENV"}
    document = json.loads(Path(path).read_text())
    banks = document.get("banks", [])
    return {
        "status": "REGISTERED_READ_ONLY_MXFP4_BANK",
        "receipt": path,
        "bank_root": document.get("bank_root"),
        "files": len(banks),
        "logical_file_bytes": sum(int(row["bytes"]) for row in banks),
        "accounting": "READ_ONLY_FILE_BYTES_NOT_ASSUMED_RESIDENT_OR_ANONYMOUS_RSS",
    }


def capture(runner, output=None):
    if os.environ.get("ROUND14A_FIVE_BUCKET_ENABLE") != "1":
        return None
    output = Path(output or os.environ.get(
        "ROUND14A_FIVE_BUCKET_PATH", "/out/round14a-five-bucket-memory.json"))
    if output.exists():
        raise FileExistsError(output)
    import torch

    ledger = StorageLedger()
    for name, tensor in _named_model_tensors(runner.model):
        ledger.add(classify_model_name(name), "model." + name, tensor)

    # These roots are intentionally ordered: PLE and request state own their
    # storage before the residual workspace census sees aliases.
    _walk_tensors(getattr(runner, "_round9_ple", None), "runner._round9_ple",
                  "PLE", ledger)
    for name in ("req_to_token_pool", "token_to_kv_pool",
                 "token_to_kv_pool_allocator"):
        _walk_tensors(getattr(runner, name, None), "runner." + name,
                      "KV+GDN state", ledger)
    for name in ("_round9_context_reference", "_round9_prefill",
                 "_round9_streaming", "graph_runner"):
        _walk_tensors(getattr(runner, name, None), "runner." + name,
                      "Workspace", ledger)

    census = ledger.document()
    persistent_npu = sum(row["npu_bytes"] for row in census["buckets"].values())
    allocated = int(torch.npu.memory_allocated())
    reserved = int(torch.npu.memory_reserved())
    board_mib = _read_int(os.environ.get("ROUND14A_BOARD_USED_MIB_FILE", ""))
    owned_mib = _read_int(os.environ.get("ROUND14A_OWNED_USED_MIB_FILE", ""))
    document = {
        "schema": 1,
        "status": "PASS_ROUND14A_FIVE_BUCKET_STORAGE_CENSUS",
        "scope": "MODEL_READY_AFTER_KV_GDN_PLE_POOLS_BEFORE_REQUESTS",
        "deduplication": "DEVICE_STORAGE_POINTER_AND_STORAGE_BYTES",
        **census,
        "allocator": summarize_allocators(
            allocated=allocated, reserved=reserved,
            persistent_npu_bytes=persistent_npu,
            board_used=None if board_mib is None else board_mib * 1024**2,
            owned_used=None if owned_mib is None else owned_mib * 1024**2,
        ),
        "declared_transient_reservations": {
            "layerwise_prefill_bytes": int(getattr(runner, "_round9_prefill_workspace_bytes", 0)),
            "ple_bytes": int(getattr(runner, "_round9_ple_workspace_bytes", 0)),
            "qsa_context_bytes": int(getattr(runner, "_round9_context_workspace_bytes", 0)),
        },
        "host_read_only_bank": _bank_document(),
        "limitations": [
            "Host file bytes are listed separately from page residency and cgroup RSS.",
            "Workspace is the allocated residual after persistent storage census; allocator slack is separate.",
            "Whole-card/owned peaks are finalized from the external monitor, not inferred here.",
        ],
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print("ROUND14A_FIVE_BUCKET_CAPTURED", persistent_npu, allocated, reserved, flush=True)
    return document


__all__ = ("BUCKETS", "StorageLedger", "capture", "classify_model_name",
           "summarize_allocators")
