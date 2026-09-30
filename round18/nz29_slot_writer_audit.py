"""Source gate for a parent-preserving Ascend NZ29 expert-slot writer.

This module is deliberately device-free.  It distinguishes the frozen
Physical-Q32 parent representation from vLLM-Ascend's separate per-expert
tensor-list EPLB representation.  The latter is useful evidence, but is not a
drop-in writer for the former.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


API = "round18:nz29-fixed-slot-source-audit:v1"

FROZEN_FILES = {
    "moe": "vllm_ascend/ops/fused_moe/fused_moe.py",
    "eplb_adaptor": "vllm_ascend/eplb/adaptor/vllm_adaptor.py",
    "eplb_loader": "vllm_ascend/eplb/core/eplb_device_transfer_loader.py",
    "eplb_utils": "vllm_ascend/eplb/core/eplb_utils.py",
    "offloader": "vllm_ascend/model_executor/offloader/prefetch.py",
}

REQUIRED_TOKENS = {
    "moe": (
        "layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)",
        "layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]",
        "w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight",
    ),
    "eplb_adaptor": (
        "buffer_tensor = torch.empty_like(expert_tensor)",
        "expert_tensor.copy_(buffer_tensor)",
        "self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)",
    ),
    "eplb_loader": (
        "local_expert_to_replace, buffer_tensor_id = recv_expert_info",
        "self.eplb_adaptor.do_update_expert_weight",
    ),
    "eplb_utils": (
        'assert not eplb_enable, "EPLB must used in expert parallelism."',
    ),
    "offloader": (
        "torch_npu.npu_format_cast(buffer, ACL_FORMAT_FRACTAL_NZ)",
        "buffer_pool._buffers[key]",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    *,
    source_root: Path,
    index_copy_source: Path,
    npu_utils_source: Path,
    round9_findings: Path,
    v018_receipt: Path,
) -> dict[str, Any]:
    texts: dict[str, str] = {}
    identities: dict[str, str] = {}
    for name, relative in FROZEN_FILES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        missing = [token for token in REQUIRED_TOKENS[name] if token not in text]
        if missing:
            raise ValueError(f"{relative} misses frozen evidence: {missing!r}")
        texts[name] = text
        identities[str(path)] = _sha256(path)

    for path in (index_copy_source, npu_utils_source, round9_findings, v018_receipt):
        if not path.is_file():
            raise FileNotFoundError(path)
        identities[str(path)] = _sha256(path)

    index_copy = index_copy_source.read_text(encoding="utf-8")
    npu_utils = npu_utils_source.read_text(encoding="utf-8")
    for token in (
        "contiguous_self = npu_utils::format_contiguous(self)",
        "npu_utils::format_fresh_view(self, result)",
    ):
        if token not in index_copy:
            raise ValueError(f"fixed IndexCopy source misses {token!r}")
    if "x.copy_(y);" not in npu_utils:
        raise ValueError("fixed NpuUtils source misses full format_fresh_view copy")

    findings = round9_findings.read_text(encoding="utf-8")
    if "shape 是完整 32-slot parent" not in findings or "完整 parent→parent" not in findings:
        raise ValueError("Round9 trace summary no longer proves the full-parent boundary")

    receipt = json.loads(v018_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "FAIL_FIXED_SLOT_POSTLOAD_FORMAT_INELIGIBLE":
        raise ValueError("v018 is not the frozen pre-mutation NZ29 rejection")
    backing = receipt.get("postload_backing", {})
    if backing.get("w13", {}).get("npu_format") != 29 or backing.get("w2", {}).get("npu_format") != 29:
        raise ValueError("v018 did not observe NZ29 for both routed weights")
    if receipt.get("mutation_started") is not False:
        raise ValueError("v018 mutated an unqualified parent")

    return {
        "schema": 1,
        "api": API,
        "status": "SOURCE_AUDIT_COMPLETE",
        "current_q32_parent": {
            "representation": "TWO_CONTIGUOUS_FORMAT29_PARENTS_W13_W2",
            "single_expert_parent_preserving_writer": "NOT_FOUND",
            "index_copy_logical_correctness": "HISTORICALLY_DEVICE_QUALIFIED",
            "index_copy_physical_scope": "FULL_PARENT_CONTIGUOUS_AND_WRITEBACK",
            "direct_slot_copy_contract": "NOT_ESTABLISHED_FOR_FORMAT29_SLICE",
            "preprocessed_blob_to_parent_slot": "NOT_FOUND",
            "disposition": "BLOCK_CURRENT_PARENT_FIXED_SLOT_PATH",
        },
        "alternate_tensor_list_seam": {
            "exists": True,
            "representation": "PER_EXPERT_TENSOR_LIST_WITH_SAME_FORMAT_EMPTY_LIKE_BUFFERS",
            "write": "expert_tensor.copy_(buffer_tensor)",
            "avoids_q32_parent_rebuild_by_construction": True,
            "frozen_activation": "FUSED_MC2_AND_USE_MEGAMOE_OR_DYNAMIC_EPLB",
            "ep1_dynamic_eplb": "REJECTED",
            "drop_in_for_current_s0": False,
            "native_consumer_change_required": True,
            "device_qualification": "NOT_RUN_FOR_ROUND18_Q32_TP1",
        },
        "offloader_seam": {
            "nz_static_buffers_exist": True,
            "granularity": "FULL_PARAMETER_SHAPE",
            "fixed_expert_slot_pool": False,
        },
        "device_evidence": {
            "v018_status": receipt["status"],
            "w13_format": backing["w13"]["npu_format"],
            "w2_format": backing["w2"]["npu_format"],
            "mutation_started": receipt["mutation_started"],
        },
        "minimum_next_probe": {
            "recommended": "PER_EXPERT_TENSOR_LIST_SAME_FORMAT_COPY_AND_NATIVE_CONSUMER",
            "scope": "ONE_LAYER_ONE_EXPERT_NO_FULL_MODEL_NO_GRAPH",
            "must_prove": [
                "stable per-expert data_ptr across A_to_B_to_A copies",
                "source and destination formats are identical",
                "native consumer changes from eager A reference to eager B reference",
                "no parent tensor or full-parent format cast exists",
                "allocator and generation publication are bounded",
            ],
            "not_authorized_claims": [
                "current S0 parent has a fixed-slot writer",
                "EP1 Dynamic EPLB is supported",
                "graph-safe residency is qualified",
                "request performance benefit",
            ],
        },
        "source_identities": identities,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--index-copy-source", type=Path, required=True)
    parser.add_argument("--npu-utils-source", type=Path, required=True)
    parser.add_argument("--round9-findings", type=Path, required=True)
    parser.add_argument("--v018-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = audit(
        source_root=args.source_root,
        index_copy_source=args.index_copy_source,
        npu_utils_source=args.npu_utils_source,
        round9_findings=args.round9_findings,
        v018_receipt=args.v018_receipt,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
