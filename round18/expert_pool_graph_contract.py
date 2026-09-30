"""Offline source contract for the Round18 P10 graph-safe expert-pool gate.

This module does not import vLLM, torch, or torch_npu.  It audits the frozen
source tree for the smallest set of control/data-flow primitives that justify
one bounded device probe.  A positive result is deliberately not an NPU PASS:
only an actual capture -> mutate fixed slot -> publish map -> replay sequence
can establish that the captured native MoE reads the new contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_SOURCE_FILES = (
    "vllm_ascend/compilation/acl_graph.py",
    "vllm_ascend/worker/model_runner_v1.py",
    "vllm_ascend/eplb/adaptor/vllm_adaptor.py",
    "vllm_ascend/eplb/core/eplb_device_transfer_loader.py",
    "vllm_ascend/eplb/core/eplb_utils.py",
    "vllm_ascend/ops/fused_moe/fused_moe.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_of(text: str, needle: str) -> int | None:
    for line_number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return line_number
    return None


def _evidence(relative_path: str, text: str, needle: str) -> dict[str, Any]:
    return {
        "file": relative_path,
        "line": _line_of(text, needle),
        "needle": needle,
    }


def audit_frozen_graph_pool_contract(source_root: str | Path) -> dict[str, Any]:
    """Return a conservative, machine-readable P10 source audit.

    The gate opens only when the frozen tree contains all of these primitives:
    graph replay without re-entering the Python runnable, model-level hooks
    outside the graph, completed-transfer wait, stable in-place expert weight
    overwrite, in-place device mapping publish, and native MoE consumption of
    that device mapping.  EP1 support and current-token miss planning are
    recorded independently; their absence limits the probe design rather than
    erasing the adjacent EPLB mechanism.
    """

    root = Path(source_root).resolve()
    paths = {relative: root / relative for relative in REQUIRED_SOURCE_FILES}
    missing_files = [relative for relative, path in paths.items() if not path.is_file()]
    if missing_files:
        return {
            "schema": "round18:p10-graph-safe-pool-source-contract:v1",
            "source_root": str(root),
            "disposition": "BLOCKED_INCOMPLETE_FROZEN_SOURCE",
            "missing_source_files": missing_files,
            "claims": {"npu_pass": False, "graph_correctness": False},
        }

    sources = {relative: path.read_text() for relative, path in paths.items()}
    acl = sources[REQUIRED_SOURCE_FILES[0]]
    runner = sources[REQUIRED_SOURCE_FILES[1]]
    adaptor = sources[REQUIRED_SOURCE_FILES[2]]
    loader = sources[REQUIRED_SOURCE_FILES[3]]
    utils = sources[REQUIRED_SOURCE_FILES[4]]
    moe = sources[REQUIRED_SOURCE_FILES[5]]

    replay_call = "entry.aclgraph.replay()"
    runnable_call = "self.runnable(*args, **kwargs)"
    replay_skips_runnable = replay_call in acl and acl.rfind(runnable_call) < acl.rfind(replay_call)
    replay_index = acl.rfind(replay_call)
    stream_sync_index = acl.rfind("torch.npu.current_stream().synchronize()", 0, replay_index)
    stream_sync_before_replay = stream_sync_index >= 0
    hook_before = "self.eplb_updator.forward_before()" in runner
    hook_after = "self.eplb_updator.forward_end(" in runner
    model_forward = "self._model_forward(" in runner
    hooks_bracket_model = (
        hook_before
        and hook_after
        and model_forward
        and runner.find("self.eplb_updator.forward_before()")
        < runner.find("self._model_forward(")
        < runner.find("self.eplb_updator.forward_end(", runner.find("self._model_forward("))
    )

    weight_copy = "expert_tensor.copy_(buffer_tensor)" in adaptor
    mapping_copy = "self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)" in adaptor
    cpu_shadow_copy = "self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)" in adaptor
    wait_index = loader.find("req.wait()")
    map_publish_index = loader.find("do_update_log2phy_map")
    weight_publish_index = loader.find("do_update_expert_weight")
    wait_precedes_publish = (
        wait_index >= 0
        and map_publish_index > wait_index
        and weight_publish_index > wait_index
    )
    native_reads_map = "log2phy=self.log2phy" in moe
    native_reads_weights = "self._quant_method.apply(" in moe
    ep1_rejected = "if ep_size == 1:" in utils and "assert not eplb_enable" in utils

    missing_primitives = []
    checks = {
        "replay_without_python_runnable": replay_skips_runnable,
        "full_current_stream_sync_before_replay": stream_sync_before_replay,
        "graph_external_model_hooks": hooks_bracket_model,
        "stable_in_place_weight_overwrite": weight_copy,
        "transfer_completion_before_publish": wait_precedes_publish,
        "in_place_device_mapping_publish": mapping_copy,
        "native_moe_reads_published_mapping": native_reads_map,
        "native_moe_reads_fixed_weight_objects": native_reads_weights,
    }
    missing_primitives.extend(name for name, present in checks.items() if not present)
    disposition = (
        "PROCEED_TO_BOUNDED_NPU_COMPONENT_PROBE"
        if not missing_primitives
        else "BLOCKED_MISSING_SOURCE_PRIMITIVE"
    )

    return {
        "schema": "round18:p10-graph-safe-pool-source-contract:v1",
        "source_root": str(root),
        "source_identity": {
            relative: {"sha256": _sha256(path)} for relative, path in paths.items()
        },
        "disposition": disposition,
        "claims": {
            "npu_pass": False,
            "graph_correctness": False,
            "full_or_full_decode_only_qualified": False,
            "piecewise_qualified": False,
            "all_native_pool_designs_impossible": False,
        },
        "full_replay": {
            "replay_skips_python_runnable": replay_skips_runnable,
            "current_stream_sync_before_replay": stream_sync_before_replay,
            "existing_cpu_python_planner_compatible": False,
            "graph_external_update_hooks_exist": hooks_bracket_model,
            "current_token_route_available_before_replay": False,
        },
        "publish_path": {
            "weight_copy_is_in_place": weight_copy,
            "wait_precedes_publish": wait_precedes_publish,
            "device_mapping_is_in_place": mapping_copy,
            "expert_map_update_is_cpu_shadow_only": cpu_shadow_copy,
            "address_stability_proved_on_npu": False,
            "captured_graph_observes_new_weight_contents_proved_on_npu": False,
        },
        "native_consumer": {
            "reads_log2phy": native_reads_map,
            "reads_weight_objects": native_reads_weights,
        },
        "mode_assessment": {
            "FULL_OR_FULL_DECODE_ONLY": "SOURCE_HAS_GRAPH_EXTERNAL_MUTATION_SEAM_DEVICE_PROBE_REQUIRED",
            "PIECEWISE": "SAME_OUTER_MUTATION_SEAM_DEVICE_PROBE_REQUIRED",
            "ACLGRAPH_WEIGHT_COPY_INSIDE_CAPTURE": "NOT_ESTABLISHED_AND_NOT_REQUIRED_BY_MINIMAL_PROBE",
        },
        "limits": {
            "current_token_route_miss_fill": "MISSING",
            "current_token_host_route_readback": "NOT_PRESENT_IN_ADJACENT_EPLB_PATH",
            "per_token_python_miss_planner": "NOT_PRESENT_IN_ADJACENT_EPLB_PATH",
            "ep1_existing_eplb_path": (
                "REJECTED_BY_FROZEN_SOURCE" if ep1_rejected else "NOT_PROVED_REJECTED"
            ),
            "existing_transfer_source": "D2D_ONLY",
            "atomic_weight_then_mapping_publish": "NOT_PROVED",
            "round18_q32_map_publish_equivalence_to_log2phy": "NOT_PROVED",
        },
        "missing_primitives": missing_primitives,
        "evidence": [
            _evidence(REQUIRED_SOURCE_FILES[0], acl, replay_call),
            _evidence(REQUIRED_SOURCE_FILES[0], acl, "torch.npu.current_stream().synchronize()"),
            _evidence(REQUIRED_SOURCE_FILES[1], runner, "self.eplb_updator.forward_before()"),
            _evidence(REQUIRED_SOURCE_FILES[1], runner, "self.eplb_updator.forward_end("),
            _evidence(REQUIRED_SOURCE_FILES[2], adaptor, "expert_tensor.copy_(buffer_tensor)"),
            _evidence(REQUIRED_SOURCE_FILES[2], adaptor, "log2phy_map_per_layer[layer_id].copy_("),
            _evidence(REQUIRED_SOURCE_FILES[3], loader, "req.wait()"),
            _evidence(REQUIRED_SOURCE_FILES[4], utils, "if ep_size == 1:"),
            _evidence(REQUIRED_SOURCE_FILES[5], moe, "log2phy=self.log2phy"),
        ],
        "minimum_device_probe": [
            "Use one native unquantized MoE layer with two preallocated physical slots and stable tensor addresses.",
            "Capture one legal FULL or PIECEWISE ACLGraph invocation with logical expert A mapped to the target slot.",
            "Outside capture and between replays, wait for copy completion, overwrite only that fixed slot with expert B weights, then publish a same-address device mapping update.",
            "Replay without Python model forward or route readback and compare output against eager A/B references plus an unchanged-slot control.",
            "Record data_ptr values, stream/event ordering, resolved graph mode, output hashes/errors, allocator deltas, and cleanup; a source-contract PASS alone is never an NPU PASS.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_frozen_graph_pool_contract(args.source_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
    return 0 if result["disposition"] == "PROCEED_TO_BOUNDED_NPU_COMPONENT_PROBE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
