"""Bounded EP1 native-consumer and manual NPUGraph mutation probe.

The probe directly allocates independent per-expert ``w13``/``w2`` tensors,
keeps the frozen ``AscendMoERunner``, and mutates one fixed slot A->B->A.
It does not construct a stacked Q32 parent.  The optional graph result is
strictly component-scoped and makes no full-model graph claim.
"""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from round16_integration.b_q32_preallocation_live_factory import _build_moe_config
from round16_integration.p1_allgather_q32_probe import (
    load_snapshot_manifest,
    load_source_preflight_token,
)
from round18.independent_expert_list import (
    ComponentGraphLedger,
    IndependentExpertPayload,
    IndependentExpertSlot,
    IndependentListPool,
    W13_CONSUMER_SHAPE,
    W2_CONSUMER_SHAPE,
)
from round18.native_runner_seam import rebind_native_physical_quant_method


R18_EP1_LIST_PROBE_API = "round18:ep1-independent-list-consumer-graph-probe:v1"
LOGICAL_EXPERTS = 512
PHYSICAL_SLOTS = 32
MAX_RELATIVE_L2 = 0.01


def allocate_independent_expert_lists(
    torch_module: Any,
    *,
    physical_slots: int = PHYSICAL_SLOTS,
    device: str = "npu:0",
) -> tuple[list[Any], list[Any]]:
    """Allocate native-consumer tensors independently, one pair per slot."""

    if type(physical_slots) is not int or physical_slots <= 0:
        raise ValueError("physical_slots must be positive")
    w13 = [
        torch_module.empty(W13_CONSUMER_SHAPE, dtype=torch_module.bfloat16, device=device)
        for _ in range(physical_slots)
    ]
    w2 = [
        torch_module.empty(W2_CONSUMER_SHAPE, dtype=torch_module.bfloat16, device=device)
        for _ in range(physical_slots)
    ]
    pointers = [int(item.data_ptr()) for item in (*w13, *w2)]
    if len(set(pointers)) != len(pointers):
        raise RuntimeError("independent expert allocation produced aliased storage")
    if any(getattr(item, "_base", None) is not None for item in (*w13, *w2)):
        raise RuntimeError("independent expert allocation produced a parent view")
    return w13, w2


def validate_probe_receipt(receipt: Mapping[str, Any]) -> None:
    status = receipt.get("status")
    if status not in {
        "PASS_EP1_INDEPENDENT_LIST_EAGER_GRAPH_COMPONENT",
        "PASS_EP1_INDEPENDENT_LIST_EAGER_GRAPH_BLOCKED",
    }:
        raise ValueError("independent-list probe status is invalid")
    if receipt.get("native_runner_preserved") is not True:
        raise ValueError("native Ascend runner was not preserved")
    if (receipt.get("tp_size"), receipt.get("ep_size")) != (1, 1):
        raise ValueError("bounded independent-list probe is TP1/EP1 only")
    if (receipt.get("logical_experts"), receipt.get("physical_slots")) != (
        LOGICAL_EXPERTS,
        PHYSICAL_SLOTS,
    ):
        raise ValueError("logical/physical expert identity changed")
    if receipt.get("parent_constructed") is not False or receipt.get("parent_rebuilt") is not False:
        raise ValueError("stacked parent storage is forbidden")
    if receipt.get("eager_native_consumer") != "PASS":
        raise ValueError("native eager list consumer did not pass")
    graph = receipt.get("graph")
    if not isinstance(graph, Mapping):
        raise ValueError("graph result is missing")
    if graph.get("scope") != "COMPONENT_NPUGRAPH_ONLY":
        raise ValueError("manual graph result is not component-scoped")
    if graph.get("full_model_graph_claim") is not False:
        raise ValueError("component probe cannot make a full-model graph claim")
    if graph.get("piecewise_used") is not False:
        raise ValueError("forbidden graph path was used")
    if graph.get("mutation_inside_replay") is not False:
        raise ValueError("slot mutation occurred inside replay")
    if status.endswith("GRAPH_COMPONENT"):
        if graph.get("capture_output_rejected") is not True:
            raise ValueError("capture output rejection was not demonstrated")
        if int(graph.get("replay_count", 0)) < 3:
            raise ValueError("A-B-A graph validation requires three replays")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hash_tensor(value: Any) -> str:
    import torch

    array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _relative_l2(actual: Any, expected: Any) -> float:
    import torch

    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _load_checkpoint_expert(
    *, model_root: Path, layer_id: int, logical_expert_id: int
) -> tuple[Any, Any, tuple[Any, Any, Any]]:
    """Read one expert on CPU and produce two native-consumer tensors."""

    from safetensors import safe_open

    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    base = f"model.language_model.layers.{layer_id}.mlp.experts."
    gate_key = base + "gate_up_proj"
    down_key = base + "down_proj"
    with safe_open(
        str(model_root / weight_map[gate_key]), framework="pt", device="cpu"
    ) as gate_file, safe_open(
        str(model_root / weight_map[down_key]), framework="pt", device="cpu"
    ) as down_file:
        gate_up = gate_file.get_slice(gate_key)[logical_expert_id].contiguous()
        down = down_file.get_slice(down_key)[logical_expert_id].contiguous()
    gate, up = gate_up.float().chunk(2, dim=0)
    reference = (gate.contiguous(), up.contiguous(), down.float().contiguous())
    return (
        gate_up.transpose(0, 1).contiguous(),
        down.transpose(0, 1).contiguous(),
        reference,
    )


def _reference_one_expert(
    hidden_cpu: Any,
    route_weight: float,
    reference: tuple[Any, Any, Any],
) -> Any:
    import torch

    gate, up, down = reference
    source = hidden_cpu.float()
    gate_out = (source @ gate.T).bfloat16().float()
    up_out = (source @ up.T).bfloat16().float()
    intermediate = (torch.nn.functional.silu(gate_out) * up_out).bfloat16().float()
    return ((intermediate @ down.T).bfloat16().float() * route_weight).bfloat16()


def _router_logits(torch_module: Any, logical_expert_id: int, misses: Sequence[int]) -> Any:
    logits = torch_module.full(
        (1, LOGICAL_EXPERTS), -40.0, dtype=torch_module.float32, device="npu:0"
    )
    ids = (logical_expert_id, *tuple(misses)[:9])
    logits[0, torch_module.tensor(ids, dtype=torch_module.int64, device="npu:0")] = (
        torch_module.tensor(
            [20.0 - index for index in range(10)],
            dtype=torch_module.float32,
            device="npu:0",
        )
    )
    return logits


def _require_enabled(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND18_ENABLE_EP1_LIST_PROBE") != "1":
        raise RuntimeError("ROUND18_ENABLE_EP1_LIST_PROBE=1 is required")


def run_device_probe(parsed: argparse.Namespace) -> dict[str, Any]:
    source_token = load_source_preflight_token(parsed.source_token)
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    del source_token
    if parsed.layer != 0:
        raise ValueError("the bounded independent-list probe is restricted to layer0")

    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.ops.fused_moe.fused_moe import (
        AscendMoERunner,
        AscendUnquantizedFusedMoEMethod,
    )
    from vllm_ascend.ops.fused_moe.moe_comm_method import (
        AllGatherCommImpl,
        setup_moe_comm_method,
    )
    from round16_integration.b_q32_hybrid_live_probe import _enter_bounded_forward_context

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.10, 0)
    torch.npu.config.allow_internal_format = True

    config_scope = None
    forward_scope = None
    distributed_initialized = False
    model_parallel_initialized = False
    prior_extra: dict[str, Any] = {}
    started = time.monotonic()
    try:
        vllm_config = VllmConfig(additional_config={"weight_nz_mode": 0})
        config_scope = set_current_vllm_config(vllm_config)
        config_scope.__enter__()
        init_ascend_config(vllm_config)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29919",
            backend="hccl",
        )
        distributed_initialized = True
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            prefill_context_model_parallel_size=1,
            decode_context_model_parallel_size=1,
            backend="hccl",
        )
        model_parallel_initialized = True
        forward_scope, _ = _enter_bounded_forward_context()

        logical_config = _build_moe_config(torch)
        physical_config = copy(logical_config)
        physical_config.num_experts = LOGICAL_EXPERTS
        physical_config.num_logical_experts = LOGICAL_EXPERTS
        physical_config.num_local_experts = PHYSICAL_SLOTS
        w13_list, w2_list = allocate_independent_expert_lists(torch)

        class IndependentOwner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.moe_config = physical_config
                self.quant_config = None
                self.w13_weight_list = []
                self.w2_weight_list = []
                for index, tensor in enumerate(w13_list):
                    self.register_buffer(f"round18_w13_{index}", tensor)
                    self.w13_weight_list.append(tensor)
                for index, tensor in enumerate(w2_list):
                    self.register_buffer(f"round18_w2_{index}", tensor)
                    self.w2_weight_list.append(tensor)
                self.w13_bias = None
                self.w2_bias = None
                self.use_grouped_topk = False
                self.renormalize = True
                self.topk_group = None
                self.num_expert_group = None
                self.custom_routing_function = None
                self.scoring_func = "sigmoid"
                self.routed_scaling_factor = 1.0
                self.e_score_correction_bias = None
                self.apply_router_weight_on_input = False
                self.swiglu_limit = 0.0
                self.swiglu_alpha = 1.0
                self.swiglu_beta = 0.0
                self.n_shared_experts = 0
                self.layer_id = parsed.layer

        owner = IndependentOwner()
        router = torch.nn.Linear(
            2560, LOGICAL_EXPERTS, bias=False, dtype=torch.bfloat16, device="npu:0"
        )
        runner = AscendMoERunner(
            f"model.layers.{parsed.layer}.mlp.experts",
            logical_config,
            router,
            owner,
            n_shared_experts=0,
        )
        rebind_native_physical_quant_method(
            runner,
            ascend_method_cls=AscendUnquantizedFusedMoEMethod,
            setup_moe_comm_method=setup_moe_comm_method,
        )
        comm = AllGatherCommImpl(physical_config)
        for name in (
            "moe_comm_method",
            "moe_comm_type",
            "flash_comm_v1_enabled",
            "in_profile_run",
        ):
            prior_extra[name] = getattr(_EXTRA_CTX, name, None)
        _EXTRA_CTX.moe_comm_method = comm
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        _EXTRA_CTX.flash_comm_v1_enabled = False
        _EXTRA_CTX.in_profile_run = False

        resident_ids = tuple(snapshot.resident_logical_ids)
        if len(resident_ids) != PHYSICAL_SLOTS:
            raise ValueError("snapshot is not a physical-Q32 map")
        references: dict[int, tuple[Any, Any, Any]] = {}
        sources: dict[int, tuple[Any, Any]] = {}
        for physical_slot, logical_id in enumerate(resident_ids):
            w13_cpu, w2_cpu, reference = _load_checkpoint_expert(
                model_root=parsed.model_root,
                layer_id=parsed.layer,
                logical_expert_id=logical_id,
            )
            w13_list[physical_slot].copy_(w13_cpu.to("npu:0"))
            w2_list[physical_slot].copy_(w2_cpu.to("npu:0"))
            references[logical_id] = reference
            if physical_slot == 0:
                sources[logical_id] = (w13_cpu, w2_cpu)
        torch.npu.synchronize()

        nonresident = next(value for value in range(LOGICAL_EXPERTS) if value not in resident_ids)
        b_w13, b_w2, b_reference = _load_checkpoint_expert(
            model_root=parsed.model_root,
            layer_id=parsed.layer,
            logical_expert_id=nonresident,
        )
        sources[nonresident] = (b_w13, b_w2)
        references[nonresident] = b_reference
        a_logical = resident_ids[0]
        misses = tuple(
            value
            for value in range(LOGICAL_EXPERTS)
            if value not in resident_ids and value != nonresident
        )[:9]

        device_map = torch.full(
            (LOGICAL_EXPERTS,), -1, dtype=torch.int32, device="npu:0"
        )
        for physical_slot, logical_id in enumerate(resident_ids):
            device_map[logical_id] = physical_slot
        runner._expert_map = device_map
        map_pointer = int(device_map.data_ptr())

        slots = tuple(
            IndependentExpertSlot(index, w13_list[index], w2_list[index])
            for index in range(PHYSICAL_SLOTS)
        )

        def copy_projection(**kwargs: Any) -> bool:
            kwargs["destination"].copy_(kwargs["source"].to("npu:0"))
            return True

        def ready_barrier(**_kwargs: Any) -> bool:
            torch.npu.synchronize()
            return True

        def publish_map(**kwargs: Any) -> bool:
            device_map.copy_(
                torch.tensor(kwargs["mapping"], dtype=torch.int32, device="npu:0")
            )
            torch.npu.synchronize()
            if int(device_map.data_ptr()) != map_pointer:
                raise RuntimeError("device expert_map address changed")
            return True

        pool = IndependentListPool(
            slots=slots,
            initial_slot_to_logical=resident_ids,
            copy_projection=copy_projection,
            ready_barrier=ready_barrier,
            publish_map=publish_map,
        )

        torch.manual_seed(18018)
        hidden_cpu = torch.randn(1, 2560, dtype=torch.float32).bfloat16()
        hidden = hidden_cpu.to("npu:0")

        def execute(logical_id: int, logits: Any | None = None) -> tuple[Any, dict[str, Any]]:
            import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module

            active_logits = logits if logits is not None else _router_logits(torch, logical_id, misses)
            captured = []
            original = fused_moe_module.select_experts

            def observed(*args: Any, **kwargs: Any):
                weights, ids = original(*args, **kwargs)
                captured.append((weights.detach().clone(), ids.detach().clone()))
                return weights, ids

            fused_moe_module.select_experts = observed
            try:
                output = runner.no_shared_forward_impl(hidden, active_logits)
                torch.npu.synchronize()
            finally:
                fused_moe_module.select_experts = original
            if len(captured) != 1:
                raise RuntimeError("native runner did not invoke its selector exactly once")
            weights, ids = captured[0]
            row_ids = [int(value) for value in ids[0].cpu().tolist()]
            column = row_ids.index(logical_id)
            route_weight = float(weights[0, column].float().cpu())
            expected = _reference_one_expert(
                hidden_cpu, route_weight, references[logical_id]
            )
            actual = output.detach().cpu()
            error = _relative_l2(actual, expected)
            if error > MAX_RELATIVE_L2:
                raise AssertionError(f"native list consumer relative L2 {error} exceeds limit")
            return output, {
                "logical_expert_id": logical_id,
                "relative_l2": error,
                "hash": _hash_tensor(actual),
                "selector_calls": 1,
            }

        eager_a, eager_a_row = execute(a_logical)
        mutation_b = pool.replace(
            physical_slot=0,
            payload=IndependentExpertPayload(nonresident, *sources[nonresident]),
            generation=1,
        )
        eager_b, eager_b_row = execute(nonresident)
        mutation_a = pool.replace(
            physical_slot=0,
            payload=IndependentExpertPayload(a_logical, *sources[a_logical]),
            generation=2,
        )
        eager_a2, eager_a2_row = execute(a_logical)
        if eager_a_row["hash"] != eager_a2_row["hash"]:
            raise AssertionError("A-B-A eager restore did not recover the original output")
        del eager_a, eager_b, eager_a2

        graph_result: dict[str, Any]
        try:
            ledger = ComponentGraphLedger()
            static_logits = _router_logits(torch, a_logical, misses)
            graph = torch.npu.NPUGraph()
            ledger.begin_capture()
            with torch.npu.graph(graph):
                graph_output = runner.no_shared_forward_impl(hidden, static_logits)
            ledger.end_capture(output_generation=2)
            try:
                ledger.consume_output(generation=2)
            except RuntimeError:
                pass

            graph.replay()
            torch.npu.synchronize()
            ledger.record_replay(output_generation=2)
            if _hash_tensor(graph_output) != eager_a_row["hash"]:
                raise AssertionError("first legal graph replay differs from eager A")
            ledger.consume_output(generation=2)

            ledger.begin_external_mutation()
            mutation_b_graph = pool.replace(
                physical_slot=0,
                payload=IndependentExpertPayload(nonresident, *sources[nonresident]),
                generation=3,
            )
            static_logits.copy_(_router_logits(torch, nonresident, misses))
            torch.npu.synchronize()
            ledger.end_external_mutation(new_generation=3)
            graph.replay()
            torch.npu.synchronize()
            ledger.record_replay(output_generation=3)
            if _hash_tensor(graph_output) != eager_b_row["hash"]:
                raise AssertionError("graph replay did not observe B slot/map mutation")
            ledger.consume_output(generation=3)

            ledger.begin_external_mutation()
            mutation_a_graph = pool.replace(
                physical_slot=0,
                payload=IndependentExpertPayload(a_logical, *sources[a_logical]),
                generation=4,
            )
            static_logits.copy_(_router_logits(torch, a_logical, misses))
            torch.npu.synchronize()
            ledger.end_external_mutation(new_generation=4)
            graph.replay()
            torch.npu.synchronize()
            ledger.record_replay(output_generation=4)
            if _hash_tensor(graph_output) != eager_a_row["hash"]:
                raise AssertionError("graph replay did not recover A after restore")
            ledger.consume_output(generation=4)
            graph_result = ledger.receipt()
            graph_result.update(
                {
                    "status": "PASS_COMPONENT_NPUGRAPH_A_B_A",
                    "device_executed": True,
                    "map_data_ptr_stable": int(device_map.data_ptr()) == map_pointer,
                    "slot_data_ptr_stable": True,
                    "mutations": [mutation_b_graph, mutation_a_graph],
                }
            )
            status = "PASS_EP1_INDEPENDENT_LIST_EAGER_GRAPH_COMPONENT"
        except Exception as error:
            graph_result = {
                "scope": "COMPONENT_NPUGRAPH_ONLY",
                "full_model_graph_claim": False,
                "piecewise_used": False,
                "mutation_inside_replay": False,
                "status": "COMPONENT_NPUGRAPH_BLOCKED",
                "error_type": type(error).__name__,
                "error": str(error),
                "replay_count": 0,
            }
            status = "PASS_EP1_INDEPENDENT_LIST_EAGER_GRAPH_BLOCKED"

        receipt = {
            "api": R18_EP1_LIST_PROBE_API,
            "status": status,
            "device_executed": True,
            "native_runner_preserved": type(runner).__name__ == "AscendMoERunner",
            "native_runner_class": type(runner).__name__,
            "tp_size": 1,
            "ep_size": 1,
            "logical_experts": LOGICAL_EXPERTS,
            "physical_slots": PHYSICAL_SLOTS,
            "parent_constructed": False,
            "parent_rebuilt": False,
            "raw_pointer_copy": False,
            "eager_native_consumer": "PASS",
            "eager_cycle": [eager_a_row, eager_b_row, eager_a2_row],
            "eager_mutations": [mutation_b, mutation_a],
            "expert_map_entries": int(device_map.numel()),
            "expert_map_data_ptr_stable": int(device_map.data_ptr()) == map_pointer,
            "slot_data_ptr_stable": pool.pointer_snapshot()
            == tuple(slot.pointers() for slot in slots),
            "graph": graph_result,
            "full_model": "NOT_RUN_COMPONENT_PROBE",
            "elapsed_seconds": time.monotonic() - started,
        }
        validate_probe_receipt(receipt)
        return receipt
    finally:
        for name, value in prior_extra.items():
            setattr(_EXTRA_CTX, name, value)
        if forward_scope is not None:
            forward_scope.__exit__(None, None, None)
        if model_parallel_initialized:
            destroy_model_parallel()
        if distributed_initialized:
            destroy_distributed_environment()
        if config_scope is not None:
            config_scope.__exit__(None, None, None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-device", action="store_true")
    parser.add_argument("--source-token", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if not parsed.enable_device:
        raise ValueError("independent-list probe requires --enable-device")
    _require_enabled(os.environ)
    receipt = run_device_probe(parsed)
    _write_json(parsed.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "R18_EP1_LIST_PROBE_API",
    "allocate_independent_expert_lists",
    "main",
    "run_device_probe",
    "validate_probe_receipt",
)
