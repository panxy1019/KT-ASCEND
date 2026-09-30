"""Round18 worker that preserves AscendMoERunner and wraps only quant apply."""

from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from vllm_ascend.worker.worker import NPUWorker


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    path = Path(value).resolve()
    if not path.exists():
        raise ValueError(f"{name} is unavailable: {path}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


class Round18NativeHybridWorker(NPUWorker):
    """Install physical-Q32 during construction and KT after post-load.

    The class does not supply ``runner_cls``.  The frozen Ascend factory
    therefore creates its normal ``AscendMoERunner`` for every layer.
    """

    def load_model(self) -> Any:
        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        if os.environ.get("ROUND18_ENABLE_NATIVE_HYBRID") != "1":
            raise RuntimeError(
                "Round18NativeHybridWorker requires explicit "
                "ROUND18_ENABLE_NATIVE_HYBRID=1"
            )

        import torch
        import torch_npu  # noqa: F401
        import vllm.model_executor.models.qwen3_next as qwen_module
        from kt_kernel import get_current_device_stream_handle
        from kt_kernel.experts_base import BaseMoEWrapper
        from kt_kernel.utils.llamafile import LlamafileMoEWrapper
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm_ascend.models.qwen4_exp.nvidia import model as _qwen_model  # noqa: F401
        from vllm_ascend.models.qwen4_exp.nvidia.ple_layer import (
            Qwen4ExpNGramEmbedding,
        )
        from vllm_ascend.models.qwen4_exp.ple_offload_layer import PleOffloadLayer
        from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa_module
        from vllm_ascend.ops import rotary_embedding as ascend_rotary_module
        from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
        from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
        from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
        from vllm_ascend.ops.fused_moe.fused_moe import (
            AscendUnquantizedFusedMoEMethod,
        )
        from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
        from vllm_ascend.patch.platform.patch_fused_moe import _ascend_FusedMoE
        from vllm_ascend.patch.worker import patch_process_weights_after_loading
        from vllm.utils.torch_utils import direct_register_custom_op

        from round16.b.exact_donor_worker_hook import (
            load_exact_worker_source_receipt,
            resolve_exact_qwen_conditional_experts,
        )
        from round16.b.selective_loader import plan_q32
        from round16_integration.b2_live_bindings import (
            FullModelKTLayerFactory,
            build_live_q32_owner_class,
            install_ascend_qsa_cache_split,
            install_ascend_qsa_multihead_store,
            install_host_ple_bridge,
        )
        from round16_integration.p1_allgather_q32_probe import (
            load_snapshot_manifest,
            load_source_preflight_token,
        )
        from round18.native_hybrid_live_binding import (
            NATIVE_HYBRID_LIVE_BINDING_API,
            attach_native_hybrid_quant_methods,
        )
        from round18.native_host_ple_graph import (
            ensure_host_ple_splitting_op,
            install_graph_safe_host_ple_bridge,
        )
        from round18.native_gdn_graph_boundary import (
            FROZEN_ASCEND_GDN_SOURCE_SHA256,
            ensure_gdn_full_forward_splitting_op,
            install_gdn_full_forward_graph_boundary,
        )
        from round18.native_graph_shape_contract import (
            install_static_rotary_cos_sin_split,
        )
        from round18.native_runner_seam import (
            ModelScopedNativePartialInstaller,
            rebind_native_physical_quant_method,
        )
        from round18.native_route_trace import NativeRouteTraceRecorder

        started = time.time()
        snapshot = load_snapshot_manifest(
            _required_path("ROUND18_SNAPSHOT_MANIFEST")
        )
        source_token = load_source_preflight_token(
            _required_path("ROUND18_SOURCE_TOKEN")
        )
        worker_token = load_exact_worker_source_receipt(
            _required_path("ROUND18_WORKER_PREFLIGHT").read_text()
        )
        del source_token, worker_token  # Validation is the required effect.
        plan = plan_q32(snapshot.resident_logical_ids)

        # Preserve the complete two-head QSA cache/store contract.  The
        # omitted-axis normalization alone is insufficient for TP1 QSA.
        install_host_ple_bridge(
            torch_module=torch,
            ple_offload_cls=PleOffloadLayer,
            ple_embedding_cls=Qwen4ExpNGramEmbedding,
            postload_patch_module=patch_process_weights_after_loading,
        )
        requested_graph_mode = os.environ.get(
            "ROUND18_REQUESTED_GRAPH_MODE", "NONE"
        )
        graph_shape_receipt = None
        if requested_graph_mode != "NONE":
            graph_shape_receipt = {
                "gdn_full_forward": install_gdn_full_forward_graph_boundary(
                    torch_module=torch,
                    gdn_cls=QwenGatedDeltaNetAttention,
                    ascend_gdn_cls=AscendGatedDeltaNetAttention,
                    direct_register_custom_op=direct_register_custom_op,
                    get_forward_context=get_forward_context,
                    expected_source_sha256=FROZEN_ASCEND_GDN_SOURCE_SHA256,
                ),
                "rotary": install_static_rotary_cos_sin_split(
                    rotary_cls=AscendMRotaryEmbedding,
                    rotary_module=ascend_rotary_module,
                ),
            }
            install_graph_safe_host_ple_bridge(
                torch_module=torch,
                ple_offload_cls=PleOffloadLayer,
                ple_embedding_cls=Qwen4ExpNGramEmbedding,
                direct_register_custom_op=direct_register_custom_op,
                get_forward_context=get_forward_context,
            )
            ensure_host_ple_splitting_op(self.vllm_config.compilation_config)
            ensure_gdn_full_forward_splitting_op(
                self.vllm_config.compilation_config
            )
        install_ascend_qsa_cache_split(qsa_module=ascend_qsa_module)
        install_ascend_qsa_multihead_store(
            qsa_impl_cls=ascend_qsa_module.AscendQSAImpl,
            qsa_module=ascend_qsa_module,
        )

        owner_cls = build_live_q32_owner_class(
            native_routed_experts_cls=RoutedExperts,
            torch_module=torch,
            ascend_method_cls=AscendUnquantizedFusedMoEMethod,
            setup_moe_comm_method=setup_moe_comm_method,
            plan=plan,
        )
        if qwen_module.FusedMoE is not _ascend_FusedMoE:
            raise RuntimeError("target Qwen lacks the frozen Ascend FusedMoE binding")
        installer = ModelScopedNativePartialInstaller(
            qwen_module=qwen_module,
            ascend_fused_moe=_ascend_FusedMoE,
            routed_experts_cls=owner_cls,
            physical_runtime_binding=lambda runner: (
                rebind_native_physical_quant_method(
                    runner,
                    ascend_method_cls=AscendUnquantizedFusedMoEMethod,
                    setup_moe_comm_method=setup_moe_comm_method,
                )
            ),
            expected_layer_count=48,
        )

        BaseMoEWrapper.set_capture_batch_sizes([1, 32])
        kt_layers = FullModelKTLayerFactory(
            torch_module=torch,
            wrapper_cls=LlamafileMoEWrapper,
            stream_handle_provider=get_current_device_stream_handle,
            bank_root=_required_path("ROUND18_BANK_ROOT"),
            plan=plan,
        )

        try:
            with installer.activate(plan=plan) as acknowledge:
                result = super().load_model()
                loaded_model = self.model_runner.get_model()
                acknowledge(loaded_model)
            resolved = resolve_exact_qwen_conditional_experts(
                loaded_model, expected_layer_count=48
            )

            def identities(layer_name: str):
                def snapshot_identity(*, generation: int, **_kwargs: Any):
                    context = get_forward_context()
                    input_ids = getattr(context, "input_ids", None)
                    pointer = (
                        int(input_ids.data_ptr())
                        if input_ids is not None
                        and callable(getattr(input_ids, "data_ptr", None))
                        else None
                    )
                    return (
                        "round18-native-hybrid",
                        id(context),
                        pointer,
                        layer_name,
                        int(generation),
                    )

                def is_current(identity_key: tuple[Any, ...]) -> bool:
                    context = get_forward_context()
                    input_ids = getattr(context, "input_ids", None)
                    pointer = (
                        int(input_ids.data_ptr())
                        if input_ids is not None
                        and callable(getattr(input_ids, "data_ptr", None))
                        else None
                    )
                    return (
                        len(identity_key) == 5
                        and identity_key[0] == "round18-native-hybrid"
                        and identity_key[1] == id(context)
                        and identity_key[2] == pointer
                        and identity_key[3] == layer_name
                    )

                return snapshot_identity, is_current

            receipts = attach_native_hybrid_quant_methods(
                resolved_layers=resolved,
                plan=plan,
                selector_module=fused_moe_module,
                callbacks_for_layer=kt_layers.callbacks,
                identity_provider_for_layer=identities,
                expected_layer_count=48,
                route_observer=(
                    NativeRouteTraceRecorder(
                        _required_path("ROUND18_ROUTE_TRACE_PARENT")
                        / "native-route-trace.jsonl",
                        max_events=int(
                            os.environ.get("ROUND18_ROUTE_TRACE_MAX_EVENTS", "20000")
                        ),
                        enabled_path=_required_path(
                            "ROUND18_ROUTE_TRACE_ENABLE_PARENT"
                        )
                        / "enabled",
                    ).record
                    if os.environ.get("ROUND18_ROUTE_TRACE_PARENT")
                    else None
                ),
            )
            self.round18_native_hybrid_factory = kt_layers
            self.round18_native_hybrid_receipts = receipts
            receipt_path = os.environ.get("ROUND18_BINDING_RECEIPT_PATH")
            if receipt_path:
                compilation = self.vllm_config.compilation_config
                resolved_graph_mode = getattr(
                    getattr(compilation, "cudagraph_mode", None),
                    "name",
                    str(getattr(compilation, "cudagraph_mode", None)),
                )
                resolved_compilation_mode = getattr(
                    getattr(compilation, "mode", None),
                    "name",
                    str(getattr(compilation, "mode", None)),
                )
                _write_json(
                    Path(receipt_path),
                    {
                        "schema": 1,
                        "status": "PASS_ROUND18_NATIVE_HYBRID_LIVE_BINDING",
                        "started_at": started,
                        "ended_at": time.time(),
                        "api": NATIVE_HYBRID_LIVE_BINDING_API,
                        "logical_experts": 512,
                        "physical_slots_per_layer": 32,
                        "layer_count": len(receipts),
                        "runner_classes": sorted(
                            {receipt.runner_class for receipt in receipts}
                        ),
                        "delegate_classes": sorted(
                            {receipt.delegate_class for receipt in receipts}
                        ),
                        "runner_preserved": all(
                            receipt.runner_preserved for receipt in receipts
                        ),
                        "qsa_cache_contract": [
                            "split_page_axis",
                            "all_tp1_kv_heads_stored",
                        ],
                        "requested_graph_mode": os.environ.get(
                            "ROUND18_REQUESTED_GRAPH_MODE", "NONE"
                        ),
                        "resolved_graph_mode": resolved_graph_mode,
                        "resolved_compilation_mode": resolved_compilation_mode,
                        "resolved_splitting_ops": list(
                            getattr(compilation, "splitting_ops", None) or ()
                        ),
                        "graph_shape_contract": graph_shape_receipt,
                        "layers": [
                            {
                                "layer_name": receipt.layer_name,
                                "runner_class": receipt.runner_class,
                                "delegate_class": receipt.delegate_class,
                            }
                            for receipt in receipts
                        ],
                    },
                )
            return result
        except BaseException:
            # No second model is loaded in this qualification worker.  The
            # process terminates after a failed load, so partially created KT
            # resources are retired by process teardown rather than exposed.
            raise


__all__ = ("Round18NativeHybridWorker",)
