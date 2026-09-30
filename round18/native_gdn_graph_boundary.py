"""Opaque full-forward boundary for the frozen native Qwen GDN.

The bundled PIECEWISE compiler passes duplicate dynamic token SymInts across
the existing ``qwen_gdn_attention_core`` split.  Its first replay can bind one
duplicate to the preceding compile-range endpoint even though both tensors
carry the current token count.  Keeping the complete, already-qualified
Ascend GDN forward eager removes those scalar ABI inputs while the surrounding
model remains eligible for PIECEWISE compilation.

This follows the frozen vLLM ``olmo_hybrid_gdn_full_forward`` pattern.  It does
not replace any GDN math, provider, state owner, or output projection.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable


GDN_FULL_FORWARD_GRAPH_API = "round18:qwen-gdn-full-forward-split:v1"
GDN_FULL_FORWARD_OP_NAME = "round18_qwen_gdn_full_forward"
GDN_FULL_FORWARD_SPLITTING_OP = f"vllm::{GDN_FULL_FORWARD_OP_NAME}"
FROZEN_ASCEND_GDN_SOURCE_SHA256 = (
    "747933e091593c92731b3b00b5f336f5aa3270b702e011426d9a632954d41bcd"
)
_MARKER = "round18_gdn_full_forward_graph_api"
_RECEIPT = "round18_gdn_full_forward_graph_receipt"


def _source_sha256(target: type) -> str:
    source_file = inspect.getsourcefile(target)
    if not source_file:
        raise RuntimeError("Ascend GDN source file is unavailable")
    return hashlib.sha256(Path(source_file).read_bytes()).hexdigest()


def ensure_gdn_full_forward_splitting_op(compilation_config: Any) -> None:
    """Add the opaque GDN call to an already normalized PIECEWISE config."""

    splitting_ops = getattr(compilation_config, "splitting_ops", None)
    if splitting_ops is None:
        splitting_ops = []
        compilation_config.splitting_ops = splitting_ops
    if GDN_FULL_FORWARD_SPLITTING_OP not in splitting_ops:
        splitting_ops.append(GDN_FULL_FORWARD_SPLITTING_OP)


def install_gdn_full_forward_graph_boundary(
    *,
    torch_module: Any,
    gdn_cls: type,
    ascend_gdn_cls: type,
    direct_register_custom_op: Callable[..., None],
    get_forward_context: Callable[[], Any],
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Run the exact frozen Ascend GDN forward behind one splitting op."""

    installed = gdn_cls.__dict__.get(_MARKER)
    if installed is not None:
        if installed != GDN_FULL_FORWARD_GRAPH_API:
            raise RuntimeError("another GDN full-forward graph adapter is installed")
        return dict(gdn_cls.__dict__[_RECEIPT])

    observed_sha256 = _source_sha256(ascend_gdn_cls)
    if observed_sha256 != expected_source_sha256:
        raise RuntimeError(
            "frozen Ascend GDN source identity mismatch: "
            f"expected {expected_source_sha256}, observed {observed_sha256}"
        )
    if gdn_cls.forward is not ascend_gdn_cls.forward:
        raise RuntimeError("target Qwen GDN lacks the frozen Ascend forward binding")
    original_forward = ascend_gdn_cls.forward
    source = inspect.getsource(original_forward)
    required_fragments = (
        "torch.ops.vllm.qwen_gdn_attention_core",
        "core_attn_out = self.norm(core_attn_out, z)",
        "out, _ = self.out_proj(core_attn_out)",
        "output[:num_tokens] = out",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise RuntimeError(
            "frozen Ascend GDN full-forward contract is unavailable: "
            + ", ".join(missing)
        )

    def full_forward(
        hidden_states: Any,
        output: Any,
        layer_name: str,
    ) -> None:
        context = get_forward_context()
        layer = context.no_compile_layers[layer_name]
        original_forward(layer, hidden_states, output=output)

    def full_forward_fake(
        hidden_states: Any,
        output: Any,
        layer_name: str,
    ) -> None:
        del hidden_states, output, layer_name

    tensor_type = torch_module.Tensor
    annotations = {
        "hidden_states": tensor_type,
        "output": tensor_type,
        "layer_name": str,
        "return": None,
    }
    full_forward.__annotations__ = dict(annotations)
    full_forward_fake.__annotations__ = dict(annotations)
    direct_register_custom_op(
        op_name=GDN_FULL_FORWARD_OP_NAME,
        op_func=full_forward,
        mutates_args=["output"],
        fake_impl=full_forward_fake,
    )

    def forward(self: Any, hidden_states: Any, output: Any = None) -> Any:
        if output is None:
            output = torch_module.empty_like(hidden_states)
        getattr(torch_module.ops.vllm, GDN_FULL_FORWARD_OP_NAME)(
            hidden_states,
            output,
            self.prefix,
        )
        return output

    forward.round18_api_identity = GDN_FULL_FORWARD_GRAPH_API
    receipt = {
        "schema": 1,
        "status": "INSTALLED_GDN_FULL_FORWARD_GRAPH_BOUNDARY",
        "api": GDN_FULL_FORWARD_GRAPH_API,
        "op": GDN_FULL_FORWARD_SPLITTING_OP,
        "class": f"{gdn_cls.__module__}.{gdn_cls.__qualname__}",
        "source_sha256": observed_sha256,
        "native_forward_preserved": True,
        "native_core_provider_preserved": True,
        "native_norm_provider_preserved": True,
        "native_out_projection_preserved": True,
        "math_changed": False,
        "state_changed": False,
        "kv_contract_claimed": False,
        "reason": "avoid duplicate dynamic token SymInt ABI across core split",
    }
    gdn_cls.forward = forward
    setattr(gdn_cls, _MARKER, GDN_FULL_FORWARD_GRAPH_API)
    setattr(gdn_cls, _RECEIPT, dict(receipt))
    return receipt


__all__ = [
    "GDN_FULL_FORWARD_GRAPH_API",
    "GDN_FULL_FORWARD_OP_NAME",
    "GDN_FULL_FORWARD_SPLITTING_OP",
    "FROZEN_ASCEND_GDN_SOURCE_SHA256",
    "ensure_gdn_full_forward_splitting_op",
    "install_gdn_full_forward_graph_boundary",
]
