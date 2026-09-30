"""Exact graph-shape adapters for the frozen Round18 Native model.

PIECEWISE AOT compilation turns the otherwise fixed Qwen GDN ``ba`` width
into an unbacked symbol.  The frozen implementation uses ``chunk(2)``; that
operation must prove a fixed two-element Python tuple and the bundled PyTorch
guard builder crashes because the synthetic symbol has no source.  This
adapter expresses the same split with model-constant slices.  It changes no
provider, arithmetic, state, or storage ownership.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any


GDN_STATIC_BA_SPLIT_API = "round18:gdn-static-ba-view-split:v1"
ROTARY_STATIC_COS_SIN_SPLIT_API = "round18:rotary-static-cos-sin-view-split:v1"
GDN_RUNTIME_TOKEN_SHAPE_API = "round18:gdn-runtime-token-shape:v1"
FROZEN_QWEN_GDN_SOURCE_SHA256 = (
    "1227d6f385a52296e9f08223544b1c5fdc7e8d9aa09a848e7a8e522a8dc51214"
)
FROZEN_ASCEND_ROTARY_SOURCE_SHA256 = (
    "6ce6f668ce3e10617fa8bbd9b7b6313a5a3fe16ea454f69a7c3d30432a85a04e"
)
FROZEN_ASCEND_GDN_SOURCE_SHA256 = (
    "747933e091593c92731b3b00b5f336f5aa3270b702e011426d9a632954d41bcd"
)
_MARKER = "round18_static_ba_split_api"
_RECEIPT = "round18_static_ba_split_receipt"
_ROTARY_MARKER = "round18_static_cos_sin_split_api"
_ROTARY_RECEIPT = "round18_static_cos_sin_split_receipt"
_GDN_TOKEN_MARKER = "round18_runtime_token_shape_api"
_GDN_TOKEN_RECEIPT = "round18_runtime_token_shape_receipt"


def _source_sha256(target: type) -> str:
    source_file = inspect.getsourcefile(target)
    if not source_file:
        raise RuntimeError("GDN class source file is unavailable")
    return hashlib.sha256(Path(source_file).read_bytes()).hexdigest()


def install_static_gdn_ba_split(
    *,
    gdn_cls: type,
    expected_source_sha256: str = FROZEN_QWEN_GDN_SOURCE_SHA256,
) -> dict[str, Any]:
    """Install the exact slice form of the frozen ``split_ba`` method.

    The physical projection width is a construction-time model constant:
    normally ``num_v_heads / tp_size`` per half.  The old Marlin replicated
    path first has ``num_v_heads`` per half and then retains its existing
    rank-local slice.  Round18 is TP1, but preserving the old branch keeps the
    adapter semantically faithful rather than silently narrowing the method.
    """

    installed = gdn_cls.__dict__.get(_MARKER)
    if installed is not None:
        if installed != GDN_STATIC_BA_SPLIT_API:
            raise RuntimeError("another GDN split adapter is already installed")
        return dict(gdn_cls.__dict__[_RECEIPT])

    observed_sha256 = _source_sha256(gdn_cls)
    if observed_sha256 != expected_source_sha256:
        raise RuntimeError(
            "frozen GDN source identity mismatch: "
            f"expected {expected_source_sha256}, observed {observed_sha256}"
        )
    original_source = inspect.getsource(gdn_cls.split_ba)
    if ".chunk(2, dim=-1)" not in original_source:
        raise RuntimeError("frozen GDN split_ba contract is unavailable")

    def split_ba(self: Any, ba: Any) -> tuple[Any, Any]:
        replicated = self.disable_tp_for_ba_proj and self.tp_size > 1
        half_width = (
            self.num_v_heads
            if replicated
            else self.num_v_heads // self.tp_size
        )
        b = ba[..., :half_width]
        a = ba[..., half_width : 2 * half_width]
        if replicated:
            # Preserve the frozen replicated-Marlin rank selection exactly.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    split_ba.round18_api_identity = GDN_STATIC_BA_SPLIT_API
    receipt = {
        "schema": 1,
        "status": "INSTALLED_STATIC_GDN_BA_SPLIT",
        "api": GDN_STATIC_BA_SPLIT_API,
        "class": f"{gdn_cls.__module__}.{gdn_cls.__qualname__}",
        "source_sha256": observed_sha256,
        "original_expression": "ba.chunk(2, dim=-1)",
        "replacement_expression": "model-constant half-width slices",
        "projected_half_width": "num_v_heads/tp_size",
        "math_changed": False,
        "state_changed": False,
        "provider_changed": False,
        "kv_contract_claimed": False,
    }
    gdn_cls.split_ba = split_ba
    setattr(gdn_cls, _MARKER, GDN_STATIC_BA_SPLIT_API)
    setattr(gdn_cls, _RECEIPT, dict(receipt))
    return receipt


def install_static_rotary_cos_sin_split(
    *,
    rotary_cls: type,
    rotary_module: Any,
    expected_source_sha256: str = FROZEN_ASCEND_ROTARY_SOURCE_SHA256,
) -> dict[str, Any]:
    """Replace only the frozen MRoPE cache ``chunk(2)`` with static slices.

    The AOT partition sees the cache row as a source-less symbolic width even
    though ``rotary_dim`` is a construction-time model constant.  Keeping the
    rest of ``forward_triton`` verbatim at the Python level preserves the
    native Triton MRoPE provider and all query/key state semantics.
    """

    installed = rotary_cls.__dict__.get(_ROTARY_MARKER)
    if installed is not None:
        if installed != ROTARY_STATIC_COS_SIN_SPLIT_API:
            raise RuntimeError("another rotary split adapter is already installed")
        return dict(rotary_cls.__dict__[_ROTARY_RECEIPT])

    observed_sha256 = _source_sha256(rotary_cls)
    if observed_sha256 != expected_source_sha256:
        raise RuntimeError(
            "frozen Ascend rotary source identity mismatch: "
            f"expected {expected_source_sha256}, observed {observed_sha256}"
        )
    original_source = inspect.getsource(rotary_cls.forward_triton)
    if "cos, sin = cos_sin.chunk(2, dim=-1)" not in original_source:
        raise RuntimeError("frozen Ascend MRoPE split contract is unavailable")
    if not callable(getattr(rotary_module, "triton_mrope", None)):
        raise RuntimeError("frozen Ascend triton_mrope provider is unavailable")

    def forward_triton(
        self: Any,
        positions: Any,
        query: Any,
        key: Any | None = None,
        offsets: Any | None = None,
    ) -> tuple[Any, Any]:
        del offsets
        assert positions.ndim == 2
        assert key is not None

        self._match_cos_sin_cache_dtype(query)
        self.cos = None
        self.sin = None
        if self.cos is None and self.sin is None:
            cos_sin = self.cos_sin_cache[positions]
            half_width = self.rotary_dim // 2
            self.cos = cos_sin[..., :half_width].contiguous()
            self.sin = cos_sin[..., half_width : 2 * half_width].contiguous()
        query_shape = query.shape
        key_shape = key.shape

        assert self.mrope_section
        if (
            query_shape[0] > self._ASCEND_TRITON_GRID_LIMIT
            and rotary_module.os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") != "1"
        ):
            rotary_module.os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

        q, k = rotary_module.triton_mrope(
            query,
            key,
            self.cos,
            self.sin,
            self.mrope_section,
            self.head_size,
            self.rotary_dim,
            self.mrope_interleaved,
        )
        return q.reshape(query_shape), k.reshape(key_shape)

    forward_triton.round18_api_identity = ROTARY_STATIC_COS_SIN_SPLIT_API
    receipt = {
        "schema": 1,
        "status": "INSTALLED_STATIC_ROTARY_COS_SIN_SPLIT",
        "api": ROTARY_STATIC_COS_SIN_SPLIT_API,
        "class": f"{rotary_cls.__module__}.{rotary_cls.__qualname__}",
        "source_sha256": observed_sha256,
        "original_expression": "cos_sin.chunk(2, dim=-1)",
        "replacement_expression": "rotary_dim/2 model-constant slices",
        "native_triton_mrope_preserved": True,
        "math_changed": False,
        "state_changed": False,
        "provider_changed": False,
        "kv_contract_claimed": False,
    }
    rotary_cls.forward_triton = forward_triton
    setattr(rotary_cls, _ROTARY_MARKER, ROTARY_STATIC_COS_SIN_SPLIT_API)
    setattr(rotary_cls, _ROTARY_RECEIPT, dict(receipt))
    return receipt


def install_gdn_runtime_token_shape_contract(
    *,
    gdn_cls: type,
    ascend_gdn_cls: type,
    ascend_gdn_module: Any,
    expected_source_sha256: str = FROZEN_ASCEND_GDN_SOURCE_SHA256,
) -> dict[str, Any]:
    """Bind GDN output allocation to the runtime ``z`` token dimension.

    The frozen Ascend forward computes ``num_tokens`` from the pre-split
    projection and later reshapes ``z``.  Across a PIECEWISE custom-op
    boundary those two equal dimensions can arrive as different runtime
    SymInt arguments; the bundled AOT wrapper then used the preceding range's
    row count for ``z``.  ``z`` is produced by a projection/view that
    preserves the token axis, so reading its first dimension immediately
    before output allocation is the same eager math while keeping the graph
    symbol local to the tensor it governs.
    """

    installed = gdn_cls.__dict__.get(_GDN_TOKEN_MARKER)
    if installed is not None:
        if installed != GDN_RUNTIME_TOKEN_SHAPE_API:
            raise RuntimeError("another GDN runtime-shape adapter is already installed")
        return dict(gdn_cls.__dict__[_GDN_TOKEN_RECEIPT])

    source_file = inspect.getsourcefile(ascend_gdn_cls)
    if not source_file:
        raise RuntimeError("Ascend GDN source file is unavailable")
    observed_sha256 = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    if observed_sha256 != expected_source_sha256:
        raise RuntimeError(
            "frozen Ascend GDN source identity mismatch: "
            f"expected {expected_source_sha256}, observed {observed_sha256}"
        )
    if gdn_cls.forward is not ascend_gdn_cls.forward:
        raise RuntimeError("target Qwen GDN lacks the frozen Ascend forward binding")
    original_source = inspect.getsource(ascend_gdn_cls.forward)
    if "num_tokens = mixed_qkvz.size(0)" not in original_source:
        raise RuntimeError("frozen Ascend GDN token-source contract is unavailable")

    torch_module = ascend_gdn_module.torch
    triton_module = ascend_gdn_module.triton
    fused_split = ascend_gdn_module.fused_qkvzba_split_reshape_cat
    rearrange_fn = ascend_gdn_module.rearrange

    def forward(self: Any, hidden_states: Any, output: Any = None):
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self._split_ba_for_tp(ba)
            b = b.contiguous()
            a = a.contiguous()
        elif not self.gqa_interleaved_layout:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            ba, _ = self.in_proj_ba(hidden_states)
            b, a = self._split_ba_for_tp(ba)
            b = b.contiguous()
            a = a.contiguous()
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            mixed_qkv, z, b, a = fused_split(
                projected_states_qkvz,
                projected_states_ba,
                triton_module.cdiv(self.num_k_heads, self.tp_size),
                triton_module.cdiv(self.num_v_heads, self.tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )

        # The only semantic edit to the frozen forward: source the runtime
        # token dimension from the exact tensor whose rows feed norm/gate.
        num_tokens = z.size(0)
        core_attn_out = torch_module.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch_module.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
            False,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange_fn(core_attn_out, "... h d -> ... (h d)")
        out, _ = self.out_proj(core_attn_out)
        if output is not None:
            output[:num_tokens] = out
        return out

    forward.round18_api_identity = GDN_RUNTIME_TOKEN_SHAPE_API
    receipt = {
        "schema": 1,
        "status": "INSTALLED_GDN_RUNTIME_TOKEN_SHAPE_CONTRACT",
        "api": GDN_RUNTIME_TOKEN_SHAPE_API,
        "class": f"{gdn_cls.__module__}.{gdn_cls.__qualname__}",
        "source_sha256": observed_sha256,
        "original_token_source": "hidden/projection SymInt before custom-op boundary",
        "replacement_token_source": "z.size(0) immediately before core output allocation",
        "projection_preserves_token_axis": True,
        "math_changed": False,
        "state_changed": False,
        "provider_changed": False,
        "kv_contract_claimed": False,
    }
    gdn_cls.forward = forward
    setattr(gdn_cls, _GDN_TOKEN_MARKER, GDN_RUNTIME_TOKEN_SHAPE_API)
    setattr(gdn_cls, _GDN_TOKEN_RECEIPT, dict(receipt))
    return receipt


__all__ = [
    "FROZEN_ASCEND_GDN_SOURCE_SHA256",
    "FROZEN_ASCEND_ROTARY_SOURCE_SHA256",
    "FROZEN_QWEN_GDN_SOURCE_SHA256",
    "GDN_STATIC_BA_SPLIT_API",
    "GDN_RUNTIME_TOKEN_SHAPE_API",
    "ROTARY_STATIC_COS_SIN_SPLIT_API",
    "install_static_gdn_ba_split",
    "install_gdn_runtime_token_shape_contract",
    "install_static_rotary_cos_sin_split",
]
