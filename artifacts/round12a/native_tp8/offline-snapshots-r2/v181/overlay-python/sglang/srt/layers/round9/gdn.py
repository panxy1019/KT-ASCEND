"""Stateless eager PyTorch reference for Qwen Gated Delta Network layers."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GDNConfig:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int = 4
    eps: float = 1e-6

    def __post_init__(self):
        for name in (
            "num_key_heads",
            "num_value_heads",
            "key_head_dim",
            "value_head_dim",
            "conv_kernel_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_value_heads % self.num_key_heads != 0:
            raise ValueError("num_value_heads must be divisible by num_key_heads")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    @property
    def query_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def key_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return self.query_dim + self.key_dim + self.value_dim


@dataclass(frozen=True)
class GDNState:
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


def zero_state(
    config: GDNConfig,
    *,
    input_dtype: torch.dtype,
    device: torch.device | str,
) -> GDNState:
    """Create fresh convolution and recurrent state for one request."""
    return GDNState(
        conv_state=torch.zeros(
            config.conv_dim,
            config.conv_kernel_size - 1,
            dtype=input_dtype,
            device=device,
        ),
        recurrent_state=torch.zeros(
            config.num_value_heads,
            config.key_head_dim,
            config.value_head_dim,
            dtype=torch.float32,
            device=device,
        ),
    )


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def gated_delta_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the causal GDN recurrence with an FP32 persistent state."""
    if q.ndim != 3:
        raise ValueError("q must have shape [tokens, key_heads, key_dim]")
    tokens, key_heads, key_dim = q.shape
    _require_shape("k", k, (tokens, key_heads, key_dim))
    if v.ndim != 3 or v.shape[0] != tokens:
        raise ValueError("v must have shape [tokens, value_heads, value_dim]")
    value_heads, value_dim = v.shape[1:]
    if value_heads % key_heads != 0:
        raise ValueError("value heads must be divisible by key heads")
    _require_shape("a", a, (tokens, value_heads))
    _require_shape("b", b, (tokens, value_heads))
    _require_shape("A_log", A_log, (value_heads,))
    _require_shape("dt_bias", dt_bias, (value_heads,))
    if eps <= 0:
        raise ValueError("eps must be positive")

    state_shape = (value_heads, key_dim, value_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, dtype=torch.float32, device=q.device)
    else:
        _require_shape("initial_state", initial_state, state_shape)
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        state = initial_state.to(dtype=torch.float32).clone()

    ratio = value_heads // key_heads
    A_log_fp32 = A_log.float()
    dt_bias_fp32 = dt_bias.float()
    outputs = []
    scale = key_dim**-0.5
    for token in range(tokens):
        query = q[token].float()
        key = k[token].float()
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + eps)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + eps)
        query = query.repeat_interleave(ratio, dim=0)
        key = key.repeat_interleave(ratio, dim=0)

        log_decay = -A_log_fp32.exp() * F.softplus(
            a[token].float() + dt_bias_fp32
        )
        state = state * log_decay.exp()[:, None, None]
        error = v[token].float() - torch.einsum("hk,hkv->hv", key, state)
        delta = b[token].float().sigmoid()[:, None] * error
        state = state + torch.einsum("hk,hv->hkv", key, delta)
        outputs.append(torch.einsum("hk,hkv->hv", query, state) * scale)

    if outputs:
        output = torch.stack(outputs).to(dtype=q.dtype)
    else:
        output = torch.empty(
            (0, value_heads, value_dim), dtype=q.dtype, device=q.device
        )
    return output, state


def sigmoid_gated_rms_norm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply per-head RMS normalization before the frozen sigmoid gate."""
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    _require_shape("z", z, tuple(x.shape))
    _require_shape("weight", weight, (x.shape[-1],))
    if eps <= 0:
        raise ValueError("eps must be positive")
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(
        x_fp32.square().mean(-1, keepdim=True) + eps
    )
    output = normalized * weight.float() * z.float().sigmoid()
    return output.to(dtype=x.dtype)


def _conv_weight_2d(
    conv_weight: torch.Tensor, config: GDNConfig
) -> torch.Tensor:
    if conv_weight.ndim == 3 and conv_weight.shape[1] == 1:
        conv_weight = conv_weight[:, 0, :]
    _require_shape(
        "conv_weight", conv_weight, (config.conv_dim, config.conv_kernel_size)
    )
    return conv_weight


def eager_gdn(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    config: GDNConfig,
    *,
    initial_state: GDNState | None = None,
) -> tuple[torch.Tensor, GDNState]:
    """Run causal convolution, GDN recurrence, and sigmoid gated RMSNorm."""
    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must have shape [tokens, conv_dim]")
    tokens = mixed_qkv.shape[0]
    _require_shape("mixed_qkv", mixed_qkv, (tokens, config.conv_dim))
    _require_shape("a", a, (tokens, config.num_value_heads))
    _require_shape("b", b, (tokens, config.num_value_heads))
    _require_shape(
        "z", z, (tokens, config.num_value_heads, config.value_head_dim)
    )
    _require_shape("A_log", A_log, (config.num_value_heads,))
    _require_shape("dt_bias", dt_bias, (config.num_value_heads,))
    _require_shape("norm_weight", norm_weight, (config.value_head_dim,))
    weight = _conv_weight_2d(conv_weight, config)

    if initial_state is None:
        current = zero_state(
            config, input_dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )
    else:
        _require_shape(
            "initial_state.conv_state",
            initial_state.conv_state,
            (config.conv_dim, config.conv_kernel_size - 1),
        )
        _require_shape(
            "initial_state.recurrent_state",
            initial_state.recurrent_state,
            (
                config.num_value_heads,
                config.key_head_dim,
                config.value_head_dim,
            ),
        )
        if (
            initial_state.conv_state.device != mixed_qkv.device
            or initial_state.recurrent_state.device != mixed_qkv.device
        ):
            raise ValueError("initial state must be on the same device as mixed_qkv")
        if initial_state.conv_state.dtype != mixed_qkv.dtype:
            raise ValueError("conv_state dtype must match mixed_qkv dtype")
        conv_state = initial_state.conv_state.clone()
        recurrent_state = initial_state.recurrent_state.float().clone()
        current = GDNState(conv_state, recurrent_state)

    conv_state = current.conv_state
    convolved = []
    for token in range(tokens):
        window = torch.cat((conv_state, mixed_qkv[token, :, None]), dim=-1)
        value = (window.float() * weight.float()).sum(-1)
        convolved.append(F.silu(value).to(dtype=mixed_qkv.dtype))
        conv_state = window[:, 1:].clone()

    if convolved:
        mixed = torch.stack(convolved)
    else:
        mixed = torch.empty(
            (0, config.conv_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
    q_flat, k_flat, v_flat = torch.split(
        mixed, (config.query_dim, config.key_dim, config.value_dim), dim=-1
    )
    q = q_flat.reshape(tokens, config.num_key_heads, config.key_head_dim)
    k = k_flat.reshape(tokens, config.num_key_heads, config.key_head_dim)
    v = v_flat.reshape(tokens, config.num_value_heads, config.value_head_dim)

    recurrent_output, recurrent_state = gated_delta_recurrence(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=current.recurrent_state,
        eps=config.eps,
    )
    output = sigmoid_gated_rms_norm(
        recurrent_output, z, norm_weight, eps=config.eps
    )
    return output, GDNState(conv_state, recurrent_state)
