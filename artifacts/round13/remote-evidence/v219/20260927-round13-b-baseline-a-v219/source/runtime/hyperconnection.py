"""Eager Round9 GR reference path; not yet integrated into the model runner.

Equations and checkpoint parameter names follow SGLang hyperconnection.py at
4ccff141dbe992794f9da6c3aa23535b4f72000d. No CUDA/Triton/compile dependency.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HyperConnectionConfig:
    hc_count: int = 4
    hidden_size: int = 2560
    params_dtype: torch.dtype = torch.bfloat16
    hc_lowrank: int = 320
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = True
    mtp_hc: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, group_size=None, *, device=None):
        super().__init__()
        if hidden_size <= 0 or eps <= 0:
            raise ValueError('positive hidden size and epsilon required')
        if group_size is not None and (group_size <= 0 or hidden_size % group_size):
            raise ValueError('group_size must divide hidden_size')
        self.weight = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight.weight_loader = self._weight_loader

    @staticmethod
    def _weight_loader(param, loaded_weight):
        if param.shape != loaded_weight.shape:
            raise ValueError('normalization weight shape mismatch')
        with torch.no_grad():
            param.copy_(loaded_weight)

    def forward(self, x):
        if x.shape[-1] != self.weight.numel():
            raise ValueError('normalization width mismatch')
        v = x.float()
        if self.group_size is not None:
            v = v.unflatten(-1, (-1, self.group_size))
        v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + self.variance_epsilon)
        if self.group_size is not None:
            v = v.flatten(-2)
        return (v * (1.0 + self.weight.float())).to(x.dtype)


class GatedResidual(nn.Module):
    def __init__(self, config, use_mix=True, use_combine=True, role=None, *, device=None):
        super().__init__()
        if config.mtp_hc or (role is not None and 'mtp' in role):
            raise ValueError('MTP is outside the Round9 text reference profile')
        if config.hc_count != 4 or config.hidden_size <= 0 or config.hc_lowrank <= 0:
            raise ValueError('four branches and positive dimensions required')
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype
        width = self.hc_count * self.hidden_size
        norm_dim = width if config.hc_per_branch_norm else self.hidden_size
        group = self.hidden_size if config.hc_per_branch_norm else None
        self.hc_norm = GroupedGemmaRMSNorm(norm_dim, config.rms_norm_eps, group, device=device)
        kw = dict(bias=False, device=device, dtype=config.params_dtype)
        if use_mix:
            self.input_mix_weight_down = nn.Linear(width, config.hc_lowrank, **kw)
            self.input_mix_weight_up = nn.Linear(config.hc_lowrank, width, **kw)
        if use_combine:
            self.block_inject_weight = nn.Linear(width, self.hc_count, **kw)

    def _check_hyper(self, x):
        if x.ndim < 2 or x.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError('hyper input width/rank mismatch')

    def mix(self, hyper_input):
        self._check_hyper(hyper_input)
        if not hasattr(self, 'input_mix_weight_down'):
            raise ValueError('mix parameters disabled')
        if hyper_input.numel() == 0:
            return hyper_input.new_empty((*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype), (hyper_input, hyper_input)
        if self.config.hc_per_branch_norm:
            normed = self.hc_norm(hyper_input)
        else:
            normed = self.hc_norm(hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))).flatten(-2)
        low = F.silu(self.input_mix_weight_down(normed) / self.hc_count)
        gates = torch.sigmoid(self.input_mix_weight_up(low))
        mixed = (gates * normed).unflatten(-1, (self.hc_count, self.hidden_size)).mean(-2)
        return mixed.to(self.params_dtype), (hyper_input, normed)

    def combine(self, block_output, residuals):
        if not hasattr(self, 'block_inject_weight'):
            raise ValueError('combine parameters disabled')
        original, normed = residuals
        self._check_hyper(original)
        if normed.shape != original.shape or block_output.shape != (*original.shape[:-1], self.hidden_size):
            raise ValueError('combine shape mismatch')
        if original.numel() == 0:
            return original.to(self.params_dtype)
        gates = 2 * torch.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        injection = block_output.unsqueeze(-2) * gates.unsqueeze(-1)
        return (original.unflatten(-1, (self.hc_count, self.hidden_size)) + injection).flatten(-2).to(self.params_dtype)
