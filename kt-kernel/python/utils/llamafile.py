from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

# Use relative imports for package structure
from ..experts_base import BaseMoEWrapper
from .loader import GGUFLoader
from kt_kernel_ext.moe import MOEConfig

try:
    from kt_kernel_ext.moe import MOE

    _HAS_LLAMAFILE_SUPPORT = True
except (ImportError, AttributeError):
    _HAS_LLAMAFILE_SUPPORT = False
    MOE = None

from kt_kernel_ext.kvcache import ggml_type

_LLAMAFILE_QK_K = 256


class LlamafileMoEWrapper(BaseMoEWrapper):
    """
    Llamafile-based MoE wrapper implementation.
    Supports GGUF quantized weights with llamafile backend.

    GGUFLoader is cached **per resolved weight path** (file or directory): multiple MoE layers
    that share one merged GGUF reuse a single mmap; **per-layer split GGUFs** each get their own
    loader.
    """

    _gguf_loaders_by_path: Dict[str, GGUFLoader] = {}

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "LLAMAFILE",
        numa_nodes: Optional[List[int]] = None,
        swiglu_limit: float = 0.0,
        swiglu_alpha: float = 0.0,
    ):
        """
        Initialize Llamafile MoE Wrapper.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            gpu_experts_mask: Boolean mask indicating which experts are on GPU.
                              Shape: [num_experts], dtype: torch.bool.
                              mask[i] = True means expert i is on GPU.
                              If None, all experts are on CPU.
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            weight_path: Path to GGUF weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Not supported for Llamafile backend
            max_deferred_experts_per_token: Number of experts per token to defer. Defaults to 0.
            method: Should be "LLAMAFILE"
        """
        if not _HAS_LLAMAFILE_SUPPORT:
            raise RuntimeError(
                "Llamafile backend not available. kt_kernel_ext was not compiled with Llamafile support.\n"
                "Please recompile with Llamafile enabled."
            )

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"GGUF weight path not found: {weight_path}")

        cache_key = os.path.realpath(weight_path)
        if cache_key not in LlamafileMoEWrapper._gguf_loaders_by_path:
            LlamafileMoEWrapper._gguf_loaders_by_path[cache_key] = GGUFLoader(weight_path)
        self.gguf_loader = LlamafileMoEWrapper._gguf_loaders_by_path[cache_key]

        # Validate TP configuration with QK_K alignment
        QK_K = _LLAMAFILE_QK_K

        qk_aligned_intermediate = moe_intermediate_size % QK_K == 0
        self._f32_tail_tp = False
        if threadpool_count > 1:
            types = [self.gguf_loader.get_undequanted_tensor_and_ggml_type(
                f"blk.{layer_idx}.ffn_{name}_exps.weight")[1] for name in ("gate", "up", "down")]
            self._f32_tail_tp = all(int(t) == int(ggml_type.FP32) for t in types)
            if self._f32_tail_tp and (
                moe_intermediate_size % threadpool_count or
                (moe_intermediate_size // threadpool_count) % 32
            ):
                raise ValueError("F32 TP requires equal 32-aligned partitions")
        if not qk_aligned_intermediate and not self._f32_tail_tp:
            if threadpool_count != 1:
                raise ValueError(
                    f"intermediate_size ({moe_intermediate_size}) is not divisible by QK_K ({QK_K}); "
                    "the LLAMAFILE tail-safe path requires threadpool_count=1"
                )
            if moe_intermediate_size % 32 != 0:
                raise ValueError(
                    f"intermediate_size ({moe_intermediate_size}) must be divisible by the "
                    "LLAMAFILE forward_one m_block (32)"
                )

        print(f"[LlamafileMoEWrapper] Layer {layer_idx} TP configuration:")
        print(f"  intermediate_size: {moe_intermediate_size}")
        print(f"  threadpool_count: {threadpool_count}")
        print(f"  QK_K: {QK_K}")
        if self._f32_tail_tp:
            print("  F32 tail-safe multi-TP path: FP32 partials, final merge cast")
            width = moe_intermediate_size // threadpool_count
            for slot in range(threadpool_count):
                print(f"  TP {slot}: size={width}, offset={slot * width}")
        elif not qk_aligned_intermediate:
            print("  Tail-safe single-TP path: enabled")
            print(f"  TP 0: size={moe_intermediate_size}, offset=0")
        else:
            # Calculate TP splits with QK_K alignment
            num_blocks = moe_intermediate_size // QK_K
            base_blocks = num_blocks // threadpool_count
            extra_blocks = num_blocks % threadpool_count
            if base_blocks == 0:
                valid_tp_counts = list(range(1, num_blocks + 1))
                raise ValueError(
                    f"intermediate_size ({moe_intermediate_size}) is too small for threadpool_count ({threadpool_count}).\n"
                    f"Total blocks: {num_blocks} (intermediate_size / QK_K)\n"
                    f"Cannot distribute to {threadpool_count} TPs (each TP needs at least 1 block).\n"
                    f"Valid threadpool_count values: {valid_tp_counts}"
                )
            print(f"  Total blocks: {num_blocks}")
            print(f"  Base blocks per TP: {base_blocks}")
            print(f"  Extra blocks (distributed to first TPs): {extra_blocks}")
            current_offset = 0
            for tp_id in range(threadpool_count):
                tp_blocks = base_blocks + (1 if tp_id < extra_blocks else 0)
                tp_size = tp_blocks * QK_K
                print(f"  TP {tp_id}: size={tp_size}, offset={current_offset}, blocks={tp_blocks}")
                current_offset += tp_size

        self._swiglu_alpha = float(swiglu_alpha)

        # Initialize base class
        super().__init__(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
            numa_nodes=numa_nodes,
            swiglu_limit=swiglu_limit,
        )

        self.weights_to_keep = None
        self.output_dtype = torch.float32

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        """
        Online quantization is not supported for Llamafile backend.
        Use pre-quantized GGUF weights instead.
        """
        raise NotImplementedError(
            "Llamafile backend does not support online quantization (load_weights_from_tensors).\n"
            "Please use pre-quantized GGUF weights and call load_weights() instead."
        )

    def load_weights(self, physical_to_logical_map_cpu: Optional[torch.Tensor] = None):
        """
        Load weights for this layer from GGUF files and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Optional mapping from physical to logical expert IDs
                                         Shape: [num_experts], dtype: int32
                                         If None, uses identity mapping [0, 1, 2, ..., num_experts-1]
        """
        if not _HAS_LLAMAFILE_SUPPORT:
            raise RuntimeError(
                "Llamafile backend not available. kt_kernel_ext was not compiled with Llamafile support.\n"
                "Please recompile with Llamafile enabled."
            )

        if physical_to_logical_map_cpu is None:
            physical_to_logical_map_cpu = torch.arange(self.num_experts, dtype=torch.int32, device="cpu")
            print(f"  Using default identity mapping for {self.num_experts} experts")
        else:
            if physical_to_logical_map_cpu.device.type != "cpu":
                raise ValueError("physical_to_logical_map_cpu must be a CPU tensor")
            if physical_to_logical_map_cpu.dtype != torch.int32:
                raise ValueError("physical_to_logical_map_cpu must use torch.int32")
            physical_to_logical_map_cpu = physical_to_logical_map_cpu.contiguous().view(-1)
            if physical_to_logical_map_cpu.numel() != self.num_experts:
                raise ValueError(
                    "physical_to_logical_map_cpu must contain exactly "
                    f"{self.num_experts} entries, got {physical_to_logical_map_cpu.numel()}"
                )
            if torch.sort(physical_to_logical_map_cpu).values.tolist() != list(
                range(self.num_experts)
            ):
                raise ValueError(
                    "physical_to_logical_map_cpu must be a permutation of all logical expert IDs"
                )

        base_key = f"blk.{self.layer_idx}"

        # Load quantized tensors from GGUF
        gate_data, gate_type = self.gguf_loader.get_undequanted_tensor_and_ggml_type(f"{base_key}.ffn_gate_exps.weight")

        up_data, up_type = self.gguf_loader.get_undequanted_tensor_and_ggml_type(f"{base_key}.ffn_up_exps.weight")

        down_data, down_type = self.gguf_loader.get_undequanted_tensor_and_ggml_type(f"{base_key}.ffn_down_exps.weight")

        # Keep tensors alive
        self.weights_to_keep = (gate_data, up_data, down_data)

        hidden_type = ggml_type.BF16

        # Configure MoE
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_

        # Llamafile-specific configuration
        moe_config.m_block = 32  # Parallel block size
        # Defensive fallback: chunked_prefill_size <= 0 (e.g. -1 meaning "disabled" in sglang
        # baseline) would otherwise let C++ compute max_possible_qlen() = max(max_len=-1,
        # group_max_len=max(1,-1)=1) = 1, sizing per-NUMA fp32 output buffer
        # (moe-tp.hpp:130 local_output_numa[i]) to a single token. The very first prefill
        # with qlen > 1 then overruns the buffer and corrupts glibc tcache metadata
        # (observed as "malloc(): unaligned tcache chunk detected" Fatal Python error).
        # Clamp to a safe positive value so KT can always alloc a fp32 buffer at least as
        # large as the per-call qlen the caller will pass.
        _effective_chunk = int(self.chunked_prefill_size)
        if _effective_chunk <= 0:
            _effective_chunk = 2048
            print(
                f"[LlamafileMoEWrapper] chunked_prefill_size={self.chunked_prefill_size} "
                f"<= 0 is unsafe for KT MoE C++ buffer sizing; falling back to "
                f"{_effective_chunk}."
            )
        # Multi-TP F32 has explicit tail tiles. Preserve the existing single-TP
        # 1408 forward_one path so C32 numerical behavior remains unchanged.
        # Block-quantized paths still require their existing QK_K alignment.
        moe_config.group_min_len = (
            10
            if self._f32_tail_tp or self.moe_intermediate_size % _LLAMAFILE_QK_K == 0
            else _effective_chunk + 1
        )
        moe_config.max_len = _effective_chunk
        moe_config.group_max_len = _effective_chunk
        moe_config.swiglu_limit = self.swiglu_limit
        moe_config.swiglu_alpha = self._swiglu_alpha

        # Set weight pointers
        moe_config.gate_proj = gate_data.data_ptr()
        moe_config.up_proj = up_data.data_ptr()
        moe_config.down_proj = down_data.data_ptr()

        # Set quantization types
        moe_config.gate_type = gate_type
        moe_config.up_type = up_type
        moe_config.down_type = down_type
        moe_config.hidden_type = hidden_type
        moe_config.output_type = ggml_type.FP32
        # Ascend's unquantized MoE exposes BF16 boundaries after gate/up GMM,
        # fused SwiGLU, and down GMM. Keep llamafile's F32 accumulation while
        # matching those values at the CPU/NPU merge boundary.
        moe_config.bf16_numerical_compat = True

        # Create MoE module
        self.moe = MOE(moe_config)

        # Load weights
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()

        # Drop original weights after loading
        self.weights_to_keep = None
