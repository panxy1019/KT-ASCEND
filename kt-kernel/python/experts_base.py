# Base classes for MoE CPU inference operations
# SPDX-License-Identifier: Apache-2.0

"""
Base infrastructure for CPU-based MoE inference.

This module contains base classes and utilities shared across all backend implementations.
"""

from __future__ import annotations

import hashlib
import json
import torch
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import os
import ctypes

from kt_kernel import kt_kernel_ext


def _cpu_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact CPU tensor storage for default-off task diagnostics."""
    value = tensor.detach().contiguous()
    if value.device.type != "cpu":
        raise ValueError("CPU task trace received a non-CPU tensor")
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _write_cpu_task_trace(payload: dict) -> None:
    """Append a compact JSONL event when explicitly armed by the operator."""
    trace_path = os.environ.get("KT_DEBUG_CPU_TASK_TRACE_FILE")
    if not trace_path:
        return
    with open(trace_path, "a", encoding="utf-8") as writer:
        writer.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _allocate_cpu_expert_mask(num_experts: int, *, zero: bool) -> torch.Tensor:
    """Prefer pinned storage, but remain usable in a CPU-only torch runtime."""
    factory = torch.zeros if zero else torch.empty
    try:
        return factory(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)
    except RuntimeError as error:
        if "pinned memory allocator" not in str(error) and "pin_memory=True" not in str(error):
            raise
        return factory(num_experts, dtype=torch.bool, device="cpu")


def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
) -> torch.Tensor:
    """
    Generate GPU experts masks based on activation frequency.

    Selects the top `num_gpu_experts` experts with highest activation frequency
    across all layers to be placed on GPU.

    Args:
        activation_freq: Activation frequency table of shape (num_layers, num_experts).
                         Higher values indicate more frequently activated experts.
        num_gpu_experts: Total number of experts to place on GPU across all layers.

    Returns:
        gpu_experts_masks: Boolean mask of shape (num_layers, num_experts) on CPU.
                           True means the expert should be on GPU.

    Example:
        >>> activation_freq = torch.tensor([
        ...     [0.1, 0.5, 0.3, 0.8],  # layer 0
        ...     [0.2, 0.4, 0.9, 0.1],  # layer 1
        ... ])
        >>> masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=3)
        >>> # Top 3: layer0-expert3 (0.8), layer1-expert2 (0.9), layer0-expert1 (0.5)
        >>> masks
        tensor([[False,  True, False,  True],
                [False, False,  True, False]])
    """
    num_layers, num_experts_per_layer = activation_freq.shape
    total_experts = num_layers * num_experts_per_layer

    # Clamp num_gpu_experts to valid range
    num_gpu_experts = min(num_gpu_experts, total_experts)
    num_gpu_experts = max(num_gpu_experts, 0)

    if num_gpu_experts == 0:
        return torch.zeros(num_layers, num_experts_per_layer, dtype=torch.bool, device="cpu")

    # Flatten and find top-k indices
    flat_freq = activation_freq.view(-1).to(device="cpu")
    _, top_indices = torch.topk(flat_freq, k=num_gpu_experts, largest=True, sorted=False)

    # Create mask
    gpu_experts_masks = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    gpu_experts_masks[top_indices] = True

    # Reshape to (num_layers, num_experts)
    gpu_experts_masks = gpu_experts_masks.view(num_layers, num_experts_per_layer)

    return gpu_experts_masks


class KExpertsCPUBuffer:
    """
    CPU buffer management for expert computation.

    Manages pinned memory buffers for efficient GPU-CPU data transfer.
    """

    capture_bs: List = list()
    capture_buffers: Dict = dict()
    temp_bs: int = 0
    temp_buffer: tuple = tuple()
    buffer_depth: int = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        hidden_size = hidden_states.shape[-1]
        batch_size = hidden_states.shape[0]

        pin_memory = True

        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size == cls.temp_bs:
            return cls.temp_buffer

        input_tensor_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        immediate_experts_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        deferred_experts_ids_cpu = [
            torch.full((batch_size, num_experts_per_tok), -1, device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        bsz_tensor_cpu = [
            torch.full((1,), batch_size, device="cpu", dtype=torch.int32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        )
        if batch_size in cls.capture_bs:
            cls.capture_buffers[batch_size] = cur_buffer
        cls.temp_bs = batch_size
        cls.temp_buffer = cur_buffer
        return cur_buffer


class _MoEBase:
    """
    Shared base class for inference and SFT MoE wrappers.

    Provides:
    - CPUInfer singleton management
    - Basic configuration validation

    This class is shared between BaseMoEWrapper (inference) and BaseSFTMoEWrapper (SFT).
    """

    _cpu_infer_instance = None
    _cpu_infer_instances = {}

    @classmethod
    def _get_cpu_infer(
        cls,
        cpuinfer_threads: int,
        threadpool_count: int,
        numa_nodes=None,
        isolation_key=None,
    ):
        """
        Get or create the CPUInfer singleton instance.

        Args:
            cpuinfer_threads: Total number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            numa_nodes: Explicit list of NUMA node IDs. If None, defaults to sequential.
            isolation_key: Optional debug-only cache discriminator.  When set,
                callers receive a CPUInfer dedicated to that wrapper.

        Returns:
            CPUInfer singleton instance
        """
        if numa_nodes is not None:
            if len(numa_nodes) != threadpool_count:
                raise ValueError(
                    f"numa_nodes length ({len(numa_nodes)}) must match "
                    f"threadpool_count ({threadpool_count})"
                )
            subpool_numa_map = list(numa_nodes)
        else:
            subpool_numa_map = list(range(threadpool_count))
        subpool_thread_count = [
            cpuinfer_threads // threadpool_count + (1 if i < cpuinfer_threads % threadpool_count else 0)
            for i in range(threadpool_count)
        ]
        config_key = (tuple(subpool_numa_map), tuple(subpool_thread_count))
        if isolation_key is not None:
            config_key = (*config_key, "debug-wrapper", isolation_key)

        if config_key not in cls._cpu_infer_instances:
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            cls._cpu_infer_instances[config_key] = kt_kernel_ext.CPUInfer(worker_config)

        cls._cpu_infer_instance = cls._cpu_infer_instances[config_key]
        return cls._cpu_infer_instance

    def cpu_runtime_diagnostics(self) -> dict:
        """Return read-only worker-pool, affinity, and NUMA visibility data."""
        config = self.cpu_infer.worker_pool_config()
        try:
            process_affinity = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            process_affinity = []

        available_numa_nodes = []
        try:
            with open("/sys/devices/system/node/online", "r", encoding="utf-8") as file:
                for part in file.read().strip().split(","):
                    bounds = [int(value) for value in part.split("-")]
                    available_numa_nodes.extend(range(bounds[0], bounds[-1] + 1))
        except (FileNotFoundError, OSError, ValueError):
            pass

        return {
            "subpool_count": int(config.subpool_count),
            "subpool_numa_map": list(config.subpool_numa_map),
            "subpool_thread_count": list(config.subpool_thread_count),
            "process_cpu_affinity": process_affinity,
            "available_numa_nodes": available_numa_nodes,
        }

    @staticmethod
    def _validate_base_config(
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts_per_tok: int,
    ) -> None:
        """
        Validate basic configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if moe_intermediate_size <= 0:
            raise ValueError(f"moe_intermediate_size must be positive, got {moe_intermediate_size}")
        if num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok must be positive, got {num_experts_per_tok}")
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed " f"num_experts ({num_experts})"
            )


class BaseMoEWrapper(_MoEBase, ABC):
    """
    Base class for MoE CPU inference operations.
    Provides common functionality for all backend implementations.
    """

    _layer_has_pending_deferred: Dict[int, bool] = {}
    _cpu_task_trace_sequence = 0

    @classmethod
    def _next_cpu_task_trace_sequence(cls) -> int:
        cls._cpu_task_trace_sequence += 1
        return cls._cpu_task_trace_sequence

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
        method: str = "AMXINT4",
        numa_nodes: Optional[List[int]] = None,
        swiglu_limit: float = 0.0,
    ):
        """
        Initialize base MoE Wrapper.

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
            threadpool_count: Number of NUMA subpools
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer on this layer. Defaults to 0 (no defer).
            method: Backend method string
            numa_nodes: Explicit list of NUMA node IDs for subpool mapping.
                        If None, defaults to [0, 1, ..., threadpool_count-1].
        """
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        # Process gpu_experts_mask: convert to bool tensor on CPU, pinned memory for async copy
        # This mask is shared between C and Python (C uses uint8_t*), both can read/write it
        if gpu_experts_mask is None:
            # No GPU experts - all experts on CPU
            self.gpu_experts_mask = _allocate_cpu_expert_mask(num_experts, zero=True)
        else:
            # Create a new pinned tensor and copy data into it
            self.gpu_experts_mask = _allocate_cpu_expert_mask(num_experts, zero=False)
            self.gpu_experts_mask.copy_(gpu_experts_mask)

        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())

        # GPU copy for mask operations in forward pass (e.g., mask_cpu_expert_ids)
        # This will be lazily initialized when needed
        self._gpu_experts_mask_gpu: Optional[torch.Tensor] = None
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpu_save = cpu_save
        self.max_deferred_experts_per_token = (
            int(max_deferred_experts_per_token) if max_deferred_experts_per_token is not None else 0
        )

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        self.method = method
        # V4-Flash 2604B SwiGLU clamp limit; 0.0 = disabled. NativeMoEWrapper
        # (MXFP4 path) reads this in load_weights() and writes it into
        # MOEConfig.swiglu_limit. Other backends ignore it (C++ act_fn skips
        # the clamp branch when limit==0). Origin: kt-sglang 耦合.
        self.swiglu_limit = float(swiglu_limit)

        # Default behavior is the historical shared worker pool.  This
        # diagnostic-only switch isolates each wrapper's queue without
        # changing arithmetic, placement, or normal production behavior.
        isolation_key = (
            self.layer_idx
            if os.environ.get("KT_DEBUG_PER_WRAPPER_CPUINFER") == "1"
            else None
        )
        self.cpu_infer = self._get_cpu_infer(
            cpuinfer_threads,
            threadpool_count,
            numa_nodes=numa_nodes,
            isolation_key=isolation_key,
        )

        # Backend-specific initialization happens in subclasses
        self.moe = None

    @abstractmethod
    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        """
        Load and quantize weights from BF16/FP16 tensors (online quantization).

        Args:
            gate_proj: Gate projection weights [num_experts, intermediate_size, hidden_size]
            up_proj: Up projection weights [num_experts, intermediate_size, hidden_size]
            down_proj: Down projection weights [num_experts, hidden_size, intermediate_size]
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

    def select_deferred_experts(
        self,
        expert_ids: torch.Tensor,
        expert_scores: torch.Tensor,
        protected_k: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, topk = expert_ids.shape
        device = expert_ids.device

        protected_k = max(0, min(int(protected_k), topk))
        if protected_k == 0:
            deferred_ids = expert_ids.clone()
            immediate_ids = torch.full_like(expert_ids, -1)
            return immediate_ids, deferred_ids

        topk_result = torch.topk(expert_scores, k=protected_k, dim=-1, largest=True, sorted=False)
        protected_indices = topk_result.indices
        protected_ids = torch.gather(expert_ids, -1, protected_indices)

        protected_flag = torch.zeros((self.num_experts,), dtype=torch.int32, device=device)
        protected_flag.scatter_(0, protected_ids.reshape(-1), 1)

        protected_mask_flat = torch.gather(protected_flag, 0, expert_ids.reshape(-1)).ne(0)
        protected_mask = protected_mask_flat.view(batch, topk)

        immediate_ids = expert_ids.clone().masked_fill(~protected_mask, -1)
        deferred_ids = expert_ids.clone().masked_fill(protected_mask, -1)

        return immediate_ids, deferred_ids

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ):
        """
        Submit forward inference task to CPU (non-blocking).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization
        """
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat_hidden_states.shape[0]

        if hidden_states.device.type == "npu" and self.max_deferred_experts_per_token == 0:
            if getattr(self, "_npu_pending_forward", None) is not None:
                raise RuntimeError("an Ascend CPU expert forward is already pending on this wrapper")
            # Keep the host tensors alive for the complete CPUInfer lifetime.
            # The transfers themselves must remain framework-managed: raw ACL
            # pointer memcpy bypasses torch_npu stream dependencies and was the
            # source of P2 same-path nondeterminism.
            import acl

            input_npu = flat_hidden_states.to(dtype=torch.bfloat16).contiguous()
            ids_npu = topk_ids.to(dtype=torch.int64).contiguous()
            weights_npu = topk_weights.to(dtype=torch.float32).contiguous()
            status = acl.rt.synchronize_stream(cuda_stream)
            if status != 0:
                raise RuntimeError(f"acl.rt.synchronize_stream failed with status {status}")
            input_cpu = torch.empty_like(input_npu, device="cpu", pin_memory=True)
            ids_cpu = torch.empty_like(ids_npu, device="cpu", pin_memory=True)
            weights_cpu = torch.empty_like(weights_npu, device="cpu", pin_memory=True)
            for destination, source in (
                (input_cpu, input_npu),
                (ids_cpu, ids_npu),
                (weights_cpu, weights_npu),
            ):
                destination.copy_(source, non_blocking=False)
            if os.environ.get("KT_DEBUG_NPU_CPU_INPUT_CLONE") == "1":
                # Diagnostic-only: ensure the CPU task consumes buffers that
                # are distinct from raw ACL D2H destinations.
                def _clone_pinned(value: torch.Tensor) -> torch.Tensor:
                    clone = torch.empty_like(value, device="cpu", pin_memory=True)
                    clone.copy_(value)
                    return clone

                input_cpu = _clone_pinned(input_cpu)
                ids_cpu = _clone_pinned(ids_cpu)
                weights_cpu = _clone_pinned(weights_cpu)
            output_cpu = torch.zeros(
                input_cpu.shape,
                dtype=getattr(self, "output_dtype", input_cpu.dtype),
                device="cpu",
                pin_memory=True,
            )
            batch_cpu = torch.tensor([batch_size], dtype=torch.int32, device="cpu")
            incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
            trace = None
            if os.environ.get("KT_DEBUG_CPU_TASK_TRACE_FILE"):
                trace = {
                    "schema_version": 1,
                    "sequence": self._next_cpu_task_trace_sequence(),
                    "layer": int(self.layer_idx),
                    "wrapper_id": id(self),
                    "cpuinfer_id": id(self.cpu_infer),
                    "batch_size": int(batch_size),
                    "input": {
                        "shape": list(input_cpu.shape),
                        "dtype": str(input_cpu.dtype),
                        "data_ptr": int(input_cpu.data_ptr()),
                        "sha256": _cpu_tensor_sha256(input_cpu),
                    },
                    "topk_ids": {
                        "shape": list(ids_cpu.shape),
                        "data_ptr": int(ids_cpu.data_ptr()),
                        "sha256": _cpu_tensor_sha256(ids_cpu),
                    },
                    "topk_weights": {
                        "shape": list(weights_cpu.shape),
                        "data_ptr": int(weights_cpu.data_ptr()),
                        "sha256": _cpu_tensor_sha256(weights_cpu),
                    },
                    "output": {
                        "shape": list(output_cpu.shape),
                        "dtype": str(output_cpu.dtype),
                        "data_ptr": int(output_cpu.data_ptr()),
                        "pre_task_sha256": _cpu_tensor_sha256(output_cpu),
                    },
                    "incremental": bool(incremental),
                }
                _write_cpu_task_trace({"event": "submit", **trace})
            task = self.moe.forward_task(
                batch_cpu.data_ptr(),
                ids_cpu.size(-1),
                ids_cpu.data_ptr(),
                weights_cpu.data_ptr(),
                input_cpu.data_ptr(),
                output_cpu.data_ptr(),
                incremental,
            )
            self._npu_pending_forward = (
                input_cpu,
                ids_cpu,
                weights_cpu,
                output_cpu,
                batch_cpu,
                trace,
            )
            self.cpu_infer.submit(task)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
            return

        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth

        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        topk_ids_long = topk_ids.to(torch.long)
        immediate_ids: torch.Tensor
        deferred_ids: Optional[torch.Tensor]
        if self.max_deferred_experts_per_token > 0:
            protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token

            immediate_ids, deferred_ids = self.select_deferred_experts(topk_ids_long, topk_weights, protected_k)
        else:
            immediate_ids = topk_ids_long
            deferred_ids = None

        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
        immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)

        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        self.cpu_infer.submit_with_cuda_stream(
            cuda_stream,
            self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                immediate_experts_ids_cpu[current_slot].size(-1),
                immediate_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[current_slot].data_ptr(),
                incremental,
            ),
        )

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        if deferred_ids is not None:
            deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(
                cuda_stream,
                self.moe.forward_task(
                    bsz_slot_tensor.data_ptr(),
                    deferred_experts_ids_cpu[current_slot].size(-1),
                    deferred_experts_ids_cpu[current_slot].data_ptr(),
                    weights_cpu[current_slot].data_ptr(),
                    input_tensor_cpu[current_slot].data_ptr(),
                    output_cpu[next_slot].data_ptr(),
                    False,
                ),
            )
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream) -> torch.Tensor:
        """
        Synchronize and retrieve forward inference results.

        Args:
            hidden_states: Original input hidden states (for getting buffer)
            cuda_stream: CUDA stream for synchronization

        Returns:
            output_gpu: Output tensor on GPU
        """
        if hidden_states.device.type == "npu" and self.max_deferred_experts_per_token == 0:
            pending = getattr(self, "_npu_pending_forward", None)
            if pending is None:
                raise RuntimeError("no Ascend CPU expert forward is pending on this wrapper")
            self.cpu_infer.sync()
            output_cpu = pending[3]
            trace = pending[5]
            if trace is not None:
                _write_cpu_task_trace(
                    {
                        "event": "sync",
                        **trace,
                        "output": {
                            **trace["output"],
                            "post_task_sha256": _cpu_tensor_sha256(output_cpu),
                            "finite": bool(torch.isfinite(output_cpu).all().item()),
                        },
                    }
                )
            output_for_h2d = output_cpu
            if os.environ.get("KT_DEBUG_NPU_CPU_OUTPUT_CLONE") == "1":
                # Diagnostic-only: decouple raw ACL H2D from the CPUInfer
                # task's pinned output allocation.  Keep the clone alive until
                # the next (already synchronized) forward on this wrapper.
                output_for_h2d = torch.empty_like(
                    output_cpu,
                    device="cpu",
                    pin_memory=True,
                )
                output_for_h2d.copy_(output_cpu)
                self._npu_debug_h2d_source = output_for_h2d
            output_npu = torch.empty(
                output_for_h2d.shape,
                dtype=output_for_h2d.dtype,
                device=hidden_states.device,
            )
            output_npu.copy_(output_for_h2d, non_blocking=False)
            self._npu_pending_forward = None
            return output_npu.view_as(hidden_states)

        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            _input_tensor_cpu,
            _immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            _weights_cpu,
            output_cpu,
            _bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
        self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)
        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
        return output_gpu[current_slot]

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ) -> torch.Tensor:
        """
        Execute forward inference synchronously (submit + sync).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization

        Returns:
            Output tensor on GPU
        """
        self.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        return self.sync_forward(hidden_states, cuda_stream)

    @staticmethod
    def set_capture_batch_sizes(capture_bs: List[int]):
        """
        Set batch sizes to capture and cache buffers for.

        This allows pre-allocation of CPU buffers for specific batch sizes,
        improving performance by avoiding buffer re-allocation during inference.

        Args:
            capture_bs: List of batch sizes to capture (e.g., [1, 2, 4, 8, 16])

        Example:
            >>> BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])
        """
        KExpertsCPUBuffer.capture_bs = capture_bs

    @staticmethod
    def get_capture_batch_sizes() -> List[int]:
        """
        Get currently configured capture batch sizes.

        Returns:
            List of batch sizes that are being captured
        """
        return KExpertsCPUBuffer.capture_bs

    @staticmethod
    def clear_buffer_cache():
        """
        Clear all cached buffers.

        This frees up memory by clearing the buffer cache. Useful when you want
        to reset the buffer state or free memory.
        """
        KExpertsCPUBuffer.capture_buffers.clear()
        KExpertsCPUBuffer.temp_bs = 0
        KExpertsCPUBuffer.temp_buffer = tuple()
