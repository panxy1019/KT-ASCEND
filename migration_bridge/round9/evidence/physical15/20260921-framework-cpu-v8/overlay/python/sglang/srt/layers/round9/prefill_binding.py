"""Opt-in real model/scheduler binding for bounded layerwise BF16 transport."""
import hashlib
import json
import os
from pathlib import Path

from .framework_config import load_framework


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_expert_method(method, index, runtime, combine_type):
    original = method.apply
    def apply(layer, dispatch_output):
        if not runtime.active:
            return original(layer=layer, dispatch_output=dispatch_output)
        topk = dispatch_output.topk_output
        value = runtime.apply(index, dispatch_output.hidden_states, topk.topk_ids, topk.topk_weights)
        return combine_type(hidden_states=value)
    method.apply = apply


def prefill_model_ready(runner):
    config = load_framework()
    if config is None or config.get('prefill') is None: return
    if getattr(runner, '_round9_prefill', None) is not None:
        raise RuntimeError('layerwise runtime may bind only once per model')
    args = runner.server_args
    if (not args.disable_overlap_schedule or not args.disable_radix_cache or
            not args.disable_cuda_graph or args.max_running_requests != 1 or
            args.tp_size != 1 or args.kt_gpu_prefill_token_threshold or
            args.kt_enable_dynamic_expert_update or
            getattr(args, 'speculative_algorithm', None) is not None or
            args.chunked_prefill_size != 32 or args.max_prefill_tokens != 32):
        raise RuntimeError('layerwise requires serial no-prefix bounded32 scheduler')
    dynamic_enabled = os.environ.get('ROUND9_DYNAMIC_BOUNDARY_ENABLE') == '1'
    if dynamic_enabled != config['prefill'].get('update_residents', False):
        raise RuntimeError('prefill-end update requires matching Dynamic scheduler binding')
    from .candidate_bank import bind_verified_banks
    binding = bind_verified_banks(os.environ['ROUND9_CANDIDATE_PROFILE'],
                                  args.kt_weight_path, args.model_path)
    import kt_kernel_ext
    from . import layerwise, dynamic_adapter, expert_source
    from .layerwise_gate import require_layerwise_device_gate
    receipt = config['prefill']['device_gate']
    gate = require_layerwise_device_gate(receipt['path'], receipt['sha256'],
        source_identity=binding['source_identity'], native_sha256=_sha(kt_kernel_ext.__file__),
        source_sha256={module.__name__.rsplit('.', 1)[-1] + '.py': _sha(module.__file__)
                       for module in (layerwise, dynamic_adapter, expert_source)})
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
    layers, methods = {}, {}
    for owner in runner.model.modules():
        method = getattr(owner, 'quant_method', None)
        if not isinstance(method, KTEPWrapperMethod): continue
        index = method.kt_config.layer_idx
        if index in methods: raise RuntimeError('duplicate layerwise KTEP layer')
        layers[index], methods[index] = owner, method
    if set(layers) != set(range(48)): raise RuntimeError('layerwise requires all48 real expert layers')
    import torch
    torch.npu.synchronize()
    base_peak = max(int(torch.npu.max_memory_allocated()), int(torch.npu.memory_allocated()))
    reservation = layerwise.make_reservation(capacity=32, hidden=2560, intermediate=640,
        max_group_rows=gate['max_group_rows'], allocator_limit_bytes=43008 * 1024**2,
        observed_base_peak_bytes=base_peak,
        observed_workspace_bytes=max(gate['constructor_peak_extra_bytes'], gate['execution_peak_extra_bytes']))
    backend = layerwise.AscendBF16TransportBackend(layers[0], args.model_path)
    transport = layerwise.LayerwiseTransport(backend, reservation)
    # Persistent private weights already reduce free memory. Reserve remaining
    # transient projection/workspace before the Qwen KV/GDN pool profiler runs.
    runner._round9_prefill_workspace_bytes = reservation.reserve_bytes - reservation.private_bank_bytes
    log = Path('/out/layerwise-events.jsonl')
    if log.exists(): raise FileExistsError('layerwise event log must be fresh')
    def event(item):
        with log.open('a') as stream: stream.write(json.dumps(item, allow_nan=False) + '\n')
    from .prefill_runtime import PrefillRuntime
    runtime = PrefillRuntime(transport, threshold_tokens=config['prefill']['threshold_tokens'], event=event)
    for index, method in methods.items():
        bind_expert_method(method, index, runtime, StandardCombineInput)
    runner._round9_prefill = runtime
    event(dict(status='PRIVATE_TRANSPORT_BOUND_BEFORE_KV_POOL', device_gate_sha256=receipt['sha256'],
               capacity=32, max_group_rows=32, reserve_bytes=reservation.reserve_bytes,
               base_peak_bytes=base_peak, constructor_peak_extra_bytes=backend.measured_preallocation_peak_bytes,
               qualification='MODULE_GATE_ONLY_FULL_MODEL_PENDING'))


def prefill_before(scheduler, batch):
    runtime = getattr(scheduler.tp_worker.model_runner, '_round9_prefill', None)
    if runtime is None: raise RuntimeError('enabled layerwise model binding is missing')
    boundary = getattr(scheduler, '_round9_dynamic_boundary', None)
    runtime.before(batch, placement_epoch=boundary.generation if boundary is not None else 0)


def prefill_after(scheduler):
    runtime = getattr(scheduler.tp_worker.model_runner, '_round9_prefill', None)
    if runtime is None: raise RuntimeError('enabled layerwise model binding is missing')
    statistics = runtime.after()
    if statistics is not None:
        boundary = getattr(scheduler, '_round9_dynamic_boundary', None)
        if boundary is not None:
            from .prefill_end_boundary import PrefillEndBoundary
            if not isinstance(boundary, PrefillEndBoundary):
                raise RuntimeError('layerwise updates require a live-prefill boundary')
            boundary.prefill_complete(statistics)
    return statistics
