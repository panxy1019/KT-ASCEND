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


def bind_expert_timeline(backend, source_identity, model_revision):
    """Observe a qualified transport from its outer Python seams."""
    from . import working_set_timeline as timeline
    if not timeline.enabled():
        return False
    import time
    adapter = backend.adapter
    class AdapterObserver:
        """Observe the qualified read-only copy facade without mutating it."""
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        @property
        def copied_bytes(self):
            return self._inner.copied_bytes

        def copy_experts(self, layer, moves):
            moves = tuple(moves)
            copied_before = self._inner.copied_bytes
            started = time.monotonic_ns()
            result = self._inner.copy_experts(layer, moves)
            timeline.event(
                'expert', 'shadow_prepare',
                clock_domain='HOST_OBSERVED_DEVICE_OPERATION',
                duration_ns=time.monotonic_ns()-started,
                layer=layer, logical_experts=[logical for unused, logical in moves],
                bytes=self._inner.copied_bytes-copied_before,
                source_identity=source_identity, model_revision=model_revision,
                precision='BF16', inference_layout='NZ29')
            return result

    backend.adapter = AdapterObserver(adapter)

    method = backend.method
    class MethodObserver:
        """Observe apply() while preserving the real method object untouched."""
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def apply(self, layer, dispatch_output):
            started = time.monotonic_ns()
            result = self._inner.apply(layer, dispatch_output)
            timeline.event(
                'expert', 'compute_enqueue',
                clock_domain='HOST_OBSERVED_DEVICE_OPERATION',
                duration_ns=time.monotonic_ns()-started)
            return result

    backend.method = MethodObserver(method)

    original_synchronize = backend.synchronize
    def synchronize(identity):
        started = time.monotonic_ns()
        result = original_synchronize(identity)
        timeline.event(
            'expert', 'device_wait',
            clock_domain='HOST_OBSERVED_DEVICE_OPERATION',
            duration_ns=time.monotonic_ns()-started,
            allocation_epoch=identity.allocation_epoch)
        return result
    backend.synchronize = synchronize
    return True


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
    from .round13_placement_runtime import profile_from_environment
    round13_profile = profile_from_environment()
    round13_nonuniform = (round13_profile is not None and
        round13_profile['strategy'] != 'uniform')
    if round13_nonuniform:
        expected = {index for index, count in enumerate(round13_profile['counts'])
                    if count < 512}
        if set(layers) != expected or not layers:
            raise RuntimeError('Round13 nonuniform KTEP layer ownership mismatch')
    elif set(layers) != set(range(48)):
        raise RuntimeError('layerwise requires all48 real expert layers')
    import torch
    torch.npu.synchronize()
    base_peak = max(int(torch.npu.max_memory_allocated()), int(torch.npu.memory_allocated()))
    try:
        allocator_limit_mib = int(os.environ['ROUND9_ALLOCATOR_LIMIT_MIB'])
    except (KeyError, ValueError) as error:
        raise RuntimeError('layerwise runtime requires admitted allocator limit') from error
    if allocator_limit_mib not in (36352, 37376, 43008):
        raise RuntimeError('layerwise allocator limit is outside admitted contracts')
    reservation = layerwise.make_reservation(capacity=32, hidden=2560, intermediate=640,
        max_group_rows=gate['max_group_rows'], allocator_limit_bytes=allocator_limit_mib * 1024**2,
        observed_base_peak_bytes=base_peak,
        observed_workspace_bytes=max(gate['constructor_peak_extra_bytes'], gate['execution_peak_extra_bytes']))
    backend = layerwise.AscendBF16TransportBackend(layers[min(layers)], args.model_path)
    transport = layerwise.LayerwiseTransport(backend, reservation)
    # LayerwiseTransport.preallocate() creates the private shadow method and
    # copy adapter.  Timeline observers bind only after those outer seams
    # exist; binding before preallocation dereferences backend.adapter=None.
    bind_expert_timeline(backend, binding['source_identity'], args.model_path)
    # Persistent private weights already reduce free memory. Reserve remaining
    # transient projection/workspace before the Qwen KV/GDN pool profiler runs.
    runner._round9_prefill_workspace_bytes = reservation.reserve_bytes - reservation.private_bank_bytes
    log = Path('/out/layerwise-events.jsonl')
    if log.exists(): raise FileExistsError('layerwise event log must be fresh')
    def event(item):
        with log.open('a') as stream: stream.write(json.dumps(item, allow_nan=False) + '\n')
    from .prefill_runtime import PrefillRuntime
    runtime = PrefillRuntime(transport, threshold_tokens=config['prefill']['threshold_tokens'], event=event)
    if round13_nonuniform:
        original_before = runtime.before
        def round13_hybrid_only_before(batch, *, placement_epoch):
            if (batch.forward_mode.is_extend() and len(batch.reqs) == 1 and
                    len(batch.reqs[0].origin_input_ids) >= runtime.threshold):
                raise RuntimeError('Round13 nonuniform placement is qualified only for H513 Hybrid <=512')
            return original_before(batch, placement_epoch=placement_epoch)
        runtime.before = round13_hybrid_only_before
    if os.environ.get('ROUND9_HYBRID_PROBE') == '1':
        from .hybrid_probe import bind as bind_hybrid_probe
        bind_hybrid_probe(runtime, methods)
    for index, method in methods.items():
        bind_expert_method(method, index, runtime, StandardCombineInput)
    runner._round9_prefill = runtime
    from .request_cost_binding import bind as bind_request_cost
    bind_request_cost(runner)
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
