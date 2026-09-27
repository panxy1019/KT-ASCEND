"""Default-OFF external observers; no device imports or extra synchronization.

Host durations include any existing provider waits; they are not device costs.
Only the admitted serial model may enable this instrumentation.
"""
import functools
import hashlib
import json
import os
import sys
from pathlib import Path
from .request_cost_ledger import CostLedger


class Proxy:
    def __init__(self, inner, method, call):
        self._inner = inner
        setattr(self, method, call)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Recorder:
    def __init__(self, output, identity, *, max_events=20000):
        self.output, self.identity = Path(output), dict(identity)
        self.max_events = max_events
        self.request_key = None
        if self.output.exists():
            raise FileExistsError('request cost output must be fresh')
        self.ledger = self.request = self.root = self.batch = None
        self.layer = self.chunk = self.group = None
        self.group_sequence = 0

    def wrap(self, original, stage):
        @functools.wraps(original)
        def call(*args, **kwargs):
            if self.ledger is None:
                return original(*args, **kwargs)
            with self.ledger.span(stage, layer=self.layer, chunk=self.chunk,
                                  group=self.group):
                return original(*args, **kwargs)
        return call

    def begin(self, request, mode, start, count, placement_epoch=0):
        if self.batch is not None:
            raise RuntimeError('cost observer batch still active')
        key = (request.rid, placement_epoch, len(getattr(request, 'origin_input_ids', ())))
        if self.request is not None and self.request_key != key:
            raise RuntimeError('cost observer request switched without retirement')
        if self.request is None:
            self.request = request
            self.request_key = key
            self.ledger = CostLedger(dict(self.identity, request_id=request.rid,
                                         placement_epoch=placement_epoch, input_tokens=key[2]),
                                     max_events=self.max_events)
            self.root = self.ledger.span('request')
            self.root.__enter__()
        self.request = request
        self.chunk = (start, count) if mode == 'extend' else None
        self.batch = self.ledger.span('prefill_batch' if mode == 'extend' else 'decode_batch',
                                     chunk=self.chunk)
        self.batch.__enter__()

    def end_batch(self, exc_info=(None, None, None)):
        if self.batch is not None:
            self.batch.__exit__(*exc_info)
            self.batch = None
        self.chunk = None

    def finish(self, reason, exc_info=(None, None, None)):
        self.end_batch(exc_info)
        if self.ledger is None:
            return
        self.root.__exit__(*exc_info)
        result = self.ledger.finish()
        result['termination'] = reason
        with self.output.open('a') as f:
            f.write(json.dumps(result, allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        self.ledger = self.request = self.root = None
        self.request_key = None
        if result['status'] != 'COMPLETE_HOST_ACCOUNTING' and exc_info[0] is None:
            raise RuntimeError('request cost evidence incomplete')


def bind(runner):
    if os.environ.get('ROUND9_REQUEST_COST_ENABLE') != '1':
        return False
    if getattr(runner, '_round9_request_cost', None) is not None:
        raise RuntimeError('request cost already bound')
    args = runner.server_args
    if (args.kt_cpuinfer != 128 or args.kt_num_gpu_experts != 32
            or os.environ.get('ROUND9_DYNAMIC_BOUNDARY_ENABLE') == '1'
            or args.max_running_requests != 1 or not args.disable_overlap_schedule
            or not args.disable_radix_cache or args.tp_size != 1):
        raise RuntimeError('request cost requires serial no-prefix TP1')
    if os.environ.get('ROUND9_TIMELINE_PATH') or os.environ.get('ROUND9_MAINLINE_ENABLE') == '1':
        raise RuntimeError('cost observer must not be stacked with previous profiling')
    manifest = Path(os.environ['ROUND9_REQUEST_COST_MANIFEST'])
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ['ROUND9_REQUEST_COST_MANIFEST_SHA256']:
        raise RuntimeError('cost manifest changed')
    identity = json.loads(raw)
    for key in ('runtime_sha256', 'profile_sha256'):
        if not isinstance(identity.get(key), str) or len(identity[key]) != 64:
            raise RuntimeError('cost manifest missing SHA256 identity')
    expected_files = identity.get('runtime_files', {})
    actual_digest = hashlib.sha256(json.dumps(expected_files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual_digest != identity['runtime_sha256']:
        raise RuntimeError('cost runtime manifest digest mismatch')
    if not {'layerwise.py', 'prefill_runtime.py', 'prefill_binding.py', 'framework_hooks.py', 'request_cost_binding.py', 'request_cost_ledger.py'} <= set(expected_files):
        raise RuntimeError('cost manifest missing runtime bindings')
    for name, digest in expected_files.items():
        if Path(name).name != name or hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('cost runtime source identity mismatch')
    candidate_raw = Path(os.environ['ROUND9_CANDIDATE_PROFILE']).read_bytes()
    if hashlib.sha256(candidate_raw).hexdigest() != identity['profile_sha256']:
        raise RuntimeError('cost candidate profile identity mismatch')
    runtime = runner._round9_prefill
    backend = runtime.transport.backend
    maximum = identity.get('max_events')
    if type(maximum) is not int or not 1 <= maximum <= 20000:
        raise RuntimeError('invalid cost event budget')
    recorder = Recorder('/out/request-cost.jsonl', identity, max_events=maximum)
    runner._round9_request_cost = recorder
    # The real copy facade is slotted/read-only; wrap outside it.
    adapter = backend.adapter
    def copy(layer, moves):
        if recorder.ledger is None:
            return adapter.copy_experts(layer, moves)
        before = adapter.copied_bytes
        with recorder.ledger.span('prepare', layer=layer, chunk=recorder.chunk,
                group=recorder.group, expert_count=len(moves)) as row:
            value = adapter.copy_experts(layer, moves)
            row['logical_bytes'] = adapter.copied_bytes - before
            if row['logical_bytes'] < 0:
                raise RuntimeError('copy counter moved backwards')
            return value
    backend.adapter = Proxy(adapter, 'copy_experts', copy)
    method = backend.method
    backend.method = Proxy(method, 'apply', recorder.wrap(method.apply, 'compute_enqueue'))
    backend.assemble = recorder.wrap(backend.assemble, 'assemble')
    for name, stage in (('synchronize','device_wait'), ('release','release')):
        original = getattr(backend, name)
        def identified(identity, _original=original, _stage=stage):
            previous = recorder.group
            recorder.group = identity.allocation_epoch
            try:
                return recorder.wrap(_original, _stage)(identity)
            finally:
                recorder.group = previous
        setattr(backend, name, identified)
    execute = backend.execute_group
    def execute_group(layer, group, hidden_states, topk_weights, identity):
        previous = recorder.group
        recorder.group = identity.allocation_epoch
        try:
            if recorder.ledger is None:
                return execute(layer,group,hidden_states,topk_weights,identity)
            with recorder.ledger.span('expert_group', layer=layer, chunk=recorder.chunk,
                    group=recorder.group, expert_count=len(group.logical_experts)) as row:
                row['token_rows'] = len(group.rows)
                return execute(layer,group,hidden_states,topk_weights,identity)
        finally:
            recorder.group = previous
    backend.execute_group = execute_group
    apply = runtime.apply
    def apply_layer(layer, hidden, ids, weights):
        previous = recorder.layer
        recorder.layer = layer
        try:
            return recorder.wrap(apply, 'expert_layer')(layer,hidden,ids,weights)
        finally:
            recorder.layer = previous
    runtime.apply = apply_layer
    runtime._rows = recorder.wrap(runtime._rows, 'route_collect')
    from . import layerwise, framework_hooks
    layerwise.group_topk_routes = recorder.wrap(layerwise.group_topk_routes, 'route_group')
    before, after, cancel = (framework_hooks.framework_before,
                            framework_hooks.framework_after, framework_hooks.framework_cancel)
    def fail_record():
        # Never replace the original provider exception with observer I/O errors.
        try:
            recorder.finish('provider_error', sys.exc_info())
        except BaseException:
            pass
    def protect(original):
        @functools.wraps(original)
        def call(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except BaseException:
                fail_record()
                raise
        return call
    def framework_before(scheduler, batch):
        if len(batch.reqs) != 1:
            raise RuntimeError('cost observer requires one request')
        req = batch.reqs[0]
        if not getattr(scheduler, '_round9_cost_error_bound', False):
            for name, stage in (('run_batch','model_forward'), ('process_batch_result','result_processing')):
                setattr(scheduler, name, protect(recorder.wrap(getattr(scheduler, name), stage)))
            scheduler._round9_cost_error_bound = True
        if batch.forward_mode.is_extend() == batch.forward_mode.is_decode():
            raise RuntimeError('unsupported cost observer forward mode')
        mode = 'extend' if batch.forward_mode.is_extend() else 'decode'
        boundary = getattr(scheduler, '_round9_dynamic_boundary', None)
        epoch = boundary.generation if boundary is not None else 0
        recorder.begin(req,mode,getattr(req,'last_extend_prefix_len',None),
                       getattr(req,'extend_input_len',None), epoch)
        value = before(scheduler,batch)
        context = runner._round9_context_reference
        if context.identity is None:
            raise RuntimeError('cost observer missing request allocation identity')
        allocation = context.identity.allocation_epoch
        previous = recorder.ledger.identity.get('request_allocation_epoch')
        if previous is not None and previous != allocation:
            raise RuntimeError('cost request allocation identity changed')
        recorder.ledger.identity['request_allocation_epoch'] = allocation
        return value
    def framework_after(scheduler):
        had_batch = recorder.batch is not None
        value = after(scheduler)
        recorder.end_batch()
        if had_batch and recorder.request is not None and recorder.request.finished():
            reason = getattr(recorder.request, 'finished_reason', None)
            recorder.finish('cancelled' if type(reason).__name__ == 'FINISH_ABORT' else 'finished')
        return value
    def framework_cancel(scheduler, request):
        value = cancel(scheduler,request)
        boundary = getattr(scheduler, '_round9_dynamic_boundary', None)
        epoch = boundary.generation if boundary is not None else 0
        key = (request.rid, epoch, len(getattr(request, 'origin_input_ids', ())))
        if recorder.request is not None and recorder.request_key == key:
            recorder.finish('cancelled')
        return value
    framework_hooks.framework_before = protect(framework_before)
    framework_hooks.framework_after = protect(framework_after)
    framework_hooks.framework_cancel = protect(framework_cancel)
    return True
