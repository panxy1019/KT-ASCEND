"""Opt-in host-only path receipt. No tensor reads, device events or syncs."""
import functools
import json
import os
from pathlib import Path
import time


def bind(runtime, methods, *, output=None):
    if os.environ.get('ROUND9_HYBRID_PROBE') != '1':
        return False
    path = Path(output or '/out/hybrid-path.jsonl')
    if path.exists():
        raise FileExistsError(path)
    state = {'context': None, 'rows': 0}

    def emit(row):
        if state['rows'] >= 4096:
            raise RuntimeError('hybrid probe evidence bound exceeded')
        with path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        state['rows'] += 1

    original_before = runtime.before
    @functools.wraps(original_before)
    def before(batch, *, placement_epoch):
        state['context'] = None
        selected = original_before(batch, placement_epoch=placement_epoch)
        if batch.forward_mode.is_extend():
            req = batch.reqs[0]
            state['context'] = dict(request_id=req.rid, generation=placement_epoch,
                request_tokens=len(req.origin_input_ids), batch_tokens=req.extend_input_len,
                start=req.last_extend_prefix_len, threshold=runtime.threshold,
                path='LAYERWISE' if selected else 'HYBRID', basis='REQUEST_TOTAL')
            emit(dict(state['context'], stage='SELECTION'))
        return selected
    runtime.before = before

    original_after = runtime.after
    @functools.wraps(original_after)
    def after():
        try:
            return original_after()
        finally:
            state['context'] = None
    runtime.after = after

    def wrap(obj, name, stage, layer):
        original = getattr(obj, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            context = state['context']
            if context is None or context['path'] != 'HYBRID':
                return original(*args, **kwargs)
            started = time.monotonic_ns()
            result = original(*args, **kwargs)
            emit(dict(context, layer=layer, stage=stage, outcome='RETURN',
                      start_ns=started, end_ns=time.monotonic_ns()))
            return result
        setattr(obj, name, call)

    for layer in (0, 3):
        method = methods[layer]
        wrap(method, 'apply', 'ORIGINAL_HYBRID_APPLY', layer)
        wrap(method, '_submit_cpu_forward', 'CPU_SUBMIT', layer)
        wrap(method, '_sync_cpu_forward', 'CPU_RESULT', layer)
        wrap(method.gpu_method, 'apply', 'NPU_RESIDENT_APPLY', layer)
    return True
