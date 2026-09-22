"""Opt-in serial F1 hooks on the real model and scheduler.

The first integration deliberately supports static Q64 only. Future modes must
extend admission explicitly, rather than inheriting this collector's receipt.
"""
import hashlib
import json
import os
from pathlib import Path

from .framework_config import load_framework
from .frequency import payload_digest, validate_frequency
from .prefill_telemetry import PrefillCollector

_MASK_BINDING = None


def prepare_masks(args, wrapper):
    global _MASK_BINDING
    config = load_framework()
    if config is None: return
    if config['resident_capacity'] != 64 or args.kt_num_gpu_experts != 64:
        raise RuntimeError('Q32 physical admission not yet installed in F1 runtime')
    if (args.kt_gpu_experts_ratio is not None or args.kt_expert_placement_strategy != 'uniform' or
            args.tp_size != 1 or args.kt_enable_dynamic_expert_update):
        raise RuntimeError('framework mask requires explicit per-layer serial capacity')
    frequency = config['frequency']
    if frequency is None: return
    if os.environ.get('ROUND9_DYNAMIC_BOUNDARY_ENABLE') == '1':
        raise RuntimeError('frequency plus Dynamic requires separate interaction gate')
    raw = Path(frequency['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != frequency['sha256']:
        raise RuntimeError('frequency manifest changed')
    document = json.loads(raw)
    from .candidate_bank import MANIFEST_SHA
    if document['identity']['model_manifest_sha256'] != MANIFEST_SHA:
        raise RuntimeError('frequency was collected from another model')
    residents = validate_frequency(document, document['identity'], capacity=64)
    import torch
    masks = torch.zeros((48, 512), dtype=torch.bool, device='cpu')
    for layer, row in enumerate(residents): masks[layer, row] = True
    if _MASK_BINDING is None:
        if wrapper._KT_GPU_EXPERTS_MASKS is not None:
            raise RuntimeError('frequency must bind before any KTEP allocation')
        wrapper._KT_GPU_EXPERTS_MASKS = masks
        _MASK_BINDING = frequency['sha256']
        print('ROUND9_FREQUENCY_BOUND', frequency['sha256'], flush=True)
    elif (_MASK_BINDING != frequency['sha256'] or
          not torch.equal(wrapper._KT_GPU_EXPERTS_MASKS, masks)):
        raise RuntimeError('frequency placement changed during model allocation')


class RuntimeCollector:
    def __init__(self, scheduler):
        from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
        args = scheduler.server_args
        if (not args.disable_overlap_schedule or not args.disable_radix_cache or
                not args.disable_cuda_graph or args.max_running_requests != 1 or args.tp_size != 1 or
                os.environ.get('ROUND9_DYNAMIC_BOUNDARY_ENABLE') == '1'):
            raise RuntimeError('calibration requires serial static no-prefix profile')
        self.collector = PrefillCollector()
        self.req = None
        self.pending_req = None
        self.record = None
        self.records = []
        self.path = Path(os.environ['ROUND9_ROUTE_STATS'])
        if self.path.exists(): raise FileExistsError('route output must be fresh per model process')
        layers = set()
        for layer in scheduler.tp_worker.model_runner.model.modules():
            method = getattr(layer, 'quant_method', None)
            if not isinstance(method, KTEPWrapperMethod): continue
            index = method.kt_config.layer_idx
            if index in layers: raise RuntimeError('duplicate calibration layer')
            layers.add(index)
            original = method.apply
            def apply(layer, dispatch_output, _index=index, _original=original):
                if self.collector.batch is not None:
                    topk = dispatch_output.topk_output
                    ids = topk.topk_ids.detach().cpu().tolist()
                    weights = topk.topk_weights.detach().float().cpu().tolist()
                    self.collector.observe(_index, ids, weights)
                return _original(layer, dispatch_output)
            method.apply = apply
        if layers != set(range(48)): raise RuntimeError('calibration requires all48 real KTEP layers')

    def before(self, batch):
        if not batch.forward_mode.is_extend(): return
        if len(batch.reqs) != 1: raise RuntimeError('calibration cannot mix requests')
        req = batch.reqs[0]
        start = req.last_extend_prefix_len
        total = len(req.origin_input_ids)
        if start == 0:
            # A prior chunked request can be aborted between scheduler iterations.
            if self.collector.request is not None:
                reason = getattr(self.pending_req, 'finished_reason', None)
                if reason is None or reason.__class__.__name__ != 'FINISH_ABORT':
                    raise RuntimeError('new request before prior prefill completed or aborted')
                self.collector.abort(self.collector.request[0])
            if len(self.records) >= 64: raise RuntimeError('bounded calibration corpus limit64')
            self.record = dict(request_id=req.rid, tokens=total,
                               input_ids_sha256=payload_digest(req.origin_input_ids))
        self.collector.begin_batch(req.rid, 0, start=start, count=req.extend_input_len, total=total)
        self.req = req
        self.pending_req = req

    def after(self):
        if self.collector.batch is None: return
        # Aborts use FINISH_ABORT; normal length/EOS completion is still usable
        # prefill calibration because all input tokens have already executed.
        reason = getattr(self.req, 'finished_reason', None)
        aborted = reason is not None and reason.__class__.__name__ == 'FINISH_ABORT'
        complete = self.collector.end_batch(aborted=aborted)
        if complete:
            self.records.append(self.record)
            document = dict(schema=1, statistics=self.collector.stats.snapshot(), requests=self.records)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + '\n')
            temp.replace(self.path)
        self.req = None
        if complete or aborted: self.pending_req = None


def framework_before(scheduler, batch):
    config = load_framework()
    if config is None or not config['route_collection']: return
    collector = getattr(scheduler, '_round9_route_collector', None)
    if collector is None:
        collector = scheduler._round9_route_collector = RuntimeCollector(scheduler)
    collector.before(batch)


def framework_after(scheduler):
    collector = getattr(scheduler, '_round9_route_collector', None)
    if collector is not None: collector.after()
