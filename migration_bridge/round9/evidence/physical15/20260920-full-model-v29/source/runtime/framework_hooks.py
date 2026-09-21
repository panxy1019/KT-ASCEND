"""Opt-in framework hooks on the real model and scheduler."""
import hashlib
import json
import os
from pathlib import Path

from .framework_config import load_framework
from .frequency import payload_digest, validate_frequency
from .prefill_telemetry import PrefillCollector

def prepare_masks(args, wrapper):
    config = load_framework()
    if config is None: return 64
    capacity = config['resident_capacity']
    if args.kt_num_gpu_experts != capacity:
        raise RuntimeError('framework resident capacity differs from server arguments')
    if (args.kt_gpu_experts_ratio is not None or args.kt_expert_placement_strategy != 'uniform' or
            args.tp_size != 1 or args.kt_enable_dynamic_expert_update):
        raise RuntimeError('framework mask requires explicit per-layer serial capacity')
    frequency = config['frequency']
    if frequency is None: return capacity
    if os.environ.get('ROUND9_DYNAMIC_BOUNDARY_ENABLE') == '1':
        raise RuntimeError('frequency plus Dynamic requires separate interaction gate')
    raw = Path(frequency['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != frequency['sha256']:
        raise RuntimeError('frequency manifest changed')
    document = json.loads(raw)
    from .candidate_bank import MANIFEST_SHA
    if document['identity']['model_manifest_sha256'] != MANIFEST_SHA:
        raise RuntimeError('frequency was collected from another model')
    residents = validate_frequency(document, document['identity'], capacity=capacity)
    import torch
    masks = torch.zeros((48, 512), dtype=torch.bool, device='cpu')
    for layer, row in enumerate(residents): masks[layer, row] = True
    binding = getattr(wrapper, '_ROUND9_FREQUENCY_BINDING', None)
    if binding is None:
        if wrapper._KT_GPU_EXPERTS_MASKS is not None:
            raise RuntimeError('frequency must bind before any KTEP allocation')
        wrapper._KT_GPU_EXPERTS_MASKS = masks
        wrapper._ROUND9_FREQUENCY_BINDING = frequency['sha256']
        print('ROUND9_FREQUENCY_BOUND', frequency['sha256'], flush=True)
    elif (binding != frequency['sha256'] or wrapper._KT_GPU_EXPERTS_MASKS is None or
          not torch.equal(wrapper._KT_GPU_EXPERTS_MASKS, masks)):
        raise RuntimeError('frequency placement changed during model allocation')
    return capacity


def validate_q32_model_gate(resident_capacity, candidate, bank_identity):
    """Bind Q32 allocation to its distinct device receipt before model binding."""
    if resident_capacity != 32: return
    import kt_kernel_ext
    native_path = Path(kt_kernel_ext.__file__).resolve()
    native_sha256 = hashlib.sha256(native_path.read_bytes()).hexdigest()
    from .q32_gate import require_q32_device_gate
    require_q32_device_gate(
        os.environ.get('ROUND9_DYNAMIC_DEVICE_GATE'),
        os.environ.get('ROUND9_DYNAMIC_DEVICE_GATE_SHA256'),
        storage=candidate.storage,
        workers=candidate.workers,
        source_identity=bank_identity['source_identity'],
        layer0_sha256=bank_identity['layer0_sha256'],
        native_sha256=native_sha256,
    )


def _values(value):
    return value.detach().cpu().tolist()


def _require_tensor(value, *, dtype, device, length=None):
    if value is None or str(value.dtype) != dtype or getattr(value.device, 'type', None) != device:
        raise RuntimeError(f'Q32 requires {dtype} tensor on {device}')
    if length is not None and (len(value.shape) != 1 or value.shape[0] != length):
        raise RuntimeError(f'Q32 requires length{length} tensor')
    return _values(value) if length is not None else None


def _tensor_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                bytes=value.numel() * value.element_size())


def collect_q32_physical_layers(model, method_type):
    """Inspect physical tensors and logical maps after all real layers allocate."""
    methods = {}
    owners = {}
    for layer in model.modules():
        method = getattr(layer, 'quant_method', None)
        if not isinstance(method, method_type): continue
        index = method.kt_config.layer_idx
        if index in methods: raise RuntimeError('duplicate Q32 physical layer')
        methods[index] = method
        owners[index] = layer
    if set(methods) != set(range(48)):
        raise RuntimeError('Q32 physical record requires exactly48 real KTEP layers')
    records = []
    maps = []
    devices = []
    for index in range(48):
        method = methods[index]
        if method.wrapper.num_experts != 512 or method.num_gpu_experts != 32:
            raise RuntimeError('Q32 requires complete512 backing and actual resident32')
        mask = _require_tensor(method.gpu_experts_mask, dtype='torch.bool', device='cpu', length=512)
        mask_cuda = _require_tensor(method.gpu_experts_mask_cuda,
                                    dtype='torch.bool', device='npu', length=512)
        pinned_mask = _require_tensor(method.wrapper.gpu_experts_mask,
                                      dtype='torch.bool', device='cpu', length=512)
        # KTEP's initial torch.full inherits the model loader's NPU context;
        # upstream dynamic publication can later replace it with a CPU tensor.
        # Both are real runtime layouts. The inference map below must stay NPU.
        primary_device = getattr(method.logical_to_gpu_index.device, 'type', None)
        if primary_device not in ('cpu', 'npu'):
            raise RuntimeError('Q32 primary logical map requires CPU or NPU storage')
        logical = _require_tensor(method.logical_to_gpu_index,
                                  dtype='torch.int32', device=primary_device, length=512)
        logical_cuda = _require_tensor(method.logical_to_gpu_index_cuda,
                                       dtype='torch.int32', device='npu', length=512)
        reverse = _require_tensor(method.gpu_index_to_logical,
                                  dtype='torch.int32', device='cpu', length=32)
        selected = [expert for expert, enabled in enumerate(mask) if enabled]
        expected_logical = [-1] * 512
        for slot, expert in enumerate(selected): expected_logical[expert] = slot
        if (len(mask) != 512 or len(logical) != 512 or len(reverse) != 32 or
                len(selected) != 32 or mask_cuda != mask or pinned_mask != mask or
                reverse != selected or logical != expected_logical or logical_cuda != logical):
            raise RuntimeError('Q32 physical logical/reverse maps are inconsistent')
        maps.append(dict(layer=index, selected_logical_experts=selected,
                         primary_logical_map_device=primary_device,
                         logical_to_gpu_index=logical,
                         gpu_index_to_logical=reverse))
        owner = owners[index]
        def tensor(name):
            return getattr(owner, name, getattr(method, name, None))
        for name in ('w13_weight', 'w2_weight'):
            _require_tensor(tensor(name), dtype='torch.bfloat16', device='npu')
        devices.append(dict(layer=index, w13_weight='npu', w2_weight='npu'))
        records.append(dict(layer=index, resident_capacity=32,
                            tensors={name: _tensor_record(tensor(name))
                                     for name in ('w13_weight', 'w2_weight')},
                            total_bytes=sum(_tensor_record(tensor(name))['bytes']
                                            for name in ('w13_weight', 'w2_weight'))))
    from .q32_gate import validate_q32_physical_layers
    summary = validate_q32_physical_layers(records)
    return dict(schema=1, status='RECORDED_Q32_PHYSICAL_NOT_F7_QUALIFIED',
                summary=summary, maps=maps, devices=devices, layers=records)


def write_q32_physical(scheduler):
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
    path = Path('/out/q32-physical.json')
    if path.exists(): raise FileExistsError('Q32 physical output must be fresh per model process')
    document = collect_q32_physical_layers(
        scheduler.tp_worker.model_runner.model, KTEPWrapperMethod)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def _validate_resident_maps(rows, capacity):
    if capacity not in (32, 64) or any(
            len(row) != capacity or len(set(row)) != capacity or
            any(type(expert) is not int or not 0 <= expert < 512 for expert in row)
            for row in rows):
        raise RuntimeError(f'calibration requires real Q{capacity} physical maps')


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
        self.broken = False
        self.path = Path(os.environ['ROUND9_ROUTE_STATS'])
        if self.path.exists(): raise FileExistsError('route output must be fresh per model process')
        methods = {}
        for layer in scheduler.tp_worker.model_runner.model.modules():
            method = getattr(layer, 'quant_method', None)
            if not isinstance(method, KTEPWrapperMethod): continue
            index = method.kt_config.layer_idx
            if index in methods: raise RuntimeError('duplicate calibration layer')
            methods[index] = method
        if set(methods) != set(range(48)): raise RuntimeError('calibration requires all48 real KTEP layers')
        self.residents = [methods[index].gpu_index_to_logical.detach().cpu().tolist() for index in range(48)]
        _validate_resident_maps(self.residents, args.kt_num_gpu_experts)
        for index, method in methods.items():
            original = method.apply
            def apply(layer, dispatch_output, _index=index, _original=original):
                try:
                    if self.broken: raise RuntimeError('calibration failure requires process restart')
                    if self.collector.batch is not None:
                        topk = dispatch_output.topk_output
                        ids = topk.topk_ids.detach().cpu().tolist()
                        weights = topk.topk_weights.detach().float().cpu().tolist()
                        self.collector.observe(_index, ids, weights)
                    return _original(layer, dispatch_output)
                except BaseException:
                    self.broken = True
                    raise
            method.apply = apply

    def before(self, batch):
        if self.broken: raise RuntimeError('calibration failure requires process restart')
        if not batch.forward_mode.is_extend(): return
        if len(batch.reqs) != 1: raise RuntimeError('calibration cannot mix requests')
        req = batch.reqs[0]
        start = req.last_extend_prefix_len
        total = len(req.origin_input_ids)
        if start == 0:
            if any(record['request_id'] == req.rid for record in self.records):
                raise RuntimeError('calibration request IDs must be unique per process')
            # A prior chunked request can be aborted between scheduler iterations.
            if self.collector.request is not None:
                reason = getattr(self.pending_req, 'finished_reason', None)
                if reason is None or reason.__class__.__name__ != 'FINISH_ABORT':
                    raise RuntimeError('new request before prior prefill completed or aborted')
                self.collector.abort(self.collector.request[0])
            if len(self.records) >= 64: raise RuntimeError('bounded calibration corpus limit64')
            self.record = dict(request_id=req.rid, tokens=total,
                               input_ids_sha256=payload_digest(req.origin_input_ids))
        # This F1 producer rejects both Dynamic modes. Static placement epoch0
        # is bound for the entire loaded process; it is not a general epoch API.
        self.collector.begin_batch(req.rid, 0, start=start, count=req.extend_input_len, total=total)
        self.req = req
        self.pending_req = req

    def after(self):
        if self.broken: raise RuntimeError('calibration failure requires process restart')
        if self.collector.batch is None: return
        # Aborts use FINISH_ABORT; normal length/EOS completion is still usable
        # prefill calibration because all input tokens have already executed.
        reason = getattr(self.req, 'finished_reason', None)
        aborted = reason is not None and reason.__class__.__name__ == 'FINISH_ABORT'
        complete = self.collector.end_batch(aborted=aborted)
        if complete:
            self.records.append(self.record)
            document = dict(schema=1, statistics=self.collector.stats.snapshot(), requests=self.records,
                            initial_residents=self.residents, placement_epoch=0)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + '\n')
            temp.replace(self.path)
        self.req = None
        if complete or aborted: self.pending_req = None


def framework_before(scheduler, batch):
    config = load_framework()
    if config is None: return
    if config['resident_capacity'] == 32 and not getattr(scheduler, '_round9_q32_physical_recorded', False):
        write_q32_physical(scheduler)
        scheduler._round9_q32_physical_recorded = True
    if not config['route_collection']: return
    collector = getattr(scheduler, '_round9_route_collector', None)
    if collector is None:
        collector = scheduler._round9_route_collector = RuntimeCollector(scheduler)
    collector.before(batch)


def framework_after(scheduler):
    collector = getattr(scheduler, '_round9_route_collector', None)
    if collector is not None: collector.after()
