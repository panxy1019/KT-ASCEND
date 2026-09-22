"""Serial f02 scheduler adapter. Disabled unless explicitly installed and enabled.

Device migration qualification is separate from scheduler/full-model admission.
"""
class RequestBoundary:
    def __init__(self,controller,event=None):
        self.controller=controller;self.request=None;self.generation=None
        self.event=event or (lambda item:None)
    def before(self,requests):
        if len(requests)!=1:raise RuntimeError('Dynamic requires exactly one active request')
        request=requests[0]
        if self.request is None:
            self.generation=self.controller.begin(request.rid);self.request=request
        elif self.request is not request:
            raise RuntimeError('previous request boundary has not drained')
    def observe(self,layer,ids):
        if self.request is None:raise RuntimeError('unleased expert dispatch')
        self.controller.observe(self.request.rid,self.generation,layer,ids)
    def after(self):
        req=self.request
        if req is None or not req.finished():return
        # f02 release_kv_cache -> free_mamba_cache + free(req) sets both to None.
        # The normal scheduler calls this after process_batch_result returns,
        # including its grouped token frees; no overlap/radix/offload is allowed.
        if req.req_pool_idx is not None or req.mamba_pool_idx is not None:
            raise RuntimeError('request KV/GDN/PLE consumers have not been released')
        self.controller.adapter.synchronize()
        self.controller.finish(req.rid,consumers_drained=True)
        plans=self.controller.plan();generation=self.controller.commit(plans)
        self.event(dict(request_id=req.rid,generation=generation,
            changed_layers=len(plans),copied_experts=sum(len(p.moves) for p in plans),
            plans=[dict(layer=p.layer,previous=list(p.previous),slots=list(p.slots),moves=list(p.moves)) for p in plans]))
        self.request=None;self.generation=None


def _enabled():
    import os
    return os.environ.get('ROUND9_DYNAMIC_BOUNDARY_ENABLE')=='1'


def scheduler_before(scheduler,batch):
    if not _enabled():return
    boundary=getattr(scheduler,'_round9_dynamic_boundary',None)
    if boundary is None:
        boundary=_bind_scheduler(scheduler)
        scheduler._round9_dynamic_boundary=boundary
    boundary.before(batch.reqs)


def scheduler_after(scheduler):
    if not _enabled():return
    boundary=getattr(scheduler,'_round9_dynamic_boundary',None)
    if boundary is not None:
        request=boundary.request
        boundary.after()
        if request is not None and boundary.request is None:
            runner=scheduler.tp_worker.model_runner
            budget=getattr(runner,'_round9_joint_host_budget',None)
            if budget is not None:
                if request.rid in runner._round9_ple_credit_requests:
                    budget.retire_request(request.rid)
                    runner._round9_ple_credit_requests.remove(request.rid)
                snapshot=budget.snapshot()
                if (snapshot['allocations'] or snapshot['active_requests']
                        or snapshot['in_flight'] or snapshot['cancelled']
                        or any(snapshot['used'].values())):
                    raise RuntimeError('joint host budget did not drain at request boundary')
                event=getattr(boundary,'event',None)
                if callable(event):
                    event(dict(status='JOINT_BUDGET_REQUEST_DRAINED',
                               request_id=request.rid,budget=snapshot))
                runner._round9_ple_request=None


def _require_capacity_gate(capacity, candidate, binding, path, expected_sha, native_sha256=None):
    identity = binding['bank_identity']
    kwargs = dict(storage=candidate.storage, workers=candidate.workers,
                  source_identity=identity['source_identity'],
                  layer0_sha256=identity['layer0_sha256'])
    if capacity == 32:
        from .q32_gate import require_q32_device_gate
        return require_q32_device_gate(path, expected_sha,
                                       native_sha256=native_sha256, **kwargs)
    if capacity != 64: raise RuntimeError('unsupported resident capacity')
    from .candidate_config import require_dynamic_device_gate
    return require_dynamic_device_gate(path, expected_sha, **kwargs)


def _optional_framework():
    """Candidate-only overlays predate the optional framework module."""
    try:
        from .framework_config import load_framework
    except ModuleNotFoundError as error:
        if not error.name or error.name.rsplit('.', 1)[-1] != 'framework_config': raise
        return None
    return load_framework()


def _framework_capacity():
    framework = _optional_framework()
    return framework['resident_capacity'] if framework is not None else 64


def _bind_scheduler(scheduler):
    import hashlib,json,os
    from pathlib import Path
    from .candidate_config import require_dynamic_model_profile,load_candidate_profile
    binding=require_dynamic_model_profile(os.environ.get("ROUND9_CANDIDATE_PROFILE"))
    candidate,_,_=load_candidate_profile(os.environ.get("ROUND9_CANDIDATE_PROFILE"))
    capacity=_framework_capacity()
    import kt_kernel_ext
    native_sha256=hashlib.sha256(Path(kt_kernel_ext.__file__).resolve().read_bytes()).hexdigest()
    gate=_require_capacity_gate(capacity,candidate,binding,
        os.environ.get('ROUND9_DYNAMIC_DEVICE_GATE'),os.environ.get('ROUND9_DYNAMIC_DEVICE_GATE_SHA256'),
        native_sha256=native_sha256)
    from .expert_source import OriginalExpertSource
    from .dynamic_adapter import TorchKTEPAdapter,bind_controller
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
    args=scheduler.server_args
    if (not args.disable_overlap_schedule or not args.disable_radix_cache or
        not args.disable_cuda_graph or args.max_running_requests!=1 or
        args.kt_enable_dynamic_expert_update or args.kt_max_deferred_experts_per_token or
        args.kt_gpu_prefill_token_threshold or args.tp_size!=1 or
        getattr(args,'speculative_algorithm',None) is not None):
        raise RuntimeError('unsupported Dynamic scheduler profile')
    if args.kt_num_gpu_experts != capacity:
        raise RuntimeError('Dynamic resident capacity differs from framework profile')
    if getattr(scheduler,'enable_hierarchical_cache',False) or getattr(scheduler,'enable_hicache_storage',False):
        raise RuntimeError('Dynamic rejects asynchronous cache offload')
    log=Path(os.environ['ROUND9_DYNAMIC_EVENTS'])
    if log.exists():raise FileExistsError('Dynamic event log must be fresh per model load')
    layers={};methods={}
    for layer in scheduler.tp_worker.model_runner.model.modules():
        method=getattr(layer,'quant_method',None)
        if isinstance(method,KTEPWrapperMethod):
            index=method.kt_config.layer_idx
            if index in layers:raise RuntimeError('duplicate KTEP layer')
            if method.wrapper.num_experts!=512:raise RuntimeError('complete512 CPU backing required')
            layers[index]=layer;methods[index]=method
    if set(layers)!=set(range(48)):raise RuntimeError('Dynamic requires exactly48 real KTEP layers')
    if str(Path(args.model_path).resolve())!=binding['bank_identity']['source_identity']['model_path']:
        raise RuntimeError('Dynamic original source differs from verified model binding')
    source=OriginalExpertSource(args.model_path)
    adapter=TorchKTEPAdapter(layers,methods,source,complete_cpu_backing_ids=range(512),
        host_staging_limit=64*1024**2,allow_unqualified_npu_probe=True,qualified_npu_formats=(29,))
    def event(item):
        with log.open('a') as f:f.write(json.dumps(item)+'\n')
    framework=_optional_framework()
    if framework is not None and (framework.get('prefill') or {}).get('update_residents',False):
        from .prefill_end_boundary import PrefillEndBoundary
        from .live_request_state import snapshot_live_request
        boundary=PrefillEndBoundary(bind_controller(adapter),
            lambda request:snapshot_live_request(scheduler.tp_worker.model_runner,request),event)
    else:
        boundary=RequestBoundary(bind_controller(adapter),event)
    shared_budget=getattr(scheduler.tp_worker.model_runner,'_round9_joint_host_budget',None)
    if shared_budget is not None:
        from .budgeted_migration import BudgetedMigration
        boundary.controller.adapter=BudgetedMigration(adapter,shared_budget,
            lambda:(boundary.request.rid,boundary.generation),
            requests=scheduler.tp_worker.model_runner._round9_ple_credit_requests)
    # Instance-local interception records the original logical routing IDs,
    # before KTEP transforms them into the current NPU slot indices.
    for index,method in methods.items():
        original=method.apply
        def apply(layer,dispatch_output,_index=index,_original=original):
            ids=dispatch_output.topk_output.topk_ids.detach().cpu().reshape(-1).tolist()
            boundary.observe(_index,[i for i in ids if i!=-1])
            return _original(layer,dispatch_output)
        method.apply=apply
    event(dict(status='BOUND_NOT_FULL_MODEL_QUALIFIED',layers=48,generation=0,**binding))
    return boundary


def installed_scheduler_boundary():
    """Inspect the effective scheduler callable, not merely an enable env var."""
    if not _enabled():return False
    import inspect
    from sglang.srt.managers.scheduler import Scheduler
    function=inspect.unwrap(Scheduler.event_loop_normal)
    code=getattr(function,'__code__',None)
    return _enabled() and code is not None and {'scheduler_before','scheduler_after'}<=set(code.co_names)
