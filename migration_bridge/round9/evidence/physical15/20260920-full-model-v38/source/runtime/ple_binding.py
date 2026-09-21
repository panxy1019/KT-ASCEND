"""Default-off binding of exact-input async PLE into the serial real scheduler."""
import hashlib
import json
import os
from pathlib import Path
from .framework_config import load_framework


def ple_model_ready(runner):
    config=load_framework()
    if config is None or config.get('ple') is None:return
    args=runner.server_args
    if (not args.disable_overlap_schedule or not args.disable_radix_cache or
            not args.disable_cuda_graph or args.max_running_requests != 1 or args.tp_size != 1 or
            getattr(args,'speculative_algorithm',None) is not None or
            args.chunked_prefill_size != 32 or args.max_prefill_tokens != 32):
        raise RuntimeError('async PLE requires serial no-prefix bounded32 scheduler')
    if getattr(runner,'_round9_ple',None) is not None:raise RuntimeError('async PLE already bound')
    joint=config.get('joint_host_budget',False)
    if joint and os.environ.get('ROUND9_DYNAMIC_BOUNDARY_ENABLE')!='1':
        raise RuntimeError('joint budget requires real Dynamic migration binding')
    from .candidate_bank import bind_verified_banks
    binding=bind_verified_banks(os.environ['ROUND9_CANDIDATE_PROFILE'],args.kt_weight_path,args.model_path)
    from . import ple_async,ple_rows,host_budget,framework_config
    from .ple_gate import require_ple_device_gate
    import torch
    import torch_npu
    receipt=config['ple']['device_gate']
    gate=require_ple_device_gate(receipt['path'],receipt['sha256'],
        source_identity=binding['source_identity'],
        source_sha256={m.__name__.rsplit('.',1)[-1]+'.py':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (ple_async,ple_rows,host_budget,framework_config)},
        torch_version=str(torch.__version__),torch_npu_version=str(torch_npu.__version__))
    if joint:
        from . import budgeted_migration
        from .joint_budget_gate import require_joint_budget_device_gate
        joint_receipt=config['joint_device_gate']
        require_joint_budget_device_gate(joint_receipt['path'],joint_receipt['sha256'],
            source_identity=binding['source_identity'],ple_gate_sha256=receipt['sha256'],
            source_sha256={m.__name__.rsplit('.',1)[-1]+'.py':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                           for m in (host_budget,budgeted_migration,ple_async)},
            torch_version=str(torch.__version__),torch_npu_version=str(torch_npu.__version__))
    layers=[m for m in runner.model.modules() if type(m).__name__=='Qwen4ReferencePLE']
    if len(layers)!=1 or layers[0].layer_id!=1 or layers[0].ngram_heads!=16:
        raise RuntimeError('async PLE requires exact original layer1/16-head geometry')
    ple=layers[0]
    reader=getattr(ple._table,'reader',None)
    if not isinstance(reader,ple_rows.ShardedBF16Rows) or reader.width!=160:
        raise RuntimeError('async PLE requires the shared original BF16 shard reader')
    slot_bytes=512*160*2
    # Two payload copies per active PLE ticket plus spare headroom. Shared mode
    # additionally reserves the existing adapter's four-expert host staging and
    # one whole resident migration batch. These are bytes, not bandwidth credits.
    expert_bytes=2560*640*3*2
    migration_host=4*expert_bytes if joint else 0
    migration_h2d=config['resident_capacity']*expert_bytes if joint else 0
    budget=host_budget.HostWorkBudget(enabled=True,host_bytes=migration_host+slot_bytes*6,
        h2d_bytes=migration_h2d+slot_bytes*2,pinned_bytes=slot_bytes*2,
        reserve_host_bytes=migration_host,reserve_h2d_bytes=migration_h2d)
    torch.npu.synchronize()
    before=int(torch.npu.memory_allocated())
    ring=ple_async.AsyncPLERing(reader,enabled=True,max_rows=512,budget=budget)
    torch.npu.synchronize()
    persistent=int(torch.npu.memory_allocated())-before
    if not 327680<=persistent<=gate['peak_extra_bytes']:
        ring.close();raise RuntimeError('PLE physical allocation exceeds component receipt')
    runner._round9_ple_workspace_bytes=gate['peak_extra_bytes']-persistent
    from .ple_runtime import PLEConsumerRuntime,bind_ple_consumer
    runtime=PLEConsumerRuntime(ring)
    bind_ple_consumer(ple,runtime)
    runner._round9_ple=runtime
    runner._round9_ple_credit_requests=set()
    runner._round9_joint_host_budget=budget if joint else None
    log=Path('/out/ple-async-events.jsonl')
    if log.exists():raise FileExistsError('PLE event log must be fresh')
    runner._round9_ple_event=lambda item: _event(log,item)
    runner._round9_ple_event(dict(status='PLE_ASYNC_BOUND',device_gate_sha256=receipt['sha256'],
        slots=2,max_rows=512,persistent_device_bytes=persistent,
        reserved_transient_bytes=runner._round9_ple_workspace_bytes,
        joint_host_budget=joint,budget=budget.snapshot(),qualification='COMPONENT_GATE_ONLY_MODEL_PENDING'))


def _event(path,item):
    with path.open('a') as stream:stream.write(json.dumps(item,allow_nan=False)+'\n')


def ple_before(scheduler,batch):
    runner=scheduler.tp_worker.model_runner
    runtime=getattr(runner,'_round9_ple',None)
    if runtime is None:raise RuntimeError('async PLE model binding is missing')
    if len(batch.reqs)!=1:raise RuntimeError('async PLE requires one live request')
    if batch.forward_mode.is_extend():phase='prefill_input'
    elif batch.forward_mode.is_decode():phase='sampled_decode'
    else:raise RuntimeError('unsupported async PLE forward mode')
    request=batch.reqs[0]
    boundary=getattr(scheduler,'_round9_dynamic_boundary',None)
    epoch=boundary.generation if boundary is not None else 0
    if request.mamba_pool_idx is None:raise RuntimeError('async PLE request state is not allocated')
    runtime.begin(request.rid,epoch,batch.input_ids.detach().cpu().tolist(),
                  int(request.mamba_pool_idx),phase=phase)
    runner._round9_ple_request=request
    runner._round9_ple_async_before=runtime.async_batches


def ple_after(scheduler):
    runner=scheduler.tp_worker.model_runner
    runtime=getattr(runner,'_round9_ple',None)
    if runtime is None:raise RuntimeError('async PLE model binding is missing')
    context=runtime.context
    if context is not None:
        asynchronous=runtime.async_batches>runner._round9_ple_async_before
        if asynchronous:runner._round9_ple_credit_requests.add(context[0])
        runtime.end()
        runner._round9_ple_event(dict(status='PLE_BATCH_DRAINED',request_id=context[0],
            generation=context[1],tokens=len(context[2]),phase=context[4],async_engaged=asynchronous))
    request=getattr(runner,'_round9_ple_request',None)
    if request is not None and request.finished() and runner._round9_joint_host_budget is None:
        if request.rid in runner._round9_ple_credit_requests:
            runtime.ring.budget.retire_request(request.rid)
            runner._round9_ple_credit_requests.remove(request.rid)
        runner._round9_ple_request=None
