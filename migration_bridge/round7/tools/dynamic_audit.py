"""Default-off Dynamic bookkeeping through existing hook seam, no arithmetic edits.

This auditor is for diagnostic runs, never timed performance. It validates
source/destination weight identity after the official update and rejects mixed
mask/mapping state. Publication safety still needs serialized-execution evidence;
a successful audit must not be relabelled as proof for arbitrary concurrency.
"""
import json
from pathlib import Path
import time
import threading


def projection_orientation(source_shape,checkpoint_shape):
    if tuple(source_shape)==tuple(checkpoint_shape):return False
    if tuple(source_shape)==tuple(reversed(checkpoint_shape)):return True
    raise ValueError('FULL_CONTEXT_CHECKPOINT_SHAPE_MISMATCH')


def checkpoint_projection(layer,logical,name):
    """Independent BF16 source lane; no inference or checkpoint mutation."""
    import torch
    from safetensors import safe_open
    model=Path('/workspace/models/DeepSeek-V2-Lite-604d5664')
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    def get(projection):
        key=f'model.layers.{layer}.mlp.experts.{logical}.{projection}_proj.weight'
        with safe_open(model/index[key],framework='pt',device='cpu') as f:
            return f.get_tensor(key).clone()
    if name=='w13_weight':return torch.cat([get('gate'),get('up')])
    if name=='w2_weight':return get('down')
    raise ValueError('UNKNOWN_BF16_PROJECTION')


def install(root,initial):
    import torch
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod, SharedFullContext, select_top_experts_from_batch
    if getattr(KTEPWrapperMethod,'_round7_generation_audit',False):return
    original=KTEPWrapperMethod._update_gpu_experts_from_batch
    original_apply=KTEPWrapperMethod.apply
    original_prepare=SharedFullContext._prepare_weight_bf16
    def prepare(context,wrapper,original_layer=None,gpu_experts_mask=None,logical_to_gpu_index=None):
        # Diagnostic-only boundary snapshots; absent from OFF/performance paths.
        layer_idx=int(wrapper.layer_idx)
        record=dict(layer=layer_idx,time_unix=time.time(),buffers={name:list(buf.shape)
            for name,buf in context.cpu_buffers.items()},before=[],after=[])
        fixed=[0,1,8,56,63]
        if original_layer is not None and gpu_experts_mask is not None:
            for logical in fixed:
                if not bool(gpu_experts_mask[logical]):continue
                slot=int(logical_to_gpu_index[logical])
                for name in ('w13_weight','w2_weight'):
                    src=getattr(original_layer,name)[slot].detach().cpu()
                    ref=checkpoint_projection(layer_idx,logical,name)
                    if projection_orientation(src.shape,ref.shape):ref=ref.T
                    record['before'].append(dict(logical=logical,name=name,slot=slot,
                        shape=list(src.shape),checkpoint_exact=torch.equal(src,ref)))
        value=original_prepare(context,wrapper,original_layer,gpu_experts_mask,logical_to_gpu_index)
        torch.npu.synchronize()
        for logical in fixed:
            for name in ('w13_weight','w2_weight'):
                src=getattr(context.gpu_layer,name)[logical].detach().cpu()
                ref=checkpoint_projection(layer_idx,logical,name)
                if projection_orientation(src.shape,ref.shape):ref=ref.T
                exact=torch.equal(src,ref)
                record['after'].append(dict(logical=logical,name=name,shape=list(src.shape),
                    checkpoint_exact=exact,max_abs=float((src.float()-ref.float()).abs().max())))
                if not exact:
                    torch.save(dict(observed=src,expected=ref),Path(root)/f'prepare-L{layer_idx}-E{logical}-{name}-mismatch.pt')
        with (Path(root)/'bf16-prepare-boundaries.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return value
    def apply(self,*args,**kwargs):
        assert not getattr(self,'_round7_updating',False),'READ_DURING_MIGRATION'
        thread=threading.get_ident()
        if hasattr(self,'_round7_execution_thread'):
            assert self._round7_execution_thread==thread,'MULTIPLE_MODEL_EXECUTION_THREADS'
        self._round7_execution_thread=thread
        g=getattr(self,'_round7_generation',0)
        components=getattr(self,'_round7_generations',dict(weights=g,mapping=g,mask=g,wrapper_mask=g))
        assert set(components.values())=={g},'GENERATION_MISMATCH'
        with (Path(root)/'generation-consumption.jsonl').open('a') as f:
            f.write(json.dumps(dict(layer=self.kt_config.layer_idx,thread=thread,generation=g,
                **components,resident=torch.where(self.gpu_experts_mask)[0].tolist()))+'\n')
        return original_apply(self,*args,**kwargs)
    def update(self,layer,ctx,dispatch_output):
        if getattr(self,'_round7_updating',False):raise ValueError('NESTED_MIGRATION')
        old=self.gpu_experts_mask.clone()
        generation=getattr(self,'_round7_generation',0)
        if generation==0:
            assert torch.where(old)[0].tolist()==initial[self.kt_config.layer_idx],'INITIAL_STATIC_MASK'
        started=time.perf_counter()
        routes=dispatch_output.topk_output.topk_ids
        expected_selection=select_top_experts_from_batch(routes,self.global_num_experts,
                                                        self.num_gpu_experts).cpu().tolist()
        route_counts=torch.bincount(routes.detach().cpu().long().flatten(),minlength=64).tolist()
        self._round7_updating=True
        try:
            production_started=time.perf_counter()
            value=original(self,layer,ctx,dispatch_output)
            production_host_seconds=time.perf_counter()-production_started
            # Diagnostic-only wait/readback; OFF timing runs retain production
            # stream ordering and do not include this identity audit overhead.
            torch.npu.synchronize()
            selected=self.gpu_index_to_logical.tolist()
            assert selected==expected_selection,'OFFICIAL_HOT_EXPERT_SELECTION'
            assert len(selected)==self.num_gpu_experts and len(set(selected))==len(selected)
            expected=torch.full((64,),-1,dtype=torch.int32)
            expected[selected]=torch.arange(len(selected),dtype=torch.int32)
            assert torch.equal(self.logical_to_gpu_index.cpu(),expected),'MAPPING_IDENTITY'
            assert torch.equal(self.logical_to_gpu_index_cuda.cpu(),expected),'DEVICE_MAPPING'
            mask=torch.zeros(64,dtype=torch.bool);mask[selected]=True
            assert torch.equal(self.gpu_experts_mask,mask),'MASK_IDENTITY'
            assert torch.equal(self.gpu_experts_mask_cuda.cpu(),mask),'DEVICE_MASK'
            assert torch.equal(self.wrapper.gpu_experts_mask,mask),'WRAPPER_MASK'
            copied=0
            checkpoint_sampled=sorted(set([0,len(selected)-1])) if selected else []
            # Audit every copied expert, not only slot0. BF16 copy must be exact.
            for name in ('w13_weight','w2_weight'):
                source=getattr(ctx.gpu_layer,name)
                dest=getattr(layer,name)
                assert dest.shape[0]==len(selected)
                for slot,logical in enumerate(selected):
                    assert torch.equal(dest[slot],source[logical]),'MIGRATED_WEIGHT_IDENTITY'
                    copied+=dest[slot].numel()*dest.element_size()
                    if slot in checkpoint_sampled:
                        reference=checkpoint_projection(self.kt_config.layer_idx,logical,name)
                        source_cpu=source[logical].detach().cpu()
                        assert reference.dtype==source_cpu.dtype==torch.bfloat16,'CHECKPOINT_SOURCE_DTYPE'
                        if projection_orientation(source_cpu.shape,reference.shape):reference=reference.T
                        if not torch.equal(source_cpu,reference):
                            failure=dict(layer=self.kt_config.layer_idx,wrapper_layer=int(self.wrapper.layer_idx),
                                logical=logical,slot=slot,name=name,source_shape=list(source_cpu.shape),
                                reference_shape=list(reference.shape),previous_resident=torch.where(old)[0].tolist(),
                                max_abs=float((source_cpu.float()-reference.float()).abs().max()),
                                finite=bool(torch.isfinite(source_cpu).all()))
                            with (Path(root)/'checkpoint-identity-failures.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
                            torch.save(dict(observed=source_cpu,expected=reference),Path(root)/'checkpoint-identity-failure.pt')
                            raise AssertionError('FULL_CONTEXT_CHECKPOINT_EXPERT_IDENTITY')
            next_generation=generation+1
            self._round7_generations=dict(weights=next_generation,mapping=next_generation,
                mask=next_generation,wrapper_mask=next_generation)
            self._round7_generation=next_generation
            record=dict(time_unix=time.time(),layer=self.kt_config.layer_idx,
                prior_generation=generation,**self._round7_generations,selected=selected,
                prefill_rows=len(routes),activation_counts=route_counts,
                previous_resident=torch.where(old)[0].tolist(),
                churn=int((old!=mask).sum())/max(1,len(selected)),migration_bytes=copied,
                diagnostic_total_update_seconds=time.perf_counter()-started,
                official_update_host_seconds=production_host_seconds,
                timing_scope='DIAGNOSTIC_HOST_CALL_ELAPSED_NOT_DEVICE_EVENT_OR_TIMED_PERFORMANCE',
                weight_identity_exact=True,mask_mapping_wrapper_exact=True,
                checkpoint_source_identity_exact=True,
                checkpoint_sampled_logical_ids=[selected[s] for s in checkpoint_sampled],
                generation_boundary='SERIAL_TP1_BATCH1_ONLY_NOT_CONCURRENCY_QUALIFIED')
            with (Path(root)/'generation-audit.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            return value
        finally:self._round7_updating=False
    KTEPWrapperMethod._update_gpu_experts_from_batch=update
    KTEPWrapperMethod.apply=apply
    SharedFullContext._prepare_weight_bf16=prepare
    KTEPWrapperMethod._round7_generation_audit=True
