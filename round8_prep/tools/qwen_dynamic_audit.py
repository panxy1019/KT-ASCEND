"""Diagnostic-only Qwen Dynamic generation and migrated-weight auditor."""
import json
import threading
import time
from pathlib import Path


def append(root,name,row):
    with (Path(root)/name).open('a') as handle:
        handle.write(json.dumps(dict(time_unix=time.time(),**row))+'\n')


def checkpoint_projection(layer,logical,name):
    from safetensors import safe_open
    model=Path('/workspace/round8/model-sample')
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    key=(f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj'
         if name=='w13_weight' else
         f'model.language_model.layers.{layer}.mlp.experts.down_proj')
    with safe_open(model/index[key],framework='pt',device='cpu') as handle:
        return handle.get_slice(key)[logical].clone()


def orient(reference,observed):
    if tuple(reference.shape)==tuple(observed.shape):return reference
    if tuple(reversed(reference.shape))==tuple(observed.shape):return reference.T
    raise AssertionError(f'checkpoint shape {reference.shape} vs {observed.shape}')


def install(root,profile):
    import torch
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod,select_top_experts_from_batch
    if getattr(KTEPWrapperMethod,'_round8_qwen_dynamic_audit',False):return
    initial={row['layer']:row['npu_ids'] for row in profile['layers']}
    original_apply=KTEPWrapperMethod.apply
    original_update=KTEPWrapperMethod._update_gpu_experts_from_batch
    def apply(self,*args,**kwargs):
        assert not getattr(self,'_round8_updating',False),'READ_DURING_MIGRATION'
        thread=threading.get_ident();previous=getattr(self,'_round8_thread',thread)
        assert previous==thread,'MULTIPLE_MODEL_EXECUTION_THREADS'
        self._round8_thread=thread
        generation=getattr(self,'_round8_generation',0)
        components=getattr(self,'_round8_generations',dict(
            weights=generation,mapping=generation,mask=generation,wrapper_mask=generation))
        assert set(components.values())=={generation},'GENERATION_MISMATCH'
        append(root,'generation-consumption.jsonl',dict(
            layer=self.kt_config.layer_idx,thread=thread,generation=generation,**components,
            resident=torch.where(self.gpu_experts_mask)[0].cpu().tolist()))
        return original_apply(self,*args,**kwargs)
    def update(self,layer,ctx,dispatch_output):
        assert not getattr(self,'_round8_updating',False),'NESTED_MIGRATION'
        old=self.gpu_experts_mask.clone();generation=getattr(self,'_round8_generation',0)
        if generation==0:
            assert torch.where(old)[0].cpu().tolist()==initial[self.kt_config.layer_idx]
        routes=dispatch_output.topk_output.topk_ids
        expected_selection=select_top_experts_from_batch(
            routes,self.global_num_experts,self.num_gpu_experts).cpu().tolist()
        assert self.global_num_experts==256 and self.num_gpu_experts==32
        self._round8_updating=True;started=time.perf_counter()
        try:
            value=original_update(self,layer,ctx,dispatch_output)
            torch.npu.synchronize()
            selected=(self.gpu_index_to_logical.tolist()
                      if hasattr(self.gpu_index_to_logical,'tolist')
                      else list(self.gpu_index_to_logical))
            assert selected==expected_selection and len(set(selected))==32
            expected=torch.full((256,),-1,dtype=torch.int32)
            expected[selected]=torch.arange(32,dtype=torch.int32)
            mask=torch.zeros(256,dtype=torch.bool);mask[selected]=True
            assert torch.equal(self.logical_to_gpu_index.cpu(),expected)
            assert torch.equal(self.logical_to_gpu_index_cuda.cpu(),expected)
            assert torch.equal(self.gpu_experts_mask,mask)
            assert torch.equal(self.gpu_experts_mask_cuda.cpu(),mask)
            assert torch.equal(self.wrapper.gpu_experts_mask,mask)
            copied=0
            for name in ('w13_weight','w2_weight'):
                source=getattr(ctx.gpu_layer,name);dest=getattr(layer,name)
                assert dest.shape[0]==32
                for slot,logical in enumerate(selected):
                    assert torch.equal(dest[slot],source[logical]),'MIGRATED_WEIGHT_IDENTITY'
                    copied+=dest[slot].numel()*dest.element_size()
                if self.kt_config.layer_idx in (0,20,39):
                    for slot in (0,31):
                        logical=selected[slot];observed=source[logical].detach().cpu()
                        reference=orient(checkpoint_projection(
                            self.kt_config.layer_idx,logical,name),observed)
                        assert observed.dtype==reference.dtype==torch.bfloat16
                        assert torch.equal(observed,reference),'CHECKPOINT_SOURCE_IDENTITY'
            next_generation=generation+1
            components=dict(weights=next_generation,mapping=next_generation,
                            mask=next_generation,wrapper_mask=next_generation)
            self._round8_generations=components;self._round8_generation=next_generation
            append(root,'generation-audit.jsonl',dict(
                layer=self.kt_config.layer_idx,prior_generation=generation,**components,
                selected=selected,previous_resident=torch.where(old)[0].cpu().tolist(),
                prefill_rows=int(routes.shape[0]),migration_bytes=copied,
                weight_identity_exact=True,mask_mapping_wrapper_exact=True,
                checkpoint_source_sample_exact=self.kt_config.layer_idx in (0,20,39),
                diagnostic_seconds=time.perf_counter()-started,
                boundary='SERIAL_TP1_BATCH1'))
            return value
        finally:self._round8_updating=False
    KTEPWrapperMethod.apply=apply
    KTEPWrapperMethod._update_gpu_experts_from_batch=update
    KTEPWrapperMethod._round8_qwen_dynamic_audit=True


def make_structure(config):
    import torch
    root=Path(config['output']);profile=json.loads(Path(config['profile']).read_text())
    install(root,profile);seen=set()
    def hook(module,inputs,output):
        assert torch.isfinite(output).all(),'NONFINITE_DYNAMIC_MLP'
        layer=module.layer_id
        if layer in seen:return
        method=module.experts.quant_method
        ids=torch.where(method.gpu_experts_mask)[0].cpu().tolist()
        assert len(ids)==32 and len(set(ids))==32
        expected=torch.full((256,),-1,dtype=torch.int32)
        expected[ids]=torch.arange(32,dtype=torch.int32)
        assert torch.equal(method.logical_to_gpu_index.cpu(),expected)
        assert method.wrapper.max_deferred_experts_per_token==0
        append(root,'dynamic-structure.jsonl',dict(layer=layer,resident=ids,
            generation=getattr(method,'_round8_generation',0),finite=True,mapping_exact=True))
        seen.add(layer)
    return hook
