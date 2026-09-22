"""Explicit opt-in cause-decomposition hooks. Never production qualification."""
import json
from pathlib import Path


def make_hook(config):
    import torch
    mode=config['mode'];root=Path(config['output']);root.mkdir(exist_ok=True)
    if not str(root).startswith('/workspace/results/r65-5a-'):
        raise ValueError('DIAGNOSTIC_SCOPE')
    counts={}
    if mode=='baseline_routes':
        source=Path(config['baseline'])/'stagewise'
        def hook(module,inputs,output):
            layer=module.layer_id
            call=counts.get(layer,0)+1;counts[layer]=call
            ids=torch.load(source/f'model_layer{layer:02d}_router_topk_ids_call{call:04d}.pt',
                           map_location='cpu',weights_only=False)['tensor']
            weights=torch.load(source/f'model_layer{layer:02d}_router_weights_call{call:04d}.pt',
                               map_location='cpu',weights_only=False)['tensor']
            if ids.shape!=output.topk_ids.shape or weights.shape!=output.topk_weights.shape:
                raise ValueError('COUNTERFACTUAL_SEQUENCE_SHAPE')
            with (root/'interventions.jsonl').open('a') as f:
                f.write(json.dumps(dict(layer=layer,call=call,mode=mode,
                    original_ids=output.topk_ids.cpu().tolist(),forced_ids=ids.tolist()))+'\n')
            return output._replace(topk_ids=ids.to(output.topk_ids.device,dtype=output.topk_ids.dtype),
                                   topk_weights=weights.to(output.topk_weights.device,dtype=output.topk_weights.dtype))
        return hook
    if mode!='npu_for_cpu':raise ValueError('UNKNOWN_DIAGNOSTIC_MODE')
    from types import SimpleNamespace as NS
    from safetensors import safe_open
    from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast
    from probe_npu_subset import metrics
    original=BaseMoEWrapper.copy_forward_output_to_device
    cache={}
    model=Path('/workspace/models/DeepSeek-V2-Lite-604d5664')
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    method=NS(with_bias=False,moe_runner_config=NS(activation='silu'),_kt_return_fp32_contribution=True)
    def replacement(self,hidden):
        layer=self.layer_idx
        if layer not in (1,9,17,26):raise ValueError('UNEXPECTED_CPU_LAYER')
        if layer not in cache:
            w13=[];w2=[]
            for expert in range(1,64,2):
                weights={}
                for name in ('gate','up','down'):
                    key=f'model.layers.{layer}.mlp.experts.{expert}.{name}_proj.weight'
                    with safe_open(model/index[key],framework='pt',device='cpu') as sf:
                        weights[name]=sf.get_tensor(key).clone()
                w13.append(torch.cat([weights['gate'],weights['up']]))
                w2.append(weights['down'])
            cache[layer]=NS(w13_weight=npu_format_cast(torch.stack(w13).to(hidden.device).transpose(1,2)),
                            w2_weight=npu_format_cast(torch.stack(w2).to(hidden.device).transpose(1,2)),top_k=6)
        buffers=KExpertsCPUBuffer.get_buffer(hidden.view(-1,hidden.shape[-1]),self.num_experts_per_tok,self.output_dtype)
        slot=layer%KExpertsCPUBuffer.buffer_depth
        x,ids,_,weights,out,_,_=buffers
        logical=ids[slot].long()
        physical=torch.where((logical>=0)&(logical%2==1),logical//2,-1).to(torch.int32)
        reference=UnquantizedFusedMoEMethod.forward_npu(method,cache[layer],NS(
            hidden_states=x[slot].to(hidden.device),
            topk_output=(weights[slot].to(hidden.device),physical.to(hidden.device),None))).hidden_states.cpu()
        if reference.shape!=out[slot].shape or reference.dtype!=out[slot].dtype:
            raise ValueError('DIAGNOSTIC_OUTPUT_CONTRACT')
        call=counts.get(layer,0)+1;counts[layer]=call
        record=dict(layer=layer,call=call,mode=mode,**metrics(out[slot],reference))
        if call==55:
            torch.save(dict(actual_cpu=out[slot].clone(),replacement=reference.clone()),root/f'L{layer}-call55.pt')
        with (root/'interventions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        # CPU work completed via original submit/sync. Change only its numeric
        # result before original framework H2D and the unchanged KTEP merge.
        out[slot].copy_(reference)
        return original(self,hidden)
    BaseMoEWrapper.copy_forward_output_to_device=replacement
    return lambda module,inputs,output: None
