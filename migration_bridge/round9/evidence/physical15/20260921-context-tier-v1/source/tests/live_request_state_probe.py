"""CPU torch integration probe for the live-state diagnostic; never opens NPU."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.live_request_state import snapshot_live_request


def main():
    modules=[]; gdn={}; short={1:torch.arange(24).reshape(2,3,4)}
    kv_ids=set(range(3,48,4))
    for layer in range(48):
        kind='Qwen4ReferenceIndexer' if layer in kv_ids else 'Qwen4ReferenceGatedDeltaNet'
        obj=type(kind,(),{})();obj.layer_id=layer;modules.append(obj)
        if layer not in kv_ids:gdn[layer]=NS(conv=[torch.arange(12).reshape(2,2,3)],temporal=torch.arange(16).reshape(2,2,4))
    ple=type('Qwen4ReferencePLE',(),{})();ple.layer_id=1;modules.append(ple)
    req_pool=NS(req_to_token=torch.tensor([[0,0],[4,5]]),ngram_pool=NS(context=torch.tensor([[0,0],[10,11]])))
    req_pool.get_mamba_indices=lambda indices:torch.ones_like(indices)
    req_pool.get_ngram_context=lambda indices:req_pool.ngram_pool.context[indices]
    req_pool.mamba2_layer_cache=lambda layer:gdn[layer]
    req_pool.short_conv_layer_cache=lambda layer:short[layer]
    k={i:torch.arange(16).reshape(8,1,2) for i in kv_ids}
    v={i:torch.arange(16).reshape(8,1,2) for i in kv_ids}
    raw={i:torch.arange(16).reshape(8,1,2) for i in kv_ids}
    compressed={i:torch.arange(4).reshape(2,1,2) for i in kv_ids}
    pool=NS(get_key_buffer=lambda i:k[i],get_value_buffer=lambda i:v[i],
            get_qsa_key_state_buffer=lambda i:raw[i],get_qsa_compressed_k_buffer=lambda i:compressed[i],
            qsa_rope_positions=torch.arange(8))
    runner=NS(req_to_token_pool=req_pool,token_to_kv_pool=pool,model=NS(modules=lambda:modules))
    req=NS(rid='r',req_pool_idx=1,mamba_pool_idx=1,origin_input_ids=[1,2],output_ids=[7],finished=lambda:False)
    before=snapshot_live_request(runner,req)
    assert snapshot_live_request(runner,req)==before,'temporary gather pointer leaked into stable identity'
    for tensor in (gdn[0].temporal,short[1],req_pool.ngram_pool.context,k[3],v[3],raw[3],compressed[3],pool.qsa_rope_positions):
        saved=tensor.clone();tensor.add_(1)
        assert snapshot_live_request(runner,req)!=before
        tensor.copy_(saved)
    assert snapshot_live_request(runner,req)==before
    req.req_pool_idx=None
    try:snapshot_live_request(runner,req)
    except RuntimeError:pass
    else:raise AssertionError('released request accepted')
    print(json.dumps(dict(status='PASS_CPU_LIVE_STATE_SNAPSHOT',components=len(before['tensors']),
                         scope='TORCH_CPU_SYNTHETIC_POOL_CONTRACT_NOT_NPU')))


if __name__=='__main__':main()
