"""Diagnostic CPU backing-pool migration regression; execute only after reaping.

Load through the ordinary wrapper with its initial static mask. Then use the
official in-place mask update to evict two initially resident experts to CPU.
Their real-operand outputs must pass the unchanged canonical local safety gate.
No model weight, routing policy or arithmetic boundary is changed by this test.
"""
import argparse
import hashlib
import json
from pathlib import Path


def eviction_mask(initial,evicted):
    """Keep slot budget fixed while moving chosen resident logical IDs to CPU."""
    initial=set(initial);evicted=set(evicted)
    if not evicted or not evicted<=initial:
        raise ValueError('INVALID_EVICTION_FIXTURE')
    replacements=[i for i in range(64) if i not in initial][:len(evicted)]
    if len(replacements)!=len(evicted):raise ValueError('NO_REPLACEMENT_CAPACITY')
    return sorted((initial-evicted)|set(replacements))


def main():
    import torch
    import torch_npu  # Register the established Ascend tensor device.
    from kt_kernel import KTMoEWrapper
    from sglang.srt.layers.moe.kt_ep_wrapper import update_kt_wrapper_masks
    from dynamic_audit import checkpoint_projection

    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    rows=json.loads(a.profile.read_text())['rows']
    initial=next(r['npu_ids'] for r in rows if r['layer']==a.layer)
    assert len(initial)>=2,'FIXTURE_NEEDS_TWO_INITIAL_RESIDENT_EXPERTS'
    evicted=[initial[0],initial[-1]]
    new_ids=eviction_mask(initial,evicted)
    mask=torch.zeros(64,dtype=torch.bool);mask[initial]=True
    cpu=KTMoEWrapper(layer_idx=a.layer,num_experts=64,num_experts_per_tok=6,
        hidden_size=2048,moe_intermediate_size=1408,gpu_experts_mask=mask,
        cpuinfer_threads=64,threadpool_count=2,numa_nodes=[4,5],
        weight_path='/workspace/results/r65-assets-v1/all-moe-f32.gguf',
        chunked_prefill_size=64,max_deferred_experts_per_token=0,method='LLAMAFILE')
    report=dict(status='RUNNING',layer=a.layer,gate=.01,repeats=10,
        initial_resident=initial,new_resident=new_ids,evicted_to_cpu=evicted,
        profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),rows=[],
        scope='STANDALONE_OFFICIAL_MASK_UPDATE_NOT_FULL_MODEL_DYNAMIC_QUALIFICATION')
    def save():
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    def digest(x):
        return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    try:
        cpu.load_weights(torch.arange(64,dtype=torch.int32))
        pointer=cpu.gpu_experts_mask.data_ptr()
        new_mask=torch.zeros(64,dtype=torch.bool);new_mask[new_ids]=True
        update_kt_wrapper_masks(cpu,new_mask)
        assert cpu.gpu_experts_mask.data_ptr()==pointer,'MASK_POINTER_CHANGED'
        assert torch.equal(cpu.gpu_experts_mask,new_mask),'WRAPPER_MASK_UPDATE'
        for expert in evicted:
            w13=checkpoint_projection(a.layer,expert,'w13_weight').float()
            gate,up=w13.chunk(2,dim=0)
            down=checkpoint_projection(a.layer,expert,'w2_weight').float()
            for capture in (1,4,7):
                path=Path('/workspace/results/r65-r67-serial-v5/R16/observer/operands')/f'L{a.layer:02d}-C{capture}.pt'
                x=torch.load(path,weights_only=False)['tensor']
                ids=torch.full((len(x),6),-1,dtype=torch.int64);ids[:,0]=expert
                routes=torch.zeros(len(x),6);routes[:,0]=1
                hashes=[];first=None
                for _ in range(10):
                    output=cpu.forward(x,ids,routes).detach().cpu().clone()
                    hashes.append(digest(output))
                    if first is None:first=output
                g=(x.float()@gate.T).bfloat16().float()
                u=(x.float()@up.T).bfloat16().float()
                activated=(torch.nn.functional.silu(g)*u).bfloat16().float()
                reference=(activated@down.T).bfloat16().float()
                relative=float((first.double()-reference.double()).norm()/reference.double().norm().clamp_min(1e-30))
                finite=bool(torch.isfinite(first).all())
                row=dict(expert=expert,capture=capture,input_sha256=digest(x),
                    output_sha256=hashes[0],repeat_exact=len(set(hashes))==1,
                    relative_l2_vs_canonical=relative if finite else None,finite=finite)
                report['rows'].append(row);save()
                assert finite,'EVICTED_CPU_OUTPUT_NONFINITE'
                assert row['repeat_exact'],'EVICTED_CPU_NONDETERMINISM'
                assert relative<=.01,'EVICTED_CPU_BACKING_LOCAL_SAFETY_FAILURE'
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
