"""Frozen Qwen head dimensions; isolated NPU GDN smoke, not model proof."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update_npu


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.manual_seed(8675309)
    report=dict(scope='SYNTHETIC_GDN_PRIMITIVES_NOT_MODEL_OR_CACHE_QUALIFICATION',rows=[])
    try:
        for mode,tokens in [('decode',1),('prefill',64)]:
            q=torch.randn(1,tokens,16,128,dtype=torch.bfloat16).npu()
            k=torch.randn_like(q)
            v=torch.randn(1,tokens,32,128,dtype=torch.bfloat16).npu()
            outputs=[]
            for repeat in range(3):
                state=torch.zeros(1,32,128,128,dtype=torch.float32,device='npu')
                seq=torch.tensor([0,tokens],dtype=torch.int32,device='npu')
                if mode=='decode':
                    out=fused_sigmoid_gating_delta_rule_update_npu(
                        A_log=torch.zeros(32,device='npu'), a=torch.zeros(1,tokens,32,device='npu'),
                        dt_bias=torch.zeros(32,device='npu'), softplus_beta=1.,softplus_threshold=20.,
                        q=q,k=k,v=v,b=torch.zeros(1,tokens,32,device='npu'),
                        initial_state_source=state, initial_state_indices=torch.tensor([0],device='npu',dtype=torch.int32),
                        cu_seqlens=seq,use_qk_l2norm_in_kernel=True)
                else:
                    out,*_=chunk_gated_delta_rule_npu(q=q,k=k,v=v,
                        g=torch.full((1,tokens,32),-.5,device='npu'),
                        beta=torch.full((1,tokens,32),.5,device='npu'),
                        initial_state=state,cu_seqlens=seq,use_qk_l2norm_in_kernel=True)
                torch.npu.synchronize()
                outputs.append(out.cpu())
            finite=all(torch.isfinite(x).all().item() for x in outputs)
            exact=all(torch.equal(outputs[0],x) for x in outputs[1:])
            report['rows'].append(dict(mode=mode,tokens=tokens,shape=list(outputs[0].shape),finite=finite,repeat_exact=exact))
            assert finite and exact
        report['status']='PRIMITIVE_SMOKE_PASS'
    except Exception as e:
        report.update(status='FAIL',error=repr(e))
        raise
    finally:
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
