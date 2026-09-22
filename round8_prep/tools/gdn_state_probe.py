"""Four decode steps against explicit CPU gated-delta recurrence; diagnostic."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update_npu as decode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.manual_seed(123456)
    torch.set_num_threads(1)
    state=torch.zeros(1,32,128,128,device='npu')
    reference=torch.zeros(32,128,128)
    report=dict(scope='SYNTHETIC_CONTINUOUS_DECODE_CPU_RECURRENCE_DIAGNOSTIC',rows=[])
    for step in range(4):
        q=torch.randn(1,1,16,128).bfloat16()
        k=torch.randn(1,1,16,128).bfloat16()
        v=torch.randn(1,1,32,128).bfloat16()
        qc=torch.nn.functional.normalize(q.float(),dim=-1).repeat_interleave(2,dim=2)[0,0]/(128**.5)
        kc=torch.nn.functional.normalize(k.float(),dim=-1).repeat_interleave(2,dim=2)[0,0]
        reference=reference*.5
        residual=(v[0,0].float()-torch.einsum('hk,hkv->hv',kc,reference))*.5
        reference=reference+kc.unsqueeze(-1)*residual.unsqueeze(-2)
        expected=torch.einsum('hk,hkv->hv',qc,reference)
        out=decode(A_log=torch.zeros(32,device='npu'),a=torch.zeros(1,1,32,device='npu'),
            dt_bias=torch.zeros(32,device='npu'),softplus_beta=1.,softplus_threshold=20.,
            q=q.npu(),k=k.npu(),v=v.npu(),b=torch.zeros(1,1,32,device='npu'),
            initial_state_source=state,initial_state_indices=torch.tensor([0],device='npu',dtype=torch.int32),
            cu_seqlens=torch.tensor([0,1],device='npu',dtype=torch.int32),use_qk_l2norm_in_kernel=True)
        torch.npu.synchronize()
        actual=out.cpu()[0,0].float()
        finite=bool(torch.isfinite(actual).all() and torch.isfinite(state).all())
        rel=float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
        report['rows'].append(dict(step=step,finite=finite,output_relative_l2=rel,state_norm=float(state.norm().cpu())))
        assert finite
    report['status']='DIAGNOSTIC_COMPLETE_NOT_MODEL_QUALIFICATION'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
