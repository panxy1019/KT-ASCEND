"""Qwen GDN convolution channels8192/width4, prefill then decode state smoke."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_npu, causal_conv1d_update_npu


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.manual_seed(123456);torch.set_num_threads(1)
    x=torch.randn(1,8192,64).bfloat16();w=torch.randn(8192,4).bfloat16()
    state=torch.zeros(1,8192,3,dtype=torch.bfloat16,device='npu')
    ids=torch.tensor([0],dtype=torch.int32,device='npu')
    report=dict(scope='CONV_PRIMITIVES_NOT_SERVING_CACHE_QUALIFICATION',rows=[])
    try:
        out=causal_conv1d_fn_npu(x[0].npu(),w.npu(),query_start_loc=torch.tensor([0,64],dtype=torch.int32,device='npu'),
            cache_indices=ids,has_initial_state=torch.tensor([False],device='npu'),conv_states=state,activation='silu')
        ref=torch.nn.functional.silu(torch.nn.functional.conv1d(torch.nn.functional.pad(x.float(),(3,0)),w.float().unsqueeze(1),groups=8192))[0]
        def record(mode,actual,expected):
            actual=actual.cpu().float();finite=bool(torch.isfinite(actual).all())
            report['rows'].append(dict(mode=mode,finite=finite,relative_l2=float((actual-expected).norm()/expected.norm().clamp_min(1e-12))))
            assert finite
        record('prefill64',out,ref)
        history=x[:,:,-3:].float()
        report['prefill_state_exact']=torch.equal(state.cpu(),x[:,:,-3:])
        for step in range(4):
            value=torch.randn(1,8192).bfloat16()
            joined=torch.cat((history,value.float().unsqueeze(-1)),dim=-1)
            expected=torch.nn.functional.silu((joined*w.float().unsqueeze(0)).sum(-1))
            actual=causal_conv1d_update_npu(value.npu(),state,w.npu(),activation='silu',conv_state_indices=ids)
            record(f'decode{step}',actual,expected)
            history=joined[:,:,1:]
            assert torch.equal(state.cpu(),history.bfloat16()),'STATE_IDENTITY_FAILURE'
        assert report['prefill_state_exact']
        report['status']='CONV_STATE_DIAGNOSTIC_COMPLETE'
    except Exception as e:
        report.update(status='FAIL',error=repr(e));raise
    finally:
        a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
