"""Reduced actual BF16 prepare Phase1: inference-resident -> raw context."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-format',choices=('nd','nz'),default='nd')
    p.add_argument('--source-layout',choices=('raw','inference'),default='inference')
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--intermediate',type=int,default=16);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.runtime/'sglang-python'))
    import torch
    import torch_npu
    import sglang.srt.layers.moe.kt_ep_wrapper as module
    torch.npu.set_device(0)
    module.get_tensor_model_parallel_rank=lambda:0
    module.get_tensor_model_parallel_world_size=lambda:1
    context=module.SharedFullContext.__new__(module.SharedFullContext)
    for flag in ('mxfp4','mxfp8','fp8','fp8_channel','bf16'):
        setattr(context,f'_is_{flag}_quant',flag=='bf16')
    layer=torch.nn.Module();layer.num_experts=2;original=torch.nn.Module()
    context.cpu_buffers={};reference={}
    for name,shape in [('w13_weight',(2,2*a.intermediate,a.hidden)),('w2_weight',(2,a.hidden,a.intermediate))]:
        raw=((torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32)%997)/128).reshape(shape).bfloat16()
        reference[name]=raw
        layer.register_parameter(name,torch.nn.Parameter(torch.full(shape,float('nan'),dtype=torch.bfloat16,device='npu'),requires_grad=False))
        resident=raw.flip(0).to('npu')
        if a.source_layout=='inference':resident=resident.transpose(1,2).contiguous()
        if a.source_format=='nz':resident=torch_npu.npu_format_cast(resident,29)
        original.register_parameter(name,torch.nn.Parameter(resident,requires_grad=False))
        context.cpu_buffers[name]=torch.empty(shape,dtype=torch.bfloat16,pin_memory=True)
    context.gpu_layer=layer
    report=dict(source=str(module.__file__),source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                source_format=a.source_format,source_layout=a.source_layout,rows=[])
    try:
        context._prepare_weight_bf16(None,original,torch.ones(2,dtype=torch.bool),torch.tensor([1,0],dtype=torch.int32))
        torch.npu.synchronize()
        for name,expected in reference.items():
            observed=getattr(layer,name).detach().cpu()
            report['rows'].append(dict(name=name,source_shape=list(getattr(original,name).shape),
                source_npu_format=torch_npu.get_npu_format(getattr(original,name)),
                destination_npu_format=torch_npu.get_npu_format(getattr(layer,name)),
                destination_shape=list(observed.shape),raw_identity_exact=torch.equal(observed,expected),
                max_abs=float((observed.float()-expected.float()).abs().max())))
            torch.save(dict(expected=expected,observed=observed),a.output/f'{name}.pt')
        report['status']='PASS' if all(row['raw_identity_exact'] for row in report['rows']) else 'RESIDENT_LAYOUT_IDENTITY_FAIL'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        # No SHM allocation in this reduced probe; release owned pinned tensors.
        context.cpu_buffers.clear()
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
