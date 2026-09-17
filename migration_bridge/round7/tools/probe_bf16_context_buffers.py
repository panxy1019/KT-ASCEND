"""Reduced actual SharedFullContext allocation probe; no model/request mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.runtime/'sglang-python'))
    import torch
    import torch_npu
    import sglang.srt.layers.moe.kt_ep_wrapper as module
    torch.npu.set_device(0)
    # This is a TP1 primitive probe, not a running serving distributed group.
    module.get_tensor_model_parallel_rank=lambda:0
    module.get_tensor_model_parallel_world_size=lambda:1
    module._all_tp_ranks_succeeded=lambda local_success:local_success
    context=module.SharedFullContext.__new__(module.SharedFullContext)
    for flag in ('mxfp4','mxfp8','fp8','fp8_channel','bf16'):
        setattr(context,f'_is_{flag}_quant',flag=='bf16')
    layer=torch.nn.Module();layer.num_experts=2
    layer.register_parameter('w13_weight',torch.nn.Parameter(torch.empty((2,12,4),dtype=torch.bfloat16,device='npu'),requires_grad=False))
    layer.register_parameter('w2_weight',torch.nn.Parameter(torch.empty((2,4,6),dtype=torch.bfloat16,device='npu'),requires_grad=False))
    context.gpu_layer=layer
    report=dict(source=str(module.__file__),source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                torch_cuda_available=torch.cuda.is_available(),buffers={})
    try:
        context._create_cpu_buffers()
        for name,buf in context.cpu_buffers.items():
            report['buffers'][name]=dict(shape=list(buf.shape),device=str(buf.device),dtype=str(buf.dtype),
                pinned=buf.is_pinned(),pointer_collected=context.all_rank_buffer_ptrs[name]==[buf.data_ptr()])
        report['status']='OBSERVED'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        context.close()
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
