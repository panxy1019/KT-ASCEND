"""A3-only same-source staging regression; no model-level threshold fitting."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layout', choices=('raw','ascend_inference'), required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',started_unix=time.time(),rows=[])
    try:
        import torch
        import torch_npu
        from kt_kernel.utils.loader import GGUFLoader
        from kt_kernel.utils.llamafile import LlamafileMoEWrapper
        from llamafile_bf16_staging import LlamafileBF16Stager
        from dynamic_audit import checkpoint_projection
        torch.npu.set_device(0)
        loader=GGUFLoader('/workspace/results/r65-assets-v1/all-moe-f32.gguf')
        for layer in (1,13,26):
            wrapper=LlamafileMoEWrapper.__new__(LlamafileMoEWrapper)
            wrapper.num_experts=64;wrapper.hidden_size=2048;wrapper.moe_intermediate_size=1408
            wrapper.layer_idx=layer;wrapper.gguf_loader=loader
            shapes=((2,2816,2048),(2,2048,1408)) if a.layout=='raw' else ((2,2048,2816),(2,1408,2048))
            buffers=dict(w13_weight=torch.empty(shapes[0],dtype=torch.bfloat16,pin_memory=True),
                         w2_weight=torch.empty(shapes[1],dtype=torch.bfloat16,pin_memory=True))
            writer=LlamafileBF16Stager(wrapper,buffers,1)
            for idx,expert in enumerate((0,1,31,63)):
                slot=idx%2;other=1-slot
                before={name:buf[other].clone() for name,buf in buffers.items()} if idx else None
                writer.submit(expert,slot);writer.sync()
                for name,buf in buffers.items():
                    reference=checkpoint_projection(layer,expert,name)
                    if a.layout=='ascend_inference':reference=reference.T
                    assert torch.equal(buf[slot],reference),'GGUF_STAGING_CHECKPOINT_IDENTITY'
                    dest=torch.empty_like(buf[slot],device='npu')
                    dest.copy_(buf[slot],non_blocking=True);torch.npu.synchronize()
                    assert torch.equal(dest.cpu(),reference),'STAGING_H2D_WEIGHT_IDENTITY'
                    if before:assert torch.equal(buf[other],before[name]),'STAGING_OTHER_SLOT_CHANGED'
                first={name:buf[slot].clone() for name,buf in buffers.items()}
                for _ in range(3):
                    writer.submit(expert,slot);writer.sync()
                    assert all(torch.equal(buf[slot],first[name]) for name,buf in buffers.items()),'STAGING_NONDETERMINISM'
                report['rows'].append(dict(layer=layer,expert=expert,slot=slot,layout=a.layout,checkpoint_exact=True,
                    h2d_exact=True,other_slot_unchanged=before is not None,repeats_exact=4))
            # The missing writer seam is not a permission to quantizeF32 weights.
            original=writer.projections['gate']
            writer.projections['gate']=original.clone()
            writer.projections['gate'][1,0,0]=0.1234567
            try:writer.submit(1,0)
            except ValueError as error:assert str(error)=='LLAMAFILE_BF16_STAGING_WEIGHT_IDENTITY'
            else:raise AssertionError('LOSSY_F32_CONVERSION_NOT_REJECTED')
        report.update(status='PASS',count=len(report['rows']),lossy_source_rejected=True)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        report['ended_unix']=time.time()
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
