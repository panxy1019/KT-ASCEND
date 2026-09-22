"""Opt-in native CPU test: real Qwen3.8 weights, synthetic BF16 activations.

Not collected by host unittest discovery. Run inside the frozen CPU-only ARM
environment with its extension on PYTHONPATH. No torch_npu, SGLang or NPU calls.
This bounded subset does not qualify full512 logical expert ownership or model
activations. Every failed attempt keeps its distinct output file.
"""
import argparse
import hashlib
import json
import pathlib
import resource
import time

import torch
from safetensors import safe_open
import kt_kernel_ext as native


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--format',choices=['BF16','F32'],required=True)
    p.add_argument('--pools',type=int,choices=[1,2],required=True)
    p.add_argument('--layer',type=int,choices=[0,23,24,47],required=True)
    p.add_argument('--compat',action='store_true')
    p.add_argument('--full-bank',action='store_true',help='one bounded layer, all512 backing experts (BF16 only)')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if a.full_bank and a.format!='BF16': raise ValueError('full-bank probe is BF16 only')
    torch.set_num_threads(1); torch.manual_seed(90015)
    report=dict(status='RUNNING',scope='REAL_WEIGHTS_SYNTHETIC_INPUT_CPU_ONLY',
                layer=a.layer,format=a.format,pools=a.pools,workers=32*a.pools,compat=a.compat,
                native_file=native.__file__,native_sha256=hashlib.sha256(pathlib.Path(native.__file__).read_bytes()).hexdigest(),
                rel_l2_gate=0.01,rows=[],full_bank=a.full_bank,
                source_experts=[0,1,2,3,4,507,508,509,510,511])
    try:
        index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
        prefix=f'model.language_model.layers.{a.layer}.mlp.experts.'
        w={}
        for part in ('gate_up_proj','down_proj'):
            key=prefix+part
            with safe_open(a.model/index[key],framework='pt',device='cpu') as f:
                if a.full_bank: w[part]=f.get_tensor(key)
                else:
                    view=f.get_slice(key)
                    w[part]=torch.stack([view[e] for e in report['source_experts']])
        gate,up=w['gate_up_proj'].chunk(2,dim=1)
        weights=[gate.contiguous(),up.contiguous(),w['down_proj'].contiguous()]
        selected=[w[report['source_experts']] for w in weights] if a.full_bank else weights
        report['selected_source_weight_sha256']=[digest(w) for w in selected]
        f32=[w.float() for w in selected]
        del w,gate,up,selected
        if a.format=='F32': weights=f32
        pool=native.WorkerPoolConfig(); pool.subpool_count=a.pools
        pool.subpool_numa_map=[6,7][:a.pools]; pool.subpool_thread_count=[32]*a.pools
        cpu=native.CPUInfer(pool)
        cfg=native.moe.MOEConfig(512 if a.full_bank else 10,10,2560,640)
        owned_by_npu=torch.zeros(512 if a.full_bank else 10,dtype=torch.bool)
        cfg.gpu_experts_mask=owned_by_npu.data_ptr()
        cfg.pool=cpu.backend_; cfg.layer_idx=a.layer
        cfg.gate_type=cfg.up_type=cfg.down_type=30 if a.format=='BF16' else 0
        cfg.hidden_type=30; cfg.output_type=0; cfg.bf16_numerical_compat=a.compat
        cfg.m_block=32; cfg.group_min_len=10; cfg.group_max_len=32; cfg.max_len=32
        cfg.gate_proj,cfg.up_proj,cfg.down_proj=[w.data_ptr() for w in weights]
        moe=native.moe.MOE(cfg)
        moe.load_weights()
        for q in (1,2,9,10,11,32):
            x=torch.randn(q,2560).bfloat16()
            logical=report['source_experts'] if a.full_bank else list(range(10))
            ids=torch.tensor(logical,dtype=torch.int64).repeat(q,1)
            routes=torch.softmax(torch.randn(q,10),dim=1).float()
            qlen=torch.tensor([q],dtype=torch.int32)
            reference=torch.zeros(q,2560)
            for e in range(10):
                g=x.float()@f32[0][e].T; u=x.float()@f32[1][e].T
                if a.compat: g=g.bfloat16().float(); u=u.bfloat16().float()
                hidden=torch.nn.functional.silu(g)*u
                if a.compat: hidden=hidden.bfloat16().float()
                down=hidden@f32[2][e].T
                if a.compat and a.pools==1: down=down.bfloat16().float()
                route=routes[:,e,None]
                if a.compat: route=route.bfloat16().float()
                reference.add_(down*route)
            hashes=[]
            for _ in range(3):
                y=torch.full((q,2560),float('nan'))
                moe.forward(qlen.data_ptr(),10,ids.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),False)
                assert torch.isfinite(y).all(), 'non-finite native output'
                hashes.append(digest(y))
            error=float((y.double()-reference.double()).norm()/reference.double().norm().clamp_min(1e-30))
            row=dict(qlen=q,rel_l2=error,repeat_exact=len(set(hashes))==1,input_sha256=digest(x),output_sha256=hashes[0])
            report['rows'].append(row)
            assert error<=report['rel_l2_gate'], row
            assert row['repeat_exact'], 'same-path nondeterminism'
            owned_by_npu[logical]=True
            y.fill_(float('nan'))
            moe.forward(qlen.data_ptr(),10,ids.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),False)
            assert torch.equal(y,torch.zeros_like(y)), 'accelerator-owned routes must be skipped'
            owned_by_npu.zero_()
            moe.forward(qlen.data_ptr(),10,ids.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),False)
            assert digest(y)==hashes[0], 'complete CPU backing was not retained'
            row['ownership_flip_exact']=True
            skipped=torch.full_like(ids,-1)
            y.fill_(float('nan'))
            moe.forward(qlen.data_ptr(),10,skipped.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),False)
            assert torch.equal(y,torch.zeros_like(y)), 'all-skipped routes must produce zero'
            # Directly exercise in-place FP32 incremental merge with no contribution.
            y.fill_(3.25)
            moe.forward(qlen.data_ptr(),10,skipped.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),True)
            assert torch.equal(y,torch.full_like(y,3.25)), 'incremental merge changed existing output'
            row['skip_zero_exact']=True; row['incremental_nohit_exact']=True
            print(json.dumps(row),flush=True)
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAIL',error=repr(exc)); raise
    finally:
        report['finished_at']=time.time()
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
