"""Single real expert through production LLAMAFILE, C64 two32 subpools."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import torch
import torch_npu
from kt_kernel import KTMoEWrapper
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(8675309)
    w = load_file(args.sample)
    gate, up = w['gate_up_proj'].chunk(2, dim=0)
    down = w['down_proj']
    path = args.output / 'layer0-expert0-f32.gguf'
    writer = gguf.GGUFWriter(str(path), 'llama')
    for name, value in [('gate', gate), ('up', up), ('down', down)]:
        writer.add_tensor(f'blk.0.ffn_{name}_exps.weight', value.unsqueeze(0).contiguous().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    cpu = KTMoEWrapper(layer_idx=0, num_experts=1, num_experts_per_tok=1,
        hidden_size=2048, moe_intermediate_size=512, gpu_experts_mask=torch.zeros(1, dtype=torch.bool),
        cpuinfer_threads=64, threadpool_count=2, numa_nodes=[4,5], weight_path=str(path),
        chunked_prefill_size=64, max_deferred_experts_per_token=0, method='LLAMAFILE')
    cpu.load_weights(torch.arange(1, dtype=torch.int32))
    report = dict(scope='SINGLE_EXPERT_NOT_256_EXPERT_MODEL', cpu_workers=64,
                  threadpool_count=2, numa_nodes=[4,5], gguf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=[])
    for qlen in (1,8,64):
        x = torch.randn(qlen,2048).bfloat16()
        ids = torch.zeros(qlen,1,dtype=torch.int64)
        routes = torch.ones(qlen,1)
        g = (x.float()@gate.T).bfloat16().float()
        u = (x.float()@up.T).bfloat16().float()
        ref = ((torch.nn.functional.silu(g)*u).bfloat16().float()@down.T).bfloat16().float()
        outputs = [cpu.forward(x,ids,routes).detach().cpu().clone() for _ in range(10)]
        finite = all(torch.isfinite(y).all().item() for y in outputs)
        exact = all(torch.equal(outputs[0],y) for y in outputs[1:])
        rel = ((outputs[0].double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-30)).item()
        report['rows'].append(dict(qlen=qlen,finite=finite,repeat_exact=exact,relative_l2=rel))
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        assert finite and exact and rel<=.01, 'LOCAL_OPERATOR_SAFETY_FAILURE'
    report['status']='PASS_SINGLE_EXPERT_C64'
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
