"""Real-weight synthetic-operand reference lanes; not production CPU/GMM proof."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from safetensors.torch import load_file


def metrics(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    delta = a - b
    return dict(maxabs=delta.abs().max().item(),
                relative_l2=(delta.norm()/b.norm().clamp_min(1e-12)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(32)
    torch.manual_seed(8675309)
    weights = load_file(args.sample)
    gate, up = weights['gate_up_proj'].chunk(2, dim=-2)
    down = weights['down_proj']
    ng, nu, nd = [w.bfloat16().npu() for w in (gate, up, down)]
    rows = []
    for qlen in (1, 8, 64):
        x = torch.randn(qlen, 2048).bfloat16()
        cg, cu = x.float() @ gate.T, x.float() @ up.T
        ca = torch.nn.functional.silu(cg.bfloat16().float()) * cu.bfloat16().float()
        co = (ca.bfloat16().float() @ down.T).bfloat16()
        nx = x.npu()
        g, u = nx @ ng.T, nx @ nu.T
        activation = torch.nn.functional.silu(g) * u
        output = activation @ nd.T
        torch.npu.synchronize()
        repeats = [((torch.nn.functional.silu(nx @ ng.T) * (nx @ nu.T)) @ nd.T).cpu() for _ in range(3)]
        finite = all(torch.isfinite(t).all().item() for t in (cg, cu, ca, co, g, u, activation, output))
        exact = all(torch.equal(repeats[0], item) for item in repeats[1:])
        rows.append(dict(qlen=qlen, finite=finite, npu_repeat_exact=exact,
                         gate=metrics(cg.bfloat16(), g), up=metrics(cu.bfloat16(), u),
                         activation=metrics(ca.bfloat16(), activation), output=metrics(co, output)))
        if not finite or not exact:
            raise RuntimeError('reference lane safety failure')
    result = dict(scope='REAL_WEIGHT_SYNTHETIC_OPERAND_REFERENCE_ONLY', seed=8675309,
                  cpu_threads=32, production_c64_qualified=False,
                  production_llamafile_qualified=False, grouped_npu_moe_qualified=False,
                  rows=rows)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
