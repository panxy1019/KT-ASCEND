"""Same captured operands, per-expert CPU/GMM, canonical global-ID reduction.

Diagnostic output only: non-exactness is localized, never automatically accepted.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from probe_npu_subset import digest, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--gguf', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--call', type=int, default=55)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    index = json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    cfg = json.loads((a.model/'config.json').read_text())
    report = dict(status='RUNNING', scope='DIAGNOSTIC_NOT_QUALIFICATION', rows=[],
                  scaling_factor=cfg['routed_scaling_factor'], workers=32, threadpool_count=1,
                  call=a.call, reduction='global logical ID ascending FP32; BF16 route weights; final BF16 cast')
    def load(layer, source, stage):
        path = a.trace/f'hybrid/stagewise/{source}_layer{layer:02d}_{stage}_call{a.call:04d}.pt'
        return torch.load(path, map_location='cpu', weights_only=False)['tensor']
    fn = UnquantizedFusedMoEMethod.forward_npu
    method = NS(with_bias=False, moe_runner_config=NS(activation='silu'), _kt_return_fp32_contribution=True)
    try:
        for layer in (1, 9, 17, 26):
            x = load(layer, 'cpu', 'input_hidden_after_sync').contiguous()
            ids = load(layer, 'model', 'router_topk_ids').long()
            rw = load(layer, 'model', 'router_weights').float()
            if x.ndim!=2 or x.shape[1]!=2048 or ids.shape!=(len(x),6):
                raise ValueError('FAILURE_OPERAND_SHAPE')
            mask = torch.tensor([e % 2 == 0 for e in range(64)], dtype=torch.bool)
            LlamafileMoEWrapper._gguf_loader_instance = None
            LlamafileMoEWrapper._gguf_loader_path = None
            cpu = KTMoEWrapper(layer_idx=layer, num_experts=64, num_experts_per_tok=6,
                hidden_size=2048, moe_intermediate_size=1408, gpu_experts_mask=mask,
                cpuinfer_threads=32, threadpool_count=1, weight_path=str(a.gguf),
                chunked_prefill_size=16, max_deferred_experts_per_token=0,
                method='LLAMAFILE', numa_nodes=[0])
            cpu.load_weights(torch.arange(64, dtype=torch.int32))
            original_cpu = cpu.forward(x, ids, rw).clone().float()
            repeat_hashes=[digest(original_cpu)]
            for _ in range(9):repeat_hashes.append(digest(cpu.forward(x,ids,rw).clone().float()))
            if len(set(repeat_hashes))!=1:raise ValueError('CPU_BATCH_NONDETERMINISM')
            gateup, down = [], []
            for e in range(64):
                projections = {}
                for name in ('gate', 'up', 'down'):
                    key = f'model.layers.{layer}.mlp.experts.{e}.{name}_proj.weight'
                    with safe_open(a.model/index[key], framework='pt', device='cpu') as sf:
                        projections[name] = sf.get_tensor(key).clone()
                gateup.append(torch.cat([projections['gate'], projections['up']]))
                down.append(projections['down'])
            # R64 real weights supply both the resident oracle and debug-only CPU reference.
            full = NS(w13_weight=npu_format_cast(torch.stack(gateup).to('npu').transpose(1,2)),
                      w2_weight=npu_format_cast(torch.stack(down).to('npu').transpose(1,2)), top_k=6)
            del gateup, down, projections
            xn = x.to('npu')
            canonical = torch.zeros_like(x, dtype=torch.float32)
            canonical_cpu = torch.zeros_like(canonical)
            canonical_npu = torch.zeros_like(canonical)
            all_npu = torch.zeros_like(canonical)
            experts = []
            for e in sorted(set(ids.flatten().tolist())):
                selected=(ids==e)
                active=selected.any(dim=1)
                single = torch.full_like(ids, -1); single[active,0] = e
                unit = torch.zeros_like(rw); unit[active,0] = 1
                n = fn(method, full, NS(hidden_states=xn,
                    topk_output=(unit.to('npu'), single.to('npu', dtype=torch.int32), None))).hidden_states.cpu().float()
                value = cpu.forward(x, single, unit).clone().float() if e % 2 else n
                weight = (rw.to(torch.bfloat16).float()*selected).sum(dim=1,keepdim=True)
                weighted = value * weight
                canonical += weighted
                all_npu += n * weight
                if e % 2: canonical_cpu += weighted
                else: canonical_npu += weighted
                experts.append(dict(expert=e, cpu_owned=bool(e % 2), route_weight=weight.tolist(),
                    output_sha256=digest(value), npu_sha256=digest(n), **metrics(value,n)))
            production_cpu = load(layer,'ktep','cpu_partial').float()
            production_npu = load(layer,'ktep','npu_owned_partial').float()
            production = load(layer,'ktep','routed_merge')
            row = dict(layer=layer, hidden_sha256=digest(x), ids=ids.tolist(), cpu_repeat_hashes=repeat_hashes,
                routing_weights_sha256=digest(rw), expert_rows=experts,
                cpu_replay_vs_capture=metrics(original_cpu,production_cpu),
                canonical_cpu_vs_capture=metrics(canonical_cpu,production_cpu),
                canonical_npu_vs_capture=metrics(canonical_npu,production_npu),
                canonical_final_vs_production=metrics(canonical.to(production.dtype),production),
                split_canonical_final_vs_production=metrics((canonical_cpu+canonical_npu).to(production.dtype),production),
                same_input_all_npu_vs_hybrid=metrics(canonical.to(production.dtype),all_npu.to(production.dtype)))
            torch.save(dict(canonical=canonical, canonical_cpu=canonical_cpu, canonical_npu=canonical_npu,
                production=production, all_npu=all_npu),a.output/f'L{layer}.pt')
            report['rows'].append(row)
            print(json.dumps(row),flush=True)
            (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            del cpu, full, xn
            torch.npu.empty_cache()
        report['status']='CANONICAL_REVIEW_REQUIRED'
    except BaseException as exc:
        report.update(status='PROBE_ERROR',error=repr(exc));raise
    finally:
        (a.output/'canonical_mixed_oracle.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
