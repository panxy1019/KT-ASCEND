"""Two fixed CPU-owned experts/layer on three real chronological decode states."""
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
from common import sha, validate_profile, write


def digest(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(x, y):
    x, y = x.double().cpu(), y.double().cpu()
    difference = x - y
    return dict(relative_l2=float(difference.norm() / y.norm().clamp_min(1e-30)),
                max_abs=float(difference.abs().max()), mean_abs=float(difference.abs().mean()),
                finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--operands', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); data = json.loads(a.profile.read_text()); validate_profile(data)
    row = data['rows'][a.layer-1]
    if a.layer not in range(1, 27) or not {1, 63} <= set(row['cpu_ids']):
        raise ValueError('FIXED_EXPERT_OWNERSHIP')
    torch.set_num_threads(1)
    model = Path('/workspace/models/DeepSeek-V2-Lite-604d5664')
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    mask = torch.ones(64, dtype=torch.bool); mask[row['cpu_ids']] = False
    LlamafileMoEWrapper._gguf_loader_instance = None
    LlamafileMoEWrapper._gguf_loader_path = None
    cpu = KTMoEWrapper(layer_idx=a.layer, num_experts=64, num_experts_per_tok=6,
        hidden_size=2048, moe_intermediate_size=1408, gpu_experts_mask=mask,
        cpuinfer_threads=32, threadpool_count=1,
        weight_path='/workspace/results/r65-assets-v1/all-moe-f32.gguf',
        chunked_prefill_size=16, max_deferred_experts_per_token=0, method='LLAMAFILE', numa_nodes=[0])
    cpu.load_weights(torch.arange(64, dtype=torch.int32))
    method = NS(with_bias=False, moe_runner_config=NS(activation='silu'), _kt_return_fp32_contribution=True)
    report = dict(status='RUNNING', layer=a.layer, rows=[], profile_sha256=sha(a.profile),
                  nohit_scope='CONTROLLED_NPU_OWNED_ROUTE_ON_REAL_CAPTURED_OPERAND_NOT_NATURAL_ROUTE',
                  control_rows=0, references='RETAINED_SGLANG_ASCEND_GROUPED_MOE')
    try:
        for expert in (1, 63):
            weights = {}
            for projection in ('gate', 'up', 'down'):
                key = f'model.layers.{a.layer}.mlp.experts.{expert}.{projection}_proj.weight'
                with safe_open(model / index[key], framework='pt', device='cpu') as reader:
                    weights[projection] = reader.get_tensor(key).clone()
                if weights[projection].dtype != torch.bfloat16: raise ValueError('SOURCE_DTYPE')
            npu = NS(w13_weight=npu_format_cast(torch.cat([weights['gate'], weights['up']])[None].to('npu').transpose(1, 2)),
                     w2_weight=npu_format_cast(weights['down'][None].to('npu').transpose(1, 2)), top_k=6)
            for call in (1, 4, 7):
                path = a.operands / f'L{a.layer:02d}-C{call}.pt'
                value = torch.load(path, map_location='cpu', weights_only=False)
                x = value['tensor']
                if value['layer'] != a.layer or value['call'] != call or x.dtype != torch.bfloat16 or tuple(x.shape) != (1, 2048):
                    raise ValueError('REAL_OPERAND')
                ids = torch.full((1, 6), -1, dtype=torch.int64); ids[:, 0] = expert
                weight = torch.zeros((1, 6), dtype=torch.float32); weight[:, 0] = 1
                outputs = [cpu.forward(x, ids, weight).clone() for _ in range(3)]
                hashes = [digest(out) for out in outputs]
                physical = ids.clone().int(); physical[:, 0] = 0
                ref = UnquantizedFusedMoEMethod.forward_npu(method, npu,
                    NS(hidden_states=x.to('npu'), topk_output=(weight.to('npu'), physical.to('npu'), None))).hidden_states.cpu()
                measured = metrics(outputs[0], ref)
                record = dict(expert=expert, call=call, operand_sha256=sha(path), input_tensor_sha256=digest(x),
                    source_weight_sha256={k:digest(v) for k,v in weights.items()}, hashes=hashes, metrics=measured)
                report['rows'].append(record)
                if not measured['finite'] or measured['relative_l2'] > .01 or len(set(hashes)) != 1:
                    raise ValueError('LOCAL_OPERATOR_GATE')
                nohit = ids.clone(); nohit[:, 0] = row['npu_ids'][0]
                zero = cpu.forward(x, nohit, weight).clone()
                if not torch.equal(zero, torch.zeros_like(zero)):
                    raise ValueError('CPU_NOT_HIT_CONTAMINATION')
                if not torch.equal((ref.float()+zero.float()).bfloat16(), ref.bfloat16()):
                    raise ValueError('CONTROLLED_MERGE_CONTAMINATION')
                report['control_rows'] += 1
                write(a.output, report)
        report['status'] = 'LOCAL_OPERATOR_AND_CONTROL_PASS'
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc)); raise
    finally: write(a.output, report)


if __name__ == '__main__': main()
