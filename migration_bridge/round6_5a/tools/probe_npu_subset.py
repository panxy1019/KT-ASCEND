"""Diagnostic only: invoke deployed forward_npu with 64/32 physical layouts.

No model patches, no new tolerance. Any non-exact layout result requires review.
Selection is frozen algorithmically before inspecting numerical outputs.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.hardware_backend.npu.utils import npu_format_cast


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(a, b):
    a, b = a.float(), b.float()
    d = a - b
    return dict(exact=torch.equal(a, b), max_abs=d.abs().max().item(),
                relative_l2=(d.norm() / b.norm().clamp_min(1e-30)).item(),
                cosine=torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    fn = UnquantizedFusedMoEMethod.forward_npu
    source = Path(inspect.getsourcefile(fn))
    report = dict(status='RUNNING', scope='DEPLOYED_GROUPED_MOE_LAYOUT_DIAGNOSTIC',
                  source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  method_sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest(),
                  selection_seed=6501, repeats=10, layers=[], rows=[])
    method = NS(with_bias=False, moe_runner_config=NS(activation='silu'),
                _kt_return_fp32_contribution=True)
    resident = list(range(0, 64, 2))
    try:
        for layer in (1, 9, 17, 26):
            stage = args.trace / 'hybrid/stagewise'
            def capture(name):
                return torch.load(stage / f'model_layer{layer:02d}_{name}_call0055.pt',
                                  map_location='cpu', weights_only=False)['tensor']
            hidden = capture('mlp_input').contiguous()
            scores = capture('router_logits').float().reshape(-1, 64)[-1]
            counts = Counter()
            for path in sorted(stage.glob(f'model_layer{layer:02d}_router_topk_ids_call*.pt')):
                counts.update(torch.load(path, map_location='cpu', weights_only=False)['tensor'].flatten().tolist())
            hot = sorted(resident, key=lambda e: (-counts[e], e))[:4]
            ranked = torch.argsort(scores, descending=True, stable=True).tolist()
            # If actual rank6/7 are CPU-owned, select nearest resident ranks;
            # record actual boundary separately instead of relabeling them.
            boundary = sorted(set(resident) - set(hot), key=lambda e: (min(abs(ranked.index(e)-5), abs(ranked.index(e)-6)), e))[:2]
            controls = random.Random(6501 + layer).sample(sorted(set(resident)-set(hot)-set(boundary)), 2)
            selected = hot + boundary + controls
            report['layers'].append(dict(layer=layer, hot=hot, nearest_resident_boundary=boundary,
                actual_rank6_7=ranked[5:7], controls=controls, selected=selected,
                hidden_sha256=digest(hidden), hidden_shape=list(hidden.shape),
                hidden_dtype=str(hidden.dtype)))
            (args.output / 'progress.json').write_text(json.dumps(report, indent=2)+'\n')
            w13, w2 = [], []
            for expert in range(64):
                prefix = f'model.layers.{layer}.mlp.experts.{expert}.'
                weights = {}
                for projection in ('gate', 'up', 'down'):
                    key = prefix + projection + '_proj.weight'
                    with safe_open(args.model / index[key], framework='pt', device='cpu') as sf:
                        weights[projection] = sf.get_tensor(key).clone()
                w13.append(torch.cat([weights['gate'], weights['up']], dim=0))
                w2.append(weights['down'])
            full13, full2 = torch.stack(w13), torch.stack(w2)
            del w13, w2, weights
            # Same transpose/ND-format boundary as deployed process_weights_after_loading.
            full = NS(w13_weight=npu_format_cast(full13.to('npu').transpose(1, 2)),
                      w2_weight=npu_format_cast(full2.to('npu').transpose(1, 2)), top_k=6)
            compact = NS(w13_weight=npu_format_cast(full13[resident].to('npu').transpose(1, 2)),
                         w2_weight=npu_format_cast(full2[resident].to('npu').transpose(1, 2)), top_k=6)
            x = hidden.to('npu')
            for expert in selected:
                outputs = []
                hashes = []
                for layout, slot in ((full, expert), (compact, resident.index(expert))):
                    ids = torch.full((len(hidden), 6), -1, dtype=torch.int32, device='npu')
                    ids[:, 0] = slot
                    weights = torch.zeros((len(hidden), 6), dtype=torch.float32, device='npu')
                    weights[:, 0] = 1
                    dispatch = NS(hidden_states=x, topk_output=(weights, ids, None))
                    repeated = [fn(method, layout, dispatch).hidden_states.cpu() for _ in range(10)]
                    hashes.append([digest(y) for y in repeated])
                    outputs.append(repeated[0])
                row = dict(layer=layer, expert=expert, physical_slots=[expert, resident.index(expert)],
                           **metrics(outputs[0], outputs[1]), repeat_hashes=hashes,
                           deterministic=all(len(set(h)) == 1 for h in hashes))
                report['rows'].append(row)
                torch.save(dict(r64=outputs[0], r32=outputs[1]), args.output/f'L{layer}-E{expert}.pt')
                print(json.dumps({k:v for k,v in row.items() if k != 'repeat_hashes'}), flush=True)
            del full, compact, full13, full2, x
            torch.npu.empty_cache()
        report['status'] = ('LAYOUT_EXACT_PASS' if all(r['exact'] and r['finite'] and r['deterministic']
                                                     for r in report['rows']) else 'LAYOUT_REVIEW_REQUIRED')
    except BaseException as exc:
        report.update(status='PROBE_ERROR', error=repr(exc))
        raise
    finally:
        (args.output / 'npu_subset_equivalence.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
