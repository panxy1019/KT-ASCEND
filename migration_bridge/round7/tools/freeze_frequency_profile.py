"""Freeze deployed official global top-K, preserve dense-layer exclusion."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    import torch
    from kt_kernel import generate_gpu_experts_masks
    base=Path(sys.argv[1]); profiles=base/'profiles'
    summary=json.loads((base/'dev-stat-summary.json').read_text())
    assert summary['status']=='DEV_STAT_COMPLETE'
    src=base/'SCHED_DEV_activation_stats.pt'
    assert hashlib.sha256(src.read_bytes()).hexdigest()==summary['activation_sha256']
    data=torch.load(src,map_location='cpu',weights_only=False)
    freq=data['logical_count'].sum(dim=0).float()
    assert freq.shape==(27,64) and freq[0].sum()==0 and torch.isfinite(freq).all()
    mask=generate_gpu_experts_masks(freq,208)
    assert mask.shape==(27,64) and mask.dtype==torch.bool
    assert mask[0].sum()==0 and int(mask[1:].sum())==208
    rows=[]
    for layer in range(1,27):
        ids=torch.where(mask[layer])[0].tolist()
        rows.append(dict(layer=layer,npu_ids=ids,cpu_ids=sorted(set(range(64))-set(ids))))
    # Keep the original rank3 frequencies: the deployed loader independently
    # reconstructs the same mask, then the observer checks the real mask.
    import shutil
    shutil.copy2(src,profiles/'frequency.pt')
    result=dict(name='FREQUENCY_R8_C64_EQUIVALENT',total_resident=208,rows=rows,
        source_sha256=summary['activation_sha256'],per_layer_budget=[len(r['npu_ids']) for r in rows],
        zero_resident_layers=[r['layer'] for r in rows if not r['npu_ids']],
        fitting_corpus='SCHED_DEV_ONLY',quality_selection=False,
        activation_sha256=hashlib.sha256((profiles/'frequency.pt').read_bytes()).hexdigest())
    (profiles/'frequency.json').write_text(json.dumps(result,indent=2)+'\n')
    (base/'frequency-candidate-freeze.json').write_text(json.dumps(dict(
        profile_sha256=hashlib.sha256((profiles/'frequency.json').read_bytes()).hexdigest(),
        activation_sha256=result['activation_sha256'],total_resident=208,
        per_layer_budget=result['per_layer_budget'],status='MASK_FROZEN_NOT_QUALIFIED'),indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
