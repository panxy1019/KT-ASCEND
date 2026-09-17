"""Pure-Python scheduler invariants, independent of torch/device imports."""


def validate_mask(rows, budget=208):
    if [r['layer'] for r in rows]!=list(range(1,27)):
        raise ValueError('LAYER_SCOPE')
    total=0
    for row in rows:
        npu,cpu=row['npu_ids'],row['cpu_ids']
        if npu!=sorted(set(npu)) or cpu!=sorted(set(cpu)):
            raise ValueError('DUPLICATE_OR_UNSORTED_IDS')
        if set(npu)&set(cpu) or set(npu)|set(cpu)!=set(range(64)):
            raise ValueError('OWNERSHIP_PARTITION')
        total+=len(npu)
    if total!=budget: raise ValueError('GLOBAL_BUDGET')
    return total


def stable_frequency_order(counts, slots):
    if len(counts)!=64 or not 0<=slots<=64 or any(x<0 for x in counts):
        raise ValueError('FREQUENCY_SCOPE')
    return sorted(sorted(range(64),key=lambda i:(-counts[i],i))[:slots])


def validate_generation(weights,mapping,mask,wrapper_mask):
    if not weights==mapping==mask==wrapper_mask:
        raise ValueError('GENERATION_MISMATCH')


def quality_gate(correct, invalid, baseline=174, count=256):
    if count!=256 or not 0<=correct<=count: raise ValueError('CORPUS_SCOPE')
    return invalid==0 and 100*(correct-baseline)/count>=-5
