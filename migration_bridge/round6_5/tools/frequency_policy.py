"""One preregistered static count-frequency policy; no runtime migration."""


def resident_ids(counts,budget):
    if len(counts)!=64 or budget not in (8,16,32) or any(c<0 for c in counts):
        raise ValueError('FREQUENCY_INPUT')
    return sorted(sorted(range(64),key=lambda e:(-counts[e],e))[:budget])


def project(position,masks):
    events=0;weight=0.;total=0.;touched=False
    if set(position)!=set(range(1,27)) or set(masks)!=set(position):raise ValueError('LAYER_COVERAGE')
    for layer,row in position.items():
        ids,weights=row['ids'],row['weights']
        if len(ids)!=6 or len(weights)!=6 or len(set(ids))!=6:raise ValueError('TOPK_CONTRACT')
        for expert,w in zip(ids,weights):
            if not 0<=expert<64:raise ValueError('EXPERT_ID')
            total+=w
            if expert not in masks[layer]:events+=1;weight+=w;touched=True
    return dict(cpu_events=events,cpu_weight_mass=weight,total_weight_mass=total,touches_cpu=touched)
