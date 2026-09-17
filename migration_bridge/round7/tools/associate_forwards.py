"""Exact request/history association; queued unused last-token forwards are raw evidence.

The production SGLang overlap scheduler may compute a final unused decode pass.
Do not silently count it as an evaluated generation position or accept unknown
extra passes. Every pass must match an exact frozen prompt or returned-token
history; raw stats and discarded indices are retained.
"""


def associate(metadata,requests):
    prefill=[False]*len(requests)
    decode=[set() for _ in requests]
    keep,unused=[],[]
    current=None
    for index,step in enumerate(metadata):
        tokens=step['input_ids'];positions=step['positions']
        matches=[]
        for r,request in enumerate(requests):
            ids=request['input_ids'];out=request['output_ids'];n=len(ids)
            if tokens==ids and positions==list(range(n)):
                matches.append((r,'prefill',None))
            elif r==current and len(tokens)==1 and len(positions)==1:
                j=positions[0]-n
                if 0<=j<len(out) and tokens==[out[j]]:
                    matches.append((r,'decode',j))
        # Identical reused prompts are not allowed in a collection epoch.
        if len(matches)!=1:raise ValueError(('UNATTRIBUTED_OR_AMBIGUOUS_FORWARD',index,step,matches))
        r,kind,j=matches[0]
        if kind=='prefill':
            if prefill[r]:raise ValueError('DUPLICATE_PREFILL')
            if r!=sum(prefill):raise ValueError('PREFILL_REQUEST_ORDER')
            current=r
            prefill[r]=True;keep.append(index)
        else:
            if not prefill[r]:raise ValueError('DECODE_BEFORE_PREFILL')
            if j in decode[r]:raise ValueError('DUPLICATE_DECODE')
            decode[r].add(j)
            if j==len(requests[r]['output_ids'])-1:unused.append(index)
            else:keep.append(index)
    for r,request in enumerate(requests):
        required=set(range(len(request['output_ids'])-1))
        if not prefill[r] or not required<=decode[r]:raise ValueError('MISSING_REQUEST_FORWARD')
    return dict(kept_indices=keep,unused_final_token_indices=unused,
                total_raw_passes=len(metadata),attributed_passes=len(keep)+len(unused),
                requests=len(requests),unknown_passes=0)
