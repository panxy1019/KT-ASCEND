"""Fail-closed tombstone for the rejected NZ29 transpose-view experiment.

S3 v25 produced relative_l2=0.9835492231451818 against tolerance0.01.
The experimental implementation is archived under artifacts/round9-autorun/
rejected-candidate and the immutable component snapshot, never a runtime option.
No production caller enables this helper. OFF preserves the original operation.
"""
STATUS = 'REJECTED_NUMERIC'

def update_parent(destination,index,staged,*,enabled=False,get_format):
    if type(enabled) is not bool:
        raise ValueError('explicit boolean candidate flag required')
    if enabled:
        raise RuntimeError('REJECTED_NUMERIC: NZ29 transpose-view update is disabled')
    return destination.index_copy_(0,index,staged)
