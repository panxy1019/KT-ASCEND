"""Credit wrapper around the existing synchronous BF16 migration adapter.

It reserves bytes before any read/copy and drains the actual adapter before
returning credits. It does not implement a second migration engine or promise
bandwidth/fairness. Resource denial is immediate and fails the calling commit.
"""
from .framework_config import BufferIdentity
from .host_budget import HostWorkBudget,Priority


class BudgetedMigration:
    def __init__(self,adapter,budget,context,*,requests=None):
        if not isinstance(budget,HostWorkBudget) or not budget.enabled or not callable(context):
            raise ValueError('migration requires an enabled shared budget and live context')
        if type(adapter.expert_bytes) is not int or adapter.expert_bytes<=0:
            raise ValueError('migration source geometry is missing')
        self.adapter,self.budget,self.context=adapter,budget,context
        if requests is not None and not isinstance(requests,set):raise TypeError('shared request ownership must be a set')
        self.requests=set() if requests is None else requests
        self.state='READY'

    def __getattr__(self,name):
        return getattr(self.adapter,name)

    def copy_expert(self,layer,slot,logical):
        return self.copy_experts(layer,((slot,logical),))

    def copy_experts(self,layer,moves):
        if self.state!='READY':raise RuntimeError(self.state)
        moves=tuple(moves)
        if (type(layer) is not int or layer<0 or
                any(not isinstance(m,(tuple,list)) or len(m)!=2 or
                    any(type(v) is not int or v<0 for v in m) for m in moves) or
                len({m[0] for m in moves})!=len(moves)):
            raise ValueError('invalid migration batch')
        if not moves:return
        rid,epoch=self.context();BufferIdentity(rid,epoch,0)
        allocation=self.budget.try_acquire(request_id=rid,placement_epoch=epoch,
            priority=Priority.EXPERT_MIGRATION,host_bytes=4*self.adapter.expert_bytes,
            h2d_bytes=len(moves)*self.adapter.expert_bytes,pinned_bytes=0)
        if allocation is None:raise RuntimeError('joint host/H2D budget unavailable before migration')
        self.requests.add(rid)
        try:self.budget.mark_in_flight(allocation)
        except BaseException:
            self.budget.release(allocation)
            raise
        primary=None
        try:self.adapter.copy_experts(layer,moves)
        except BaseException as error:primary=error
        try:self.adapter.synchronize()
        except BaseException as cleanup:
            # Copy or synchronization may still own staging. Keep credits until
            # process teardown; no subsequent copy may reuse this adapter.
            self.state='BROKEN_RELOAD_REQUIRED'
            if primary is not None:raise primary from cleanup
            raise
        self.budget.complete(allocation);self.budget.release(allocation)
        if primary is not None:
            self.state='BROKEN_RELOAD_REQUIRED'
            raise primary
