"""Serial request-boundary placement transactions; failure requires model reload.

All inference entry points must acquire a request lease. The caller owns actual
KV/GDN/PLE cleanup and must confirm consumers drained before finishing the lease.
No automatic scheduler hook and no token-level update is installed here.
"""
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class LayerPlan:
    layer: int
    generation: int
    observation_epoch: int
    previous: tuple
    slots: tuple
    moves: tuple


class PlacementController:
    def __init__(self,residents,adapter,*,experts=512,resident=64):
        if type(experts) is not int or type(resident) is not int or not 0<resident<experts:
            raise ValueError('invalid expert/resident geometry')
        self.experts=experts;self.resident=resident;self.adapter=adapter
        self._residents={k:tuple(v) for k,v in residents.items()}
        if not self._residents:raise ValueError('no layers')
        for layer,slots in self._residents.items():
            if type(layer) is not int or layer<0:raise ValueError('invalid layer')
            self._validate_slots(slots)
        self.generation=0;self.state='READY';self._active=None;self._counts={};self._observation_epoch=0;self._lock=threading.RLock()

    def _validate_slots(self,slots):
        if len(slots)!=self.resident or len(set(slots))!=self.resident or any(type(x) is not int or not 0<=x<self.experts for x in slots):
            raise ValueError('invalid/duplicate logical resident IDs')

    def _ready(self):
        if self.state!='READY':raise RuntimeError(self.state)

    def begin(self,request_id):
        with self._lock:
            self._ready()
            if not isinstance(request_id,str) or not request_id:raise ValueError('invalid request ID')
            if self._active is not None:raise RuntimeError('only one request lease supported')
            self._active=request_id;self._prefill_committed=False;return self.generation

    def finish(self,request_id,*,consumers_drained):
        with self._lock:
            self._ready()
            if self._active is None or self._active!=request_id:raise ValueError('request lease mismatch')
            if consumers_drained is not True:raise RuntimeError('KV/GDN/PLE/device consumers not drained')
            self._active=None

    def observe(self,request_id,generation,layer,logical_ids):
        with self._lock:
            self._ready()
            if self._active is None or request_id!=self._active or type(generation) is not int or generation!=self.generation:
                raise RuntimeError('stale or unleased route observation')
            ids=list(logical_ids)
            if type(layer) is not int or layer not in self._residents or any(type(i) is not int or not 0<=i<self.experts for i in ids):raise ValueError('invalid observed logical IDs')
            counts=self._counts.setdefault(layer,[0]*self.experts)
            for logical in ids:counts[logical]+=1
            if ids:self._observation_epoch+=1

    def plan(self):
        return self._plan()

    def _plan(self,*,allow_live_prefill=False):
        with self._lock:
            self._ready()
            if self._active is not None and not allow_live_prefill:raise RuntimeError('request still active')
            plans=[]
            for layer,counts in sorted(self._counts.items()):
                if not any(counts):continue
                chosen=set(sorted(range(self.experts),key=lambda i:(-counts[i],i))[:self.resident])
                old=self._residents[layer];incoming=iter(sorted(chosen-set(old)))
                slots=tuple(logical if logical in chosen else next(incoming) for logical in old)
                moves=tuple((slot,logical) for slot,logical in enumerate(slots) if old[slot]!=logical)
                if moves:plans.append(LayerPlan(layer,self.generation,self._observation_epoch,old,slots,moves))
            return tuple(plans)

    def commit(self,plans):
        return self._commit(plans)

    def _commit(self,plans,*,allow_live_prefill=False):
        with self._lock:
            self._ready()
            if self._active is not None and not allow_live_prefill:raise RuntimeError('request still active')
            plans=tuple(plans)
            if plans!=self._plan(allow_live_prefill=allow_live_prefill):raise ValueError('stale/incomplete or forged migration plan')
            seen=set()
            for plan in plans:
                if not isinstance(plan,LayerPlan) or plan.layer in seen or plan.layer not in self._residents:raise ValueError('invalid layer plan')
                seen.add(plan.layer);self._validate_slots(plan.slots)
                if plan.generation!=self.generation or plan.observation_epoch!=self._observation_epoch or plan.previous!=self._residents[plan.layer]:raise ValueError('stale generation/placement')
                expected=tuple((i,x) for i,x in enumerate(plan.slots) if x!=plan.previous[i])
                if not expected or plan.moves!=expected:raise ValueError('incomplete migration diff')
            if not plans:
                self._counts.clear();return self.generation
            self.state='MIGRATING';next_generation=self.generation+1
            try:
                self.adapter.synchronize()
                for plan in plans:
                    batch_copy=getattr(self.adapter,'copy_experts',None)
                    if batch_copy is not None:batch_copy(plan.layer,plan.moves)
                    else:
                        for slot,logical in plan.moves:self.adapter.copy_expert(plan.layer,slot,logical)
                self.adapter.synchronize()
                for plan in plans:self.adapter.publish(plan.layer,plan.slots,next_generation)
                self.adapter.synchronize()
                for plan in plans:self._residents[plan.layer]=plan.slots
                self.generation=next_generation;self._counts.clear();self.state='READY'
                return self.generation
            except BaseException:
                # Device writes may already have happened. Metadata rollback alone
                # cannot restore old weights, therefore there is no retry/reset API.
                self.state='BROKEN_RELOAD_REQUIRED'
                raise

    def commit_prefill(self,request_id,generation,statistics,*,placement_consumers_drained):
        """Opt-in live barrier; preserve the active request and reuse one engine.

        Caller synchronizes old-placement consumers, verifies the same KV/GDN/PLE
        pools stay live, and invokes this only after the final prefill chunk. This
        API does not release, reset or allocate request state. The already sampled
        first output token belongs to the previous placement epoch.
        """
        from .route_stats import frequency_mask
        with self._lock:
            self._ready()
            if (self._active is None or self._active!=request_id or
                    type(generation) is not int or generation!=self.generation or self._prefill_committed):
                raise RuntimeError('stale, finished or already committed prefill lease')
            if placement_consumers_drained is not True:
                raise RuntimeError('old-placement consumers must drain without releasing request state')
            if (not isinstance(statistics,dict) or statistics.get('request_id')!=request_id or
                    type(statistics.get('generation')) is not int or statistics['generation']!=generation or
                    statistics.get('completed_phases')!=1 or statistics.get('experts')!=self.experts or
                    statistics.get('layers')!=len(self._residents) or
                    set(self._residents)!=set(range(len(self._residents)))):
                raise ValueError('complete source-bound prefill statistics required')
            frequency_mask(statistics,self.resident)  # geometry/token conservation validation
            if (set(self._counts)!=set(self._residents) or
                    any(self._counts[layer]!=statistics['counts'][layer] for layer in self._residents)):
                raise ValueError('prefill statistics differ from leased logical routes')
            plans=self._plan(allow_live_prefill=True)
            result=self._commit(plans,allow_live_prefill=True)
            self._prefill_committed=True
            return result
