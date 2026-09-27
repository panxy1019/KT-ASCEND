"""Opt-in live-state-preserving barrier using the existing migration engine."""
from .dynamic_boundary import RequestBoundary


class PrefillEndBoundary(RequestBoundary):
    def __init__(self, controller, live_state, event=None):
        super().__init__(controller, event)
        self.live_state = live_state
        self.state = 'IDLE'

    def before(self, requests):
        if self.state == 'BROKEN_RELOAD_REQUIRED': raise RuntimeError(self.state)
        fresh = self.request is None
        super().before(requests)
        if fresh: self.state = 'PREFILL_ACTIVE'

    def prefill_complete(self, statistics):
        req = self.request
        if req is None or self.state != 'PREFILL_ACTIVE':
            raise RuntimeError('prefill barrier is duplicate, stale or unleased')
        if req.finished(): return False
        if req.req_pool_idx is None or req.mamba_pool_idx is None:
            raise RuntimeError('prefill barrier requires live KV/GDN/PLE allocation')
        if len(req.output_ids) != 1:
            raise RuntimeError('prefill barrier must follow exactly one sampled output token')
        old_generation = self.generation
        try:
            self.state = 'PREFILL_DRAINING'
            self.controller.adapter.synchronize()
            before = self.live_state(req)
            old_residents = dict(self.controller._residents)
            self.state = 'MIGRATION_ACTIVE'
            generation = self.controller.commit_prefill(req.rid, old_generation, statistics,
                                                        placement_consumers_drained=True)
            if self.live_state(req) != before:
                raise RuntimeError('prefill migration changed live KV/GDN/PLE state')
            changes = [dict(layer=layer, previous=list(old_residents[layer]), slots=list(slots),
                            moves=[[slot, logical] for slot, logical in enumerate(slots)
                                   if old_residents[layer][slot] != logical])
                       for layer, slots in sorted(self.controller._residents.items())
                       if slots != old_residents[layer]]
            self.generation = generation
            self.state = 'DECODE_ACTIVE'
            self.event(dict(status='PREFILL_END_COMMIT', request_id=req.rid,
                first_output_token_generation=old_generation, decode_generation=generation,
                generation=generation, live_state_preserved=True, live_state=before,
                changed_layers=len(changes), copied_experts=sum(len(p['moves']) for p in changes),
                plans=changes))
            return True
        except BaseException:
            self.state = self.controller.state = 'BROKEN_RELOAD_REQUIRED'
            raise

    def after(self):
        if self.state == 'BROKEN_RELOAD_REQUIRED': raise RuntimeError(self.state)
        req = self.request
        if req is None or not req.finished(): return
        if req.req_pool_idx is not None or req.mamba_pool_idx is not None:
            raise RuntimeError('finished prefill-end request consumers are not released')
        try:
            self.controller.adapter.synchronize()
            self.controller.finish(req.rid, consumers_drained=True)
            self.controller.discard_observations()
            self.event(dict(status='PREFILL_END_REQUEST_DRAINED', request_id=req.rid,
                            generation=self.generation, changed_layers=0, copied_experts=0))
            self.request = self.generation = None
            self.state = 'IDLE'
        except BaseException:
            self.state = self.controller.state = 'BROKEN_RELOAD_REQUIRED'
            raise
