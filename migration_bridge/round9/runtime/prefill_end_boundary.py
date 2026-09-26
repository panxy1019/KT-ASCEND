"""Opt-in live-state-preserving barrier using the existing migration engine."""
import copy
import time

from .dynamic_boundary import RequestBoundary


class PrefillEndBoundary(RequestBoundary):
    _ROUND11_PHASES = (
        'route_statistics', 'consumer_drain_sync', 'state_fence_before',
        'selection', 'migration_copy_and_sync', 'mapping_publication',
        'state_fence_after', 'cleanup',
    )

    def __init__(self, controller, live_state, event=None, *, round11_timing=False, clock=None):
        super().__init__(controller, event)
        self.live_state = live_state
        self.state = 'IDLE'
        self._round11_timing_enabled = round11_timing is True
        self._round11_clock = clock or time.monotonic_ns
        self._round11_timing = None
        self._round11_total_started_ns = None

    def _round11_start(self):
        return self._round11_clock() if self._round11_timing_enabled else None

    def _round11_finish(self, name, started_ns):
        if not self._round11_timing_enabled:
            return
        elapsed = int(self._round11_clock()) - int(started_ns)
        if elapsed < 0:
            raise RuntimeError('non-monotonic Round11 Dynamic timing clock')
        self._round11_timing[name] += elapsed

    def before(self, requests):
        if self.state == 'BROKEN_RELOAD_REQUIRED': raise RuntimeError(self.state)
        fresh = self.request is None
        super().before(requests)
        if fresh: self.state = 'PREFILL_ACTIVE'

    def hybrid_statistics(self, *, topk=10):
        """Publish the already leased Hybrid route counts at final prefill.

        Dynamic's KTEP interception records original logical IDs for both
        Hybrid and Layerwise.  Layerwise owns a separate weighted collector;
        Hybrid reaches this method only after its final chunk sampled the first
        output token.  No tensor read or extra device synchronization is added.
        """
        req = self.request
        if req is None or self.state != 'PREFILL_ACTIVE':
            raise RuntimeError('hybrid prefill statistics require one live lease')
        if req.finished(): return None
        if len(req.output_ids) == 0: return None
        if len(req.output_ids) != 1:
            raise RuntimeError('hybrid prefill barrier must follow exactly one sampled output token')
        timing_started = self._round11_start()
        tokens = len(req.origin_input_ids)
        if type(topk) is not int or topk < 1 or tokens < 1:
            raise ValueError('invalid Hybrid route geometry')
        counts = copy.deepcopy(self.controller._counts)
        layers = sorted(self.controller._residents)
        if set(counts) != set(layers) or layers != list(range(len(layers))):
            raise ValueError('incomplete Hybrid layer coverage')
        experts = self.controller.experts
        ordered = []
        for layer in layers:
            row = counts[layer]
            if (not isinstance(row, list) or len(row) != experts or
                    any(type(value) is not int or value < 0 for value in row) or
                    sum(row) != tokens * topk):
                raise ValueError('Hybrid route conservation failed')
            ordered.append(row)
        result = dict(schema=1, layers=len(layers), experts=experts, topk=topk,
                      completed_phases=1, tokens=tokens, pending=False,
                      counts=ordered, weight_mass=[[0.0] * experts for _ in layers],
                      request_id=req.rid, generation=self.generation)
        if self._round11_timing_enabled:
            if self._round11_timing is not None:
                raise RuntimeError('Round11 Dynamic timing already active')
            self._round11_timing = {name: 0 for name in self._ROUND11_PHASES}
            self._round11_total_started_ns = timing_started
            self._round11_finish('route_statistics', timing_started)
        return result

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
            if self._round11_timing_enabled and self._round11_timing is None:
                self._round11_timing = {name: 0 for name in self._ROUND11_PHASES}
                self._round11_total_started_ns = self._round11_clock()
            self.state = 'PREFILL_DRAINING'
            phase_started = self._round11_start()
            self.controller.adapter.synchronize()
            self._round11_finish('consumer_drain_sync', phase_started)
            phase_started = self._round11_start()
            before = self.live_state(req)
            old_residents = dict(self.controller._residents)
            self._round11_finish('state_fence_before', phase_started)
            self.state = 'MIGRATION_ACTIVE'
            generation = self.controller.commit_prefill(req.rid, old_generation, statistics,
                                                        placement_consumers_drained=True,
                                                        timing=self._round11_timing,
                                                        clock=self._round11_clock)
            phase_started = self._round11_start()
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
            self._round11_finish('state_fence_after', phase_started)
            if self._round11_timing_enabled:
                total = int(self._round11_clock()) - int(self._round11_total_started_ns)
                if total < sum(self._round11_timing.values()):
                    raise RuntimeError('Round11 Dynamic phase timings overlap')
                self._round11_timing['total_exposed'] = total
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
            if self._round11_timing_enabled:
                timing = dict(self._round11_timing or {})
                if set(timing) != set(self._ROUND11_PHASES) | {'total_exposed'}:
                    raise RuntimeError('Round11 Dynamic timing receipt is incomplete')
                self.event(dict(status='ROUND11_DYNAMIC_TIMING', request_id=req.rid,
                                generation=self.generation, timing_ns=timing))
            self.event(dict(status='PREFILL_END_REQUEST_DRAINED', request_id=req.rid,
                            generation=self.generation, changed_layers=0, copied_experts=0))
            self.request = self.generation = None
            self.state = 'IDLE'
            self._round11_timing = self._round11_total_started_ns = None
        except BaseException:
            self.state = self.controller.state = 'BROKEN_RELOAD_REQUIRED'
            raise

    def cancel(self, request):
        if self.state == 'BROKEN_RELOAD_REQUIRED': raise RuntimeError(self.state)
        try:
            super().cancel(request)
            self.state = 'IDLE'
            self._round11_timing = self._round11_total_started_ns = None
        except BaseException:
            self.state = self.controller.state = 'BROKEN_RELOAD_REQUIRED'
            raise
