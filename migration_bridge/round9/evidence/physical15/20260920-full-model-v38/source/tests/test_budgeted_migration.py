import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from migration_bridge.round9.runtime.host_budget import HostWorkBudget,Priority
from migration_bridge.round9.runtime.budgeted_migration import BudgetedMigration


class Adapter:
    expert_bytes=10
    def __init__(self):self.events=[];self.fail_copy=False;self.fail_sync=False
    def copy_experts(self,layer,moves):
        self.events.append(('copy',layer,tuple(moves)))
        if self.fail_copy:raise OSError('copy failure')
    def synchronize(self):
        self.events.append(('sync',))
        if self.fail_sync:raise OSError('sync failure')
    def publish(self,*args):self.events.append(('publish',args))


class MigrationBudgetTests(unittest.TestCase):
    def test_joint_zero_allocation_completion_clears_finished_request(self):
        from migration_bridge.round9.runtime.dynamic_boundary import scheduler_after
        _,_,budget=self.setup();req=NS(rid='sync-fallback-no-moves')
        boundary=NS(request=req)
        def finish():boundary.request=None
        boundary.after=finish
        runner=NS(_round9_joint_host_budget=budget,_round9_ple_credit_requests=set(),_round9_ple_request=req)
        scheduler=NS(_round9_dynamic_boundary=boundary,tp_worker=NS(model_runner=runner))
        with patch('migration_bridge.round9.runtime.dynamic_boundary._enabled',return_value=True):scheduler_after(scheduler)
        self.assertIsNone(runner._round9_ple_request)

    def test_joint_request_retirement_follows_dynamic_request_end(self):
        from migration_bridge.round9.runtime.dynamic_boundary import scheduler_after
        wrapped,adapter,budget=self.setup()
        wrapped.copy_experts(0,[(0,8)])
        req=NS(rid='request')
        events=[]
        boundary=NS(request=req,event=events.append)
        def finish():boundary.request=None
        boundary.after=finish
        runner=NS(_round9_joint_host_budget=budget,_round9_ple_credit_requests=wrapped.requests,
                  _round9_ple_request=req)
        scheduler=NS(_round9_dynamic_boundary=boundary,tp_worker=NS(model_runner=runner))
        with patch('migration_bridge.round9.runtime.dynamic_boundary._enabled',return_value=True):
            scheduler_after(scheduler)
        self.assertEqual(budget.snapshot()['active_requests'],0)
        self.assertEqual(wrapped.requests,set())
        self.assertIsNone(runner._round9_ple_request)
        self.assertEqual(events, [{
            'status': 'JOINT_BUDGET_REQUEST_DRAINED',
            'request_id': 'request',
            'budget': {
                'enabled': True,
                'capacity': {'host_bytes': 48, 'h2d_bytes': 64, 'pinned_bytes': 8},
                'reserve': {'host_bytes': 0, 'h2d_bytes': 0, 'pinned_bytes': 0},
                'used': {'host_bytes': 0, 'h2d_bytes': 0, 'pinned_bytes': 0},
                'allocations': 0, 'active_requests': 0,
                'in_flight': 0, 'cancelled': 0,
            },
        }])

    def setup(self):
        budget=HostWorkBudget(enabled=True,host_bytes=48,h2d_bytes=64,pinned_bytes=8)
        adapter=Adapter()
        wrapped=BudgetedMigration(adapter,budget,lambda:('request',2))
        return wrapped,adapter,budget

    def test_shared_ple_allocation_and_copy_hold_then_release_exact_credits(self):
        wrapped,adapter,budget=self.setup()
        ple=budget.try_acquire(request_id='request',placement_epoch=2,priority=Priority.PLE_CURRENT,
            host_bytes=8,h2d_bytes=8,pinned_bytes=8)
        original=adapter.copy_experts
        def copy(layer,moves):
            self.assertEqual(budget.snapshot()['used'],dict(host_bytes=48,h2d_bytes=28,pinned_bytes=8))
            original(layer,moves)
        adapter.copy_experts=copy
        wrapped.copy_experts(0,[(0,8),(1,9)])
        self.assertEqual(adapter.events,[('copy',0,((0,8),(1,9))),('sync',)])
        self.assertEqual(budget.snapshot()['used'],dict(host_bytes=8,h2d_bytes=8,pinned_bytes=8))
        budget.release(ple);budget.retire_request('request')

    def test_budget_denial_has_no_copy_and_cannot_deadlock_waiter(self):
        wrapped,adapter,budget=self.setup()
        hold=budget.try_acquire(request_id='ple',placement_epoch=0,priority=Priority.PLE_CURRENT,
            host_bytes=16,h2d_bytes=0,pinned_bytes=0)
        with self.assertRaisesRegex(RuntimeError,'budget'):wrapped.copy_experts(0,[(0,8)])
        self.assertEqual(adapter.events,[])
        budget.release(hold)
        wrapped.copy_experts(0,[(0,8)])

    def test_copy_error_drains_before_releasing_and_sync_error_quarantines(self):
        for fail_sync in (False,True):
            wrapped,adapter,budget=self.setup();adapter.fail_copy=True;adapter.fail_sync=fail_sync
            with self.assertRaisesRegex(OSError,'copy failure'):wrapped.copy_experts(0,[(0,8)])
            self.assertEqual(adapter.events[-1],('sync',))
            self.assertEqual(budget.snapshot()['in_flight'],int(fail_sync))
            self.assertEqual(wrapped.state,'BROKEN_RELOAD_REQUIRED')
            with self.assertRaisesRegex(RuntimeError,'BROKEN'):wrapped.copy_experts(0,[(0,8)])

    def test_invalid_identity_or_moves_rejected_before_admission(self):
        wrapped,adapter,budget=self.setup()
        for moves in ([(0,1),(0,2)],[(True,1)],[(0,-1)]):
            with self.assertRaises(ValueError):wrapped.copy_experts(0,moves)
        self.assertEqual(budget.snapshot()['allocations'],0)


if __name__=='__main__':unittest.main()
