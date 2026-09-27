"""CPU-only contract tests for bounded shared host-work credits."""
import dataclasses
import importlib
import unittest


class HostWorkBudgetTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('migration_bridge.round9.runtime.host_budget')

    def enabled_budget(self, **overrides):
        api = self.api()
        values = dict(
            enabled=True,
            host_bytes=100,
            h2d_bytes=80,
            pinned_bytes=60,
            reserve_host_bytes=20,
            reserve_h2d_bytes=10,
            reserve_pinned_bytes=10,
        )
        values.update(overrides)
        return api.HostWorkBudget(**values)

    def test_default_is_off_and_admission_is_nonblocking(self):
        api = self.api()
        budget = api.HostWorkBudget()

        self.assertIsNone(budget.try_acquire(
            request_id='request-a', placement_epoch=0,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        ))
        self.assertEqual(
            budget.snapshot(),
            {
                'enabled': False,
                'capacity': {'host_bytes': 0, 'h2d_bytes': 0, 'pinned_bytes': 0},
                'reserve': {'host_bytes': 0, 'h2d_bytes': 0, 'pinned_bytes': 0},
                'used': {'host_bytes': 0, 'h2d_bytes': 0, 'pinned_bytes': 0},
                'allocations': 0,
                'active_requests': 0,
                'in_flight': 0,
                'cancelled': 0,
            },
        )

    def test_joint_limits_are_atomic_and_never_exceeded(self):
        api = self.api()
        budget = self.enabled_budget(
            host_bytes=10, h2d_bytes=8, pinned_bytes=6,
            reserve_host_bytes=0, reserve_h2d_bytes=0,
            reserve_pinned_bytes=0,
        )
        first = budget.try_acquire(
            request_id='r', placement_epoch=0,
            priority=api.Priority.EXPERT_MIGRATION,
            host_bytes=7, h2d_bytes=5, pinned_bytes=4,
        )
        self.assertIsNotNone(first)

        # Each denial is caused by a different exhausted dimension. A denied
        # request must not partially consume any of the other dimensions.
        for requested in (
            dict(host_bytes=4, h2d_bytes=1, pinned_bytes=1),
            dict(host_bytes=1, h2d_bytes=4, pinned_bytes=1),
            dict(host_bytes=1, h2d_bytes=1, pinned_bytes=3),
        ):
            with self.subTest(requested=requested):
                self.assertIsNone(budget.try_acquire(
                    request_id='other', placement_epoch=0,
                    priority=api.Priority.PLE_CURRENT, **requested,
                ))
                self.assertEqual(
                    budget.snapshot()['used'],
                    {'host_bytes': 7, 'h2d_bytes': 5, 'pinned_bytes': 4},
                )

        second = budget.try_acquire(
            request_id='other', placement_epoch=0,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=3, h2d_bytes=3, pinned_bytes=2,
        )
        self.assertIsNotNone(second)
        self.assertEqual(
            budget.snapshot()['used'],
            {'host_bytes': 10, 'h2d_bytes': 8, 'pinned_bytes': 6},
        )

    def test_prefetch_cannot_consume_correctness_reserve(self):
        api = self.api()
        budget = self.enabled_budget()

        prefetch = budget.try_acquire(
            request_id='prefetch', placement_epoch=2,
            priority=api.Priority.PLE_PREFETCH,
            host_bytes=80, h2d_bytes=70, pinned_bytes=50,
        )
        self.assertIsNotNone(prefetch)
        self.assertIsNone(budget.try_acquire(
            request_id='prefetch', placement_epoch=2,
            priority=api.Priority.PLE_PREFETCH,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        ))

        # Correctness traffic may use the reserve that prefetch cannot touch.
        current = budget.try_acquire(
            request_id='current', placement_epoch=0,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=20, h2d_bytes=10, pinned_bytes=10,
        )
        self.assertIsNotNone(current)
        self.assertEqual(
            budget.snapshot()['used'],
            {'host_bytes': 100, 'h2d_bytes': 80, 'pinned_bytes': 60},
        )

    def test_cancelled_in_flight_prefetch_is_not_reclaimed_early(self):
        api = self.api()
        budget = self.enabled_budget(
            reserve_host_bytes=0, reserve_h2d_bytes=0,
            reserve_pinned_bytes=0,
        )
        prefetch = budget.try_acquire(
            request_id='prefetch', placement_epoch=0,
            priority=api.Priority.PLE_PREFETCH,
            host_bytes=100, h2d_bytes=80, pinned_bytes=60,
        )
        budget.mark_in_flight(prefetch)

        cancelled = budget.cancel_prefetches()
        self.assertEqual(cancelled, (prefetch,))
        self.assertEqual(budget.cancel_prefetches(), ())
        self.assertEqual(
            budget.snapshot()['used'],
            {'host_bytes': 100, 'h2d_bytes': 80, 'pinned_bytes': 60},
        )
        self.assertIsNone(budget.try_acquire(
            request_id='current', placement_epoch=0,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        ))
        with self.assertRaises(RuntimeError):
            budget.release(prefetch)

        budget.complete(prefetch)
        self.assertEqual(
            budget.snapshot()['used'],
            {'host_bytes': 100, 'h2d_bytes': 80, 'pinned_bytes': 60},
        )
        budget.release(prefetch)
        current = budget.try_acquire(
            request_id='current', placement_epoch=0,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        )
        self.assertIsNotNone(current)

    def test_cancel_is_idempotent_for_same_live_allocation(self):
        api = self.api(); budget = self.enabled_budget()
        allocation = budget.try_acquire(
            request_id='prefetch', placement_epoch=0,
            priority=api.Priority.PLE_PREFETCH,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        )
        budget.cancel(allocation)
        budget.cancel(allocation)
        self.assertEqual(budget.snapshot()['cancelled'], 1)

        copied = dataclasses.replace(allocation)
        with self.assertRaises(ValueError):
            budget.cancel(copied)

    def test_request_placement_allocation_identity_rejects_stale_use(self):
        api = self.api()
        budget = self.enabled_budget()
        self.assertGreater(api.Priority.PLE_CURRENT, api.Priority.EXPERT_MIGRATION)
        self.assertGreater(api.Priority.EXPERT_MIGRATION, api.Priority.PLE_PREFETCH)
        old = budget.try_acquire(
            request_id='request', placement_epoch=3,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        )
        newer = budget.try_acquire(
            request_id='request', placement_epoch=4,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        )
        self.assertGreater(newer.allocation_epoch, old.allocation_epoch)
        self.assertIsInstance(old.identity, api.BufferIdentity)
        with self.assertRaises(ValueError):
            budget.try_acquire(
                request_id='request', placement_epoch=3,
                priority=api.Priority.PLE_CURRENT,
                host_bytes=1, h2d_bytes=1, pinned_bytes=1,
            )

        forged = dataclasses.replace(
            old,
            identity=api.BufferIdentity('request', 4, old.allocation_epoch),
        )
        with self.assertRaises(ValueError):
            budget.cancel(forged)
        copied = dataclasses.replace(old)
        with self.assertRaises(ValueError):
            budget.cancel(copied)
        budget.release(old)
        with self.assertRaises(ValueError):
            budget.release(old)
        with self.assertRaises(ValueError):
            budget.mark_in_flight(old)

    def test_request_retirement_requires_all_allocations_released(self):
        api = self.api()
        budget = self.enabled_budget()
        allocation = budget.try_acquire(
            request_id='request', placement_epoch=5,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        )
        self.assertEqual(budget.snapshot()['active_requests'], 1)
        with self.assertRaises(RuntimeError):
            budget.retire_request('request')

        budget.mark_in_flight(allocation)
        with self.assertRaises(RuntimeError):
            budget.retire_request('request')
        budget.complete(allocation)
        budget.release(allocation)

        budget.retire_request('request')
        self.assertEqual(budget.snapshot()['active_requests'], 0)
        with self.assertRaises(ValueError):
            budget.retire_request('request')

        # Retirement explicitly permits later reuse of the external ID. The
        # caller must not reuse it before this lifecycle boundary.
        reused = budget.try_acquire(
            request_id='request', placement_epoch=0,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        )
        self.assertIsNotNone(reused)

    def test_invalid_capacities_credits_and_identities_are_rejected(self):
        api = self.api()
        for kwargs in (
            dict(enabled=True, host_bytes=-1),
            dict(enabled=True, host_bytes=1, reserve_host_bytes=2),
            dict(enabled=True, host_bytes=True),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.HostWorkBudget(**kwargs)

        budget = self.enabled_budget()
        base = dict(
            request_id='r', placement_epoch=0,
            priority=api.Priority.PLE_CURRENT,
            host_bytes=1, h2d_bytes=1, pinned_bytes=1,
        )
        for changed in (
            {'request_id': ''}, {'placement_epoch': -1}, {'placement_epoch': True},
            {'priority': 1}, {'host_bytes': -1}, {'h2d_bytes': True},
        ):
            args = dict(base)
            args.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                budget.try_acquire(**args)


if __name__ == '__main__':
    unittest.main()
