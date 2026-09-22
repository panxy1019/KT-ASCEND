"""CPU-only F8 context sizing and state-identity contracts."""
from dataclasses import replace
import unittest


class ContextProfileTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import context_profile
        return context_profile

    def geometry(self):
        api = self.api()
        return api.ContextGeometry(
            model_layers=48,
            qsa_layers=12,
            qsa_kv_heads=2,
            qsa_head_dim=256,
            qsa_kv_element_bytes=2,
            qsa_index_heads=1,
            qsa_index_head_dim=128,
            qsa_index_element_bytes=2,
            qsa_compress_ratio=4,
            page_size=4,
            request_slots=1,
            reserved_token_slots=4,
            page_metadata_bytes=16,
            position_element_bytes=8,
            gdn_bytes_per_request=4096,
            ple_bytes_per_request=64,
        )

    def test_four_context_targets_derive_exact_page_and_component_budgets(self):
        api = self.api()
        geometry = self.geometry()
        for target in (4096, 16384, 65536, 131072):
            with self.subTest(target=target):
                plan = api.derive_context_plan(target, geometry)
                self.assertEqual(plan.context_tokens, target)
                self.assertEqual(plan.token_capacity, target + 4)
                self.assertEqual(plan.pages, (target + 4) // 4)
                self.assertEqual(plan.qsa_kv_bytes,
                    plan.token_capacity * 12 * 2 * 2 * 256 * 2)
                self.assertEqual(plan.qsa_index_bytes,
                    (plan.token_capacity // 4) * 12 * 1 * 128 * 2)
                self.assertEqual(plan.qsa_pending_bytes,
                    4 * (12 * 1 * 128 * 2 + 8))
                self.assertEqual(plan.gdn_bytes, 4096)
                self.assertEqual(plan.ple_bytes, 64)
                self.assertFalse(plan.native_kv_qualified)
                self.assertEqual(plan.scope, 'CPU_CONTEXT_PLAN_ONLY')
                self.assertEqual(plan.total_bytes, sum((
                    plan.qsa_kv_bytes, plan.qsa_index_bytes,
                    plan.qsa_pending_bytes, plan.gdn_bytes, plan.ple_bytes,
                    plan.page_metadata_bytes,
                )))

    def test_budget_limit_and_checked_integer_overflow_are_rejected(self):
        api = self.api()
        geometry = self.geometry()
        plan = api.derive_context_plan(4096, geometry)
        with self.assertRaises(ValueError):
            api.derive_context_plan(4096, geometry,
                                    budget_limit_bytes=plan.total_bytes - 1)
        self.assertEqual(api.derive_context_plan(
            4096, geometry, budget_limit_bytes=plan.total_bytes).total_bytes,
            plan.total_bytes)
        huge = replace(geometry, qsa_head_dim=2**60)
        with self.assertRaises(OverflowError):
            api.derive_context_plan(4096, huge)

    def test_unsupported_targets_and_page_compression_boundaries_are_rejected(self):
        api = self.api()
        for target in (0, 4095, 4097, 32768, 262144, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                api.derive_context_plan(target, self.geometry())
        for changed in (
            {'page_size': 3},
            {'page_size': 8, 'reserved_token_slots': 2},
            {'qsa_compress_ratio': 3},
            {'qsa_layers': 49},
            {'qsa_index_heads': 2},
            {'qsa_kv_element_bytes': 1},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.geometry().__class__(**(
                    self.geometry().__dict__ | changed
                ))

    def test_page_mapping_accepts_last_slot_and_rejects_overflow_or_misalignment(self):
        api = self.api()
        plan = api.derive_context_plan(4096, self.geometry())
        self.assertEqual(api.token_page(plan, 0), (0, 0))
        self.assertEqual(api.token_page(plan, 4095), (1023, 3))
        self.assertEqual(api.token_page(plan, 4099, include_reserved=True), (1024, 3))
        for token in (-1, 4096, 4100, True):
            with self.subTest(token=token), self.assertRaises(ValueError):
                api.token_page(plan, token, include_reserved=(token == 4100))
        self.assertEqual(api.compressed_page(plan, 4092), 1023)
        for start in (1, 4093, 4096, -4, True):
            with self.subTest(start=start), self.assertRaises(ValueError):
                api.compressed_page(plan, start)

    def test_qsa_kv_index_gdn_and_ple_have_separate_bound_identities(self):
        api = self.api()
        identities = api.context_identities(
            request_id='request-7', prefix_id='prefix-a', allocation_epoch=3)
        self.assertEqual(set(identities), {'QSA_KV', 'QSA_INDEX', 'GDN', 'PLE'})
        self.assertEqual(len(set(identities.values())), 4)
        api.validate_context_identities(identities)
        stale = dict(identities)
        stale['PLE'] = replace(stale['PLE'], allocation_epoch=4)
        with self.assertRaises(ValueError):
            api.validate_context_identities(stale)
        aliased = dict(identities)
        aliased['PLE'] = identities['GDN']
        with self.assertRaises(ValueError):
            api.validate_context_identities(aliased)
        plan = api.derive_context_plan(4096, self.geometry())
        first_page = api.qsa_kv_page_owner(plan, identities, 0)
        last_page = api.qsa_kv_page_owner(plan, identities, plan.pages - 1)
        self.assertEqual((first_page.identity.component, first_page.start, first_page.stop),
                         ('QSA_KV', 0, 4))
        self.assertEqual(last_page.stop, plan.token_capacity)
        last_index = api.qsa_index_owner(plan, identities, plan.compressed_slots - 1)
        self.assertEqual(last_index.identity.component, 'QSA_INDEX')
        self.assertEqual(last_index.stop, plan.token_capacity)
        for function, value in ((api.qsa_kv_page_owner, plan.pages),
                                (api.qsa_index_owner, plan.compressed_slots)):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(plan, identities, value)
        for request, prefix, epoch in (('', None, 0), ('r', '', 0), ('r', None, -1),
                                       ('r', None, True)):
            with self.subTest(request=request, prefix=prefix, epoch=epoch), \
                 self.assertRaises(ValueError):
                api.context_identities(request, prefix, epoch)


if __name__ == '__main__': unittest.main()
