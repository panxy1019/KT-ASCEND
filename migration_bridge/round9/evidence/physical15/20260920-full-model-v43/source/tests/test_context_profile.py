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

    def qwen4_text_config(self):
        layer_types = []
        for layer in range(48):
            layer_types.append(
                'full_attention' if (layer + 1) % 4 == 0 else 'linear_attention')
        return {
            'model_type': 'qwen4_exp_text',
            'dtype': 'bfloat16',
            'num_hidden_layers': 48,
            'layer_types': layer_types,
            'full_attention_interval': 4,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'indexer_kv_heads': 1,
            'indexer_head_dim': 128,
            'indexer_compress_ratio': 4,
            'indexer_budget': 2048,
            'linear_num_value_heads': 48,
            'linear_value_head_dim': 128,
            'linear_num_key_heads': 16,
            'linear_key_head_dim': 128,
            'linear_conv_kernel_dim': 4,
            'mamba_ssm_dtype': 'float32',
            'hidden_size': 2560,
            'hc_count': 4,
            'ple_embed_dim': 2560,
            'ple_layer_ids': [2],
            'ple_conv_kernel_size': 4,
            'ngram_size': 3,
            'heads_per_ngram': 8,
        }

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

    def test_qwen4_tp1_geometry_is_derived_from_explicit_text_or_full_config(self):
        api = self.api()
        text = self.qwen4_text_config()
        direct = api.derive_qwen4_tp1_geometry(text, config_scope='text_config')
        nested = api.derive_qwen4_tp1_geometry(
            {'text_config': text}, config_scope='full_config')
        self.assertEqual(direct, nested)
        self.assertEqual(direct.gdn_layers, 36)
        self.assertEqual(direct.qsa_layers, 12)
        self.assertEqual(direct.gdn_conv_shape_per_layer, (10240, 3))
        self.assertEqual(direct.gdn_temporal_shape_per_layer, (48, 128, 128))
        self.assertEqual(direct.gdn_bytes_per_live_request, 115458048)
        self.assertEqual(direct.ple_conv_shape_per_live_request, (1, 10240, 9))
        self.assertEqual(direct.ple_history_shape_per_live_request, (2,))
        self.assertEqual(direct.ple_bytes_per_live_request, 184336)
        self.assertEqual(direct.raw_request_slots, 1)
        self.assertEqual(direct.recurrent_request_slots, 1)
        self.assertEqual(direct.recurrent_storage_slots, 2)
        self.assertEqual(direct.context_geometry.gdn_bytes_per_request, 115458048)
        self.assertEqual(direct.context_geometry.ple_bytes_per_request, 184336)

    def test_qwen4_tp1_geometry_rejects_ambiguous_scope_and_checkpoint_drift(self):
        api = self.api()
        text = self.qwen4_text_config()
        for document, scope in ((text, 'full_config'),
                                ({'text_config': text}, 'text_config'),
                                (text, 'auto')):
            with self.subTest(scope=scope), self.assertRaises((TypeError, ValueError)):
                api.derive_qwen4_tp1_geometry(document, config_scope=scope)
        mutations = (
            ('dtype', 'float16'),
            ('num_hidden_layers', 47),
            ('num_key_value_heads', 1),
            ('indexer_kv_heads', True),
            ('indexer_head_dim', 64),
            ('linear_num_value_heads', 32),
            ('mamba_ssm_dtype', 'bfloat16'),
            ('hc_count', 2),
            ('ple_layer_ids', [1]),
        )
        for name, value in mutations:
            changed = dict(text)
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                api.derive_qwen4_tp1_geometry(changed, config_scope='text_config')
        changed = dict(text)
        changed['layer_types'] = list(text['layer_types'])
        changed['layer_types'][0] = 'full_attention'
        with self.assertRaises(ValueError):
            api.derive_qwen4_tp1_geometry(changed, config_scope='text_config')

    def test_qwen4_tp1_footprints_distinguish_logical_and_padded_storage(self):
        api = self.api()
        geometry = api.derive_qwen4_tp1_geometry(
            self.qwen4_text_config(), config_scope='text_config')
        expected_live = {
            4096: 219581504,
            16384: 531057728,
            65536: 1776962624,
            131072: 3438169152,
        }
        expected_floor = {
            4096: 335308864,
            16384: 646735936,
            65536: 1892444224,
            131072: 3553388608,
        }
        for target in expected_live:
            with self.subTest(target=target):
                footprint = api.derive_qwen4_tp1_footprint(target, geometry)
                self.assertEqual(footprint.plan.total_bytes, expected_live[target])
                self.assertEqual(footprint.logical_token_capacity, target + 4)
                self.assertEqual(footprint.logical_pages, (target + 4) // 4)
                self.assertEqual(footprint.physical_padding_pages, 1)
                self.assertEqual(footprint.physical_pages,
                                 footprint.logical_pages + 1)
                self.assertEqual(footprint.physical_token_slots, target + 8)
                self.assertEqual(footprint.raw_request_slots, 1)
                self.assertEqual(footprint.recurrent_request_slots, 1)
                self.assertEqual(footprint.recurrent_storage_slots, 2)
                self.assertEqual(footprint.physical_floor_bytes,
                                 expected_floor[target])
        current = api.derive_qwen4_tp1_footprint(4096, geometry)
        tensors = {item.name: item for item in current.expected_tensors}
        self.assertEqual(tensors['ordinary_kv'].shape, (2, 12, 1026, 4, 2, 256))
        self.assertEqual(tensors['qsa_compressed'].shape, (12, 1026, 1, 128))
        self.assertEqual(tensors['qsa_raw'].shape, (12, 4, 1, 128))
        self.assertEqual(tensors['qsa_positions'].shape, (4,))
        self.assertEqual(tensors['gdn_conv'].shape, (36, 2, 10240, 3))
        self.assertEqual(tensors['gdn_temporal'].shape, (36, 2, 48, 128, 128))
        self.assertEqual(tensors['ple_conv'].shape, (1, 2, 10240, 9))
        self.assertEqual(tensors['ple_history'].shape, (2, 2))

    def test_qwen4_measurement_interface_accepts_exact_runtime_shapes_only(self):
        api = self.api()
        geometry = api.derive_qwen4_tp1_geometry(
            {'text_config': self.qwen4_text_config()}, config_scope='full_config')
        footprint = api.derive_qwen4_tp1_footprint(4096, geometry)
        measured = {
            expected.name: api.TensorMeasurement(
                expected.shape, expected.dtype, expected.nbytes)
            for expected in footprint.expected_tensors
        }
        summary = api.validate_qwen4_tp1_measurements(footprint, measured)
        self.assertEqual(summary.total_bytes, 335308864)
        self.assertEqual(summary.tensor_count, 8)
        for name, bad in (
            ('ordinary_kv', api.TensorMeasurement(
                (2, 12, 1025, 4, 2, 256), 'torch.bfloat16', 100761600)),
            ('gdn_temporal', api.TensorMeasurement(
                (36, 2, 48, 128, 128), 'torch.bfloat16', 113246208)),
            ('ple_history', api.TensorMeasurement((2, 2), 'torch.int64', 16)),
        ):
            changed = dict(measured)
            changed[name] = bad
            with self.subTest(name=name), self.assertRaises(ValueError):
                api.validate_qwen4_tp1_measurements(footprint, changed)
        missing = dict(measured)
        missing.pop('qsa_positions')
        with self.assertRaises(ValueError):
            api.validate_qwen4_tp1_measurements(footprint, missing)


if __name__ == '__main__': unittest.main()
