"""CPU reference tests for request-local selected-row QSA staging."""
import importlib
import gc
import unittest
import weakref


class ContextStagingImportTests(unittest.TestCase):
    def test_import_is_lazy_and_scope_is_explicit(self):
        import sys
        before = sys.modules.get('torch')
        api = importlib.import_module(
            'migration_bridge.round9.runtime.context_staging')
        self.assertEqual(api.SCOPE, 'CPU_SELECTED_ROW_STAGING_REFERENCE_ONLY')
        self.assertEqual(api.RUNTIME_INTEGRATION_STATUS, 'NOT_INSTALLED')
        self.assertEqual(api.HBM_RELEASE_STATUS, 'NO_HBM_RELEASE')
        self.assertEqual(api.MAX_FINAL_STAGING_BYTES, 4_200_448)
        self.assertEqual(api.BF16_LAYER_PAGE_BYTES, 8_192)
        self.assertIs(sys.modules.get('torch'), before)


class ContextStagingTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module('torch')
        except ModuleNotFoundError as error:
            raise unittest.SkipTest('PyTorch is required for staging tests') from error
        cls.api = importlib.import_module(
            'migration_bridge.round9.runtime.context_staging')
        cls.store_api = importlib.import_module(
            'migration_bridge.round9.runtime.context_page_store')
        cls.fp8_api = importlib.import_module(
            'migration_bridge.round9.runtime.kv_quant_reference')
        cls.qsa = importlib.import_module(
            'migration_bridge.round9.runtime.qsa')
        from migration_bridge.round9.runtime.context_profile import ContextStateIdentity
        cls.Identity = ContextStateIdentity

    def identity(self, request='request-a', epoch=7):
        return self.Identity(request, None, epoch, 'QSA_KV')

    def page_tensors(self, page):
        torch = self.torch
        logical = torch.arange(page * 4, page * 4 + 4, dtype=torch.float32)
        base = logical[:, None, None].expand(4, 2, 256)
        key = (base + torch.arange(256, dtype=torch.float32)[None, None, :] / 512)
        value = base * 3 + torch.arange(2, dtype=torch.float32)[None, :, None]
        return key.to(torch.bfloat16), value.expand(4, 2, 256).to(torch.bfloat16)

    def bf16_bytes(self, tensor):
        return tensor.contiguous().view(self.torch.uint8).numpy().tobytes()

    def host_record(self, page):
        api = self.store_api
        key = api.PageKey('request-a', 7, page)
        k, v = self.page_tensors(page)
        layers = {
            layer: api.make_bf16_layer_payload(
                layer, 4, self.bf16_bytes(k), self.bf16_bytes(v))
            for layer in api.QSA_LAYER_IDS
        }
        store = api.RequestLocalPageStore(
            max_pages=1, max_host_bytes=1 << 21)
        return store.publish_export(key, layers)

    def test_bf16_pages_preserve_order_duplicates_padding_and_sparse_result(self):
        torch = self.torch
        pages = {page: self.host_record(page) for page in (0, 1)}
        calls = []

        def records(key, layer_id):
            calls.append(key)
            self.assertEqual(layer_id, 3)
            return pages.get(key.logical_page)

        selected = torch.tensor([[5, 1, 5, 6, -1, -1]], dtype=torch.int64)
        staged = self.api.stage_selected_rows(
            self.identity(), 3, selected, current_logical_position=6,
            record_reader=records)
        self.assertEqual(staged.logical_tokens, (5, 1, 6))
        self.assertTrue(torch.equal(
            staged.remapped,
            torch.tensor([[0, 1, 0, 2, -1, -1]], dtype=torch.int64)))
        self.assertEqual([key.logical_page for key in calls], [1, 0])
        self.assertEqual(staged.keys.dtype, torch.bfloat16)
        self.assertEqual(staged.keys.device.type, 'cpu')
        self.assertEqual(tuple(staged.keys.shape), (3, 2, 256))
        self.assertEqual(staged.staging_bytes, 3 * 2 * 2 * 256 * 2)

        full_k = torch.cat([self.page_tensors(page)[0] for page in (0, 1)])
        full_v = torch.cat([self.page_tensors(page)[1] for page in (0, 1)])
        query = torch.linspace(-1, 1, 24 * 256, dtype=torch.float32).reshape(
            1, 24, 256).to(torch.bfloat16)
        baseline = self.qsa.sparse_gqa(query, full_k, full_v, selected)
        compact = self.qsa.sparse_gqa(
            query, staged.keys, staged.values, staged.remapped)
        self.assertTrue(torch.equal(compact, baseline))

    def test_all_padding_publishes_empty_compact_tensors(self):
        torch = self.torch; calls = []
        staged = self.api.stage_selected_rows(
            self.identity(), 3,
            torch.full((1, 7), -1, dtype=torch.int32),
            current_logical_position=0,
            record_reader=lambda key, layer: calls.append((key, layer)))
        self.assertEqual(staged.logical_tokens, ())
        self.assertEqual(tuple(staged.keys.shape), (0, 2, 256))
        self.assertEqual(staged.remapped.dtype, torch.int32)
        self.assertEqual(calls, [])

    def test_pages_are_consumed_once_and_released_before_the_next_read(self):
        torch = self.torch; refs = []; calls = []

        def resident(page_key, layer_id):
            gc.collect()
            if refs:
                self.assertIsNone(refs[-2]())
                self.assertIsNone(refs[-1]())
            keys, values = self.page_tensors(page_key.logical_page)
            refs.extend((weakref.ref(keys), weakref.ref(values)))
            calls.append(page_key.logical_page)
            return self.api.ResidentBF16Page(
                page_key, layer_id, keys, values)

        staged = self.api.stage_selected_rows(
            self.identity(), 3,
            torch.tensor([[8, 0, 5, 8, -1]], dtype=torch.int64),
            current_logical_position=8, resident_reader=resident)
        self.assertEqual(calls, [2, 0, 1])
        self.assertEqual(staged.logical_tokens, (8, 0, 5))
        self.assertTrue(torch.equal(
            staged.remapped,
            torch.tensor([[0, 1, 2, 0, -1]], dtype=torch.int64)))
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))

    def test_declared_final_bound_matches_2051_tp1_rows(self):
        api = self.api
        expected = 2051 * 2 * 2 * 256 * 2
        self.assertEqual(api.MAX_FINAL_STAGING_BYTES, expected)
        self.assertEqual(api.BF16_LAYER_PAGE_BYTES, 4 * 2 * 2 * 256 * 2)

    def test_fp8_record_decodes_through_exact_identity(self):
        torch = self.torch; identity = self.identity(); key, value = self.page_tensors(0)
        record = self.fp8_api.encode_kv_page(identity, 0, 7, key, value)
        selected = torch.tensor([[3, 0, 3, -1]], dtype=torch.int64)
        staged = self.api.stage_selected_rows(
            identity, 7, selected, current_logical_position=3,
            record_reader=lambda page_key, layer_id: record)
        decoded_k, decoded_v = self.fp8_api.decode_kv_page(
            record, identity, logical_page=0, layer_id=7)
        self.assertTrue(torch.equal(staged.keys, decoded_k[[3, 0]]))
        self.assertTrue(torch.equal(staged.values, decoded_v[[3, 0]]))
        self.assertTrue(torch.equal(
            staged.remapped, torch.tensor([[0, 1, 0, -1]])))

    def test_resident_reader_is_synchronous_identity_stamped_and_strict(self):
        torch = self.torch; api = self.api; store_api = self.store_api
        key, value = self.page_tensors(0); calls = []

        def resident(page_key, layer_id):
            calls.append((page_key, layer_id))
            return api.ResidentBF16Page(page_key, layer_id, key, value)

        staged = api.stage_selected_rows(
            self.identity(), 11,
            torch.tensor([[2, 1, -1]], dtype=torch.int64),
            current_logical_position=2,
            record_reader=lambda page_key, layer_id: None,
            resident_reader=resident)
        self.assertEqual(len(calls), 1)
        self.assertEqual(staged.logical_tokens, (2, 1))
        staged_snapshot = staged.keys.clone()
        key.zero_()
        self.assertTrue(torch.equal(staged.keys, staged_snapshot))

        bad_results = (
            object(),
            api.ResidentBF16Page(
                store_api.PageKey('request-b', 7, 0), 11, key, value),
            api.ResidentBF16Page(
                store_api.PageKey('request-a', 7, 0), 7, key, value),
            api.ResidentBF16Page(
                store_api.PageKey('request-a', 7, 0), 11,
                key.float(), value),
            api.ResidentBF16Page(
                store_api.PageKey('request-a', 7, 0), 11,
                key[:, :1], value[:, :1]),
        )
        for result in bad_results:
            with self.subTest(result=type(result).__name__), self.assertRaises(
                    (TypeError, ValueError)):
                api.stage_selected_rows(
                    self.identity(), 11,
                    torch.tensor([[0]], dtype=torch.int64),
                    current_logical_position=0,
                    resident_reader=lambda page_key, layer_id, result=result: result)

    def test_selection_and_identity_fail_closed_before_result_publication(self):
        torch = self.torch; api = self.api
        invalid = (
            torch.tensor([[0] * 2052], dtype=torch.int64),
            torch.tensor([[2]], dtype=torch.int64),
            torch.tensor([[0, -1, 1]], dtype=torch.int64),
            torch.tensor([[-2]], dtype=torch.int64),
            torch.tensor([[0]], dtype=torch.int16),
            torch.tensor([[0], [0]], dtype=torch.int64),
        )
        for selected in invalid:
            with self.subTest(shape=tuple(selected.shape), dtype=str(selected.dtype)), \
                    self.assertRaises(ValueError):
                api.stage_selected_rows(
                    self.identity(), 3, selected,
                    current_logical_position=1,
                    record_reader=lambda key, layer: self.host_record(0))

        wrong_records = (
            self.host_record(1),
            self.fp8_api.encode_kv_page(
                self.identity(request='request-b'), 0, 3,
                *self.page_tensors(0)),
            self.fp8_api.encode_kv_page(
                self.identity(epoch=8), 0, 3, *self.page_tensors(0)),
            self.fp8_api.encode_kv_page(
                self.identity(), 0, 7, *self.page_tensors(0)),
        )
        for record in wrong_records:
            with self.subTest(record=type(record).__name__), self.assertRaises(ValueError):
                api.stage_selected_rows(
                    self.identity(), 3,
                    torch.tensor([[0]], dtype=torch.int64),
                    current_logical_position=0,
                    record_reader=lambda key, layer, record=record: record)

    def test_missing_offset_callback_failure_and_nonfinite_resident_fail_closed(self):
        torch = self.torch; api = self.api
        short_k, short_v = (tensor[:2] for tensor in self.page_tensors(0))
        short = api.ResidentBF16Page(
            self.store_api.PageKey('request-a', 7, 0), 3,
            short_k, short_v)
        with self.assertRaisesRegex(ValueError, 'offset'):
            api.stage_selected_rows(
                self.identity(), 3, torch.tensor([[3]], dtype=torch.int64),
                current_logical_position=3,
                resident_reader=lambda key, layer: short)

        bad_k = self.page_tensors(0)[0].clone(); bad_k[0, 0, 0] = float('nan')
        bad = api.ResidentBF16Page(
            self.store_api.PageKey('request-a', 7, 0), 3,
            bad_k, self.page_tensors(0)[1])
        with self.assertRaisesRegex(ValueError, 'finite'):
            api.stage_selected_rows(
                self.identity(), 3, torch.tensor([[0]], dtype=torch.int64),
                current_logical_position=0,
                resident_reader=lambda key, layer: bad)

        nan_k, nan_v = self.page_tensors(0)
        nan_k[0, 0, 0] = float('nan')
        layers = {
            layer: self.store_api.make_bf16_layer_payload(
                layer, 4, self.bf16_bytes(nan_k), self.bf16_bytes(nan_v))
            for layer in self.store_api.QSA_LAYER_IDS
        }
        store = self.store_api.RequestLocalPageStore(
            max_pages=1, max_host_bytes=1 << 21)
        host_nan = store.publish_export(
            self.store_api.PageKey('request-a', 7, 0), layers)
        with self.assertRaisesRegex(ValueError, 'finite'):
            api.stage_selected_rows(
                self.identity(), 3, torch.tensor([[0]], dtype=torch.int64),
                current_logical_position=0,
                record_reader=lambda key, layer: host_nan)

        def failed_reader(key, layer_id):
            raise RuntimeError('synchronous read failed')

        with self.assertRaisesRegex(RuntimeError, 'synchronous read failed'):
            api.stage_selected_rows(
                self.identity(), 3, torch.tensor([[0]], dtype=torch.int64),
                current_logical_position=0, record_reader=failed_reader)


if __name__ == '__main__':
    unittest.main()
