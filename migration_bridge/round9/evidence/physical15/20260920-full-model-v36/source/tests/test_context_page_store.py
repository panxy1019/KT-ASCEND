"""CPU-only request-local QSA page ownership and exact BF16 artifacts."""
import dataclasses
import struct
import unittest


class ContextPageStoreTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import context_page_store
        return context_page_store

    def key(self, request='request-a', epoch=3, page=7):
        return self.api().PageKey(request, epoch, page)

    @staticmethod
    def bf16_bytes(word, elements=4 * 2 * 256):
        return struct.pack('<H', word) * elements

    def layers(self, *, tokens=4):
        api = self.api(); elements = tokens * 2 * 256
        return {
            layer: api.make_bf16_layer_payload(
                layer, tokens,
                self.bf16_bytes(0x3F80 + index, elements),
                self.bf16_bytes(0x4000 + index, elements),
            )
            for index, layer in enumerate(api.QSA_LAYER_IDS)
        }

    def store(self, *, pages=4, host_bytes=1 << 20):
        return self.api().RequestLocalPageStore(
            max_pages=pages, max_host_bytes=host_bytes)

    def test_publish_requires_all_layers_and_preserves_exact_bf16_bytes(self):
        api = self.api(); store = self.store(); key = self.key(); layers = self.layers()
        record = store.publish_export(key, layers)
        self.assertEqual(record.scope, 'CPU_REQUEST_LOCAL_OWNERSHIP_ONLY')
        self.assertEqual(record.state, 'HOST_PUBLISHED_NO_HBM_ACTION')
        self.assertEqual(record.key, key)
        self.assertEqual(record.page_generation, 0)
        self.assertEqual(tuple(layer.layer_id for layer in record.layers), api.QSA_LAYER_IDS)
        self.assertEqual(len(record.digest_sha256), 64)
        self.assertGreater(record.host_bytes, 12 * 2 * 4 * 2 * 256)

        lease = store.pin(key)
        pinned = store.read(lease)
        for layer in pinned.layers:
            source = layers[layer.layer_id]
            self.assertEqual(layer.k_bytes, source.k_bytes)
            self.assertEqual(layer.v_bytes, source.v_bytes)
        self.assertEqual(store.snapshot()['active_leases'], 1)
        store.release(lease)
        snapshot = store.snapshot()
        self.assertEqual(snapshot['active_leases'], 0)
        self.assertEqual(snapshot['allocator_free_calls'], 0)
        self.assertEqual(snapshot['hbm_bytes_freed_claimed'], 0)

    def test_incomplete_corrupt_tail_or_wrong_geometry_never_publishes(self):
        api = self.api(); key = self.key()
        cases = []
        incomplete = self.layers(); incomplete.pop(api.QSA_LAYER_IDS[-1]); cases.append(incomplete)
        corrupt = self.layers(); first = api.QSA_LAYER_IDS[0]
        corrupt[first] = dataclasses.replace(corrupt[first], k_bytes=b'\x00' + corrupt[first].k_bytes[1:])
        cases.append(corrupt)
        wrong_layer = self.layers(); payload = wrong_layer.pop(first)
        wrong_layer[first] = dataclasses.replace(payload, layer_id=0); cases.append(wrong_layer)
        tail = self.layers(tokens=3); cases.append(tail)
        for index, layers in enumerate(cases):
            store = self.store()
            with self.subTest(index=index), self.assertRaises(ValueError):
                store.publish_export(key, layers)
            self.assertEqual(store.snapshot()['pages'], 0)
            self.assertEqual(store.snapshot()['host_bytes'], 0)

        tail_payload = api.make_bf16_layer_payload(
            3, 1, self.bf16_bytes(0x3F80, 2 * 256),
            self.bf16_bytes(0x4000, 2 * 256))
        self.assertEqual(tail_payload.shape, (1, 2, 256))

    def test_page_and_host_caps_are_atomic(self):
        api = self.api(); generous = self.store()
        first = generous.publish_export(self.key(page=0), self.layers())
        exact = api.RequestLocalPageStore(
            max_pages=1, max_host_bytes=first.host_bytes)
        exact.publish_export(self.key(page=0), self.layers())
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            exact.publish_export(self.key(page=1), self.layers())
        self.assertEqual(exact.snapshot()['pages'], 1)
        self.assertEqual(exact.snapshot()['host_bytes'], first.host_bytes)

        too_small = api.RequestLocalPageStore(
            max_pages=2, max_host_bytes=first.host_bytes - 1)
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            too_small.publish_export(self.key(page=0), self.layers())
        self.assertEqual(too_small.snapshot()['pages'], 0)

    def test_retired_requests_do_not_accumulate_generation_metadata(self):
        store = self.store(pages=1)
        layers = self.layers()
        previous = -1
        for index in range(100):
            key = self.key(request=f'request-{index}', page=0)
            record = store.publish_export(key, layers)
            self.assertGreater(record.page_generation, previous)
            previous = record.page_generation
            self.assertTrue(store.invalidate(key.request_id, key.allocation_epoch))
            self.assertEqual(store.snapshot()['retained_request_identities'], 0)

    def test_exact_lease_identity_and_delayed_invalidation(self):
        store = self.store(); key = self.key(); store.publish_export(key, self.layers())
        lease = store.pin(key)
        copied = dataclasses.replace(lease)
        with self.assertRaises(ValueError): store.read(copied)
        with self.assertRaises(ValueError): store.release(copied)

        self.assertFalse(store.invalidate('request-a', 3))
        self.assertEqual(store.snapshot()['invalidating'], 1)
        with self.assertRaises(RuntimeError): store.pin(key)
        self.assertEqual(store.read(lease).key, key)
        store.release(lease)
        self.assertEqual(store.snapshot()['pages'], 0)
        with self.assertRaises(ValueError): store.release(lease)

    def test_reload_callback_is_generation_bound_and_stale_completion_drains(self):
        store = self.store(); key = self.key(); first = store.publish_export(key, self.layers())
        ticket = store.begin_reload(key)
        self.assertFalse(store.invalidate('request-a', 3))
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            store.complete_reload(ticket)
        self.assertEqual(store.snapshot()['pages'], 0)

        second = store.publish_export(key, self.layers())
        self.assertGreater(second.page_generation, first.page_generation)
        with self.assertRaises(ValueError): store.complete_reload(ticket)
        fresh = store.begin_reload(key)
        staging = store.complete_reload(fresh)
        self.assertEqual(staging.page_generation, second.page_generation)
        self.assertEqual(store.read(staging).digest_sha256, second.digest_sha256)
        store.release(staging)

    def test_abort_and_request_epoch_invalidation_never_touch_other_owner(self):
        store = self.store(pages=4, host_bytes=1 << 21)
        old = self.key('same-request', 1, 0)
        current = self.key('same-request', 2, 0)
        other = self.key('other-request', 1, 0)
        for key in (old, current, other): store.publish_export(key, self.layers())
        ticket = store.begin_reload(old)
        self.assertFalse(store.invalidate('same-request', 1))
        store.abort_reload(ticket)
        self.assertEqual(store.snapshot()['pages'], 2)
        for key in (current, other):
            lease = store.pin(key)
            self.assertEqual(store.read(lease).key, key)
            store.release(lease)
        self.assertTrue(store.invalidate('same-request', 2))
        self.assertEqual(store.snapshot()['pages'], 1)
        lease = store.pin(other); store.release(lease)

    def test_invalid_keys_payloads_leases_and_reload_tickets_fail_closed(self):
        api = self.api()
        for args in (('', 0, 0), ('r', -1, 0), ('r', 0, -1), ('r', True, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError): api.PageKey(*args)
        for kwargs in ({'max_pages': 0, 'max_host_bytes': 1},
                       {'max_pages': 1, 'max_host_bytes': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.RequestLocalPageStore(**kwargs)
        store = self.store(); key = self.key(); store.publish_export(key, self.layers())
        with self.assertRaises(ValueError): store.pin(self.key(page=8))
        ticket = store.begin_reload(key)
        copied = dataclasses.replace(ticket)
        with self.assertRaises(ValueError): store.complete_reload(copied)
        with self.assertRaises(ValueError): store.abort_reload(copied)
        store.abort_reload(ticket)

        record = store.pin(key)
        published = store.read(record)
        with self.assertRaises(ValueError):
            api._validate_record(dataclasses.replace(
                published, format_version=True))
        malformed_layer = dataclasses.replace(
            published.layers[0], shape=(api.PAGE_SIZE, True, api.HEAD_DIM))
        malformed_layers = (malformed_layer,) + published.layers[1:]
        with self.assertRaises(ValueError):
            api._validate_record(dataclasses.replace(
                published, layers=malformed_layers))
        store.release(record)


if __name__ == '__main__':
    unittest.main()
