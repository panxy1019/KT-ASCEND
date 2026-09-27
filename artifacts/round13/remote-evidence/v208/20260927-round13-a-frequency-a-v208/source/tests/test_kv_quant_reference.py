"""CPU-only canonical FP8 page-record tests; no native backend qualification."""
import dataclasses
import importlib
import json
import unittest
from unittest import mock


class KVQuantImportTests(unittest.TestCase):
    def test_module_declares_reference_only_without_importing_torch(self):
        import sys
        before = sys.modules.get('torch')
        api = importlib.import_module(
            'migration_bridge.round9.runtime.kv_quant_reference')
        self.assertEqual(api.SCOPE, 'CPU_REFERENCE_ONLY')
        self.assertEqual(api.NATIVE_BACKEND_STATUS, 'NOT_IMPLEMENTED_UNQUALIFIED')
        self.assertIs(sys.modules.get('torch'), before)

    def test_record_serialization_bounds_do_not_require_torch(self):
        api = importlib.import_module(
            'migration_bridge.round9.runtime.kv_quant_reference')
        from migration_bridge.round9.runtime.context_profile import ContextStateIdentity
        shape = (1, 2, 256); payload = b'\x00' * (1 * 2 * 256)

        identity = ContextStateIdentity('request-a', None, 0, 'QSA_KV')
        metadata = api._metadata_bytes(identity, 0, 3, shape, 1.0, 1.0)
        record = api.KVPageRecord(
            identity, 0, 3, 1, shape, 1.0, 1.0, payload, payload,
            metadata, api._digest(metadata, payload, payload))
        with self.assertRaisesRegex(ValueError, 'format version'):
            dataclasses.replace(record, format_version=True).to_bytes()

        oversized_identity = ContextStateIdentity(
            'r' * 5000, None, 0, 'QSA_KV')
        oversized_metadata = api._metadata_bytes(
            oversized_identity, 0, 3, shape, 1.0, 1.0)
        oversized = dataclasses.replace(
            record, identity=oversized_identity,
            metadata_bytes=oversized_metadata,
            digest_sha256=api._digest(
                oversized_metadata, payload, payload))
        with self.assertRaisesRegex(ValueError, 'metadata'):
            oversized.to_bytes()


class KVQuantReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module('torch')
        except ModuleNotFoundError as error:
            raise unittest.SkipTest('PyTorch is required for CPU FP8 reference tests') from error
        cls.api = importlib.import_module(
            'migration_bridge.round9.runtime.kv_quant_reference')
        from migration_bridge.round9.runtime.context_profile import ContextStateIdentity
        cls.Identity = ContextStateIdentity

    def identity(self, *, request='request-a', epoch=7, prefix=None):
        return self.Identity(request, prefix, epoch, 'QSA_KV')

    def tensors(self, tokens=4):
        torch = self.torch
        values = torch.linspace(-3, 3, tokens * 2 * 256,
                                dtype=torch.float32).reshape(tokens, 2, 256)
        return values.to(torch.bfloat16), values.flip(0).to(torch.bfloat16)

    def record(self, *, tokens=4, logical_page=9, layer_id=3):
        key, value = self.tensors(tokens)
        return self.api.encode_kv_page(
            self.identity(), logical_page, layer_id, key, value)

    def test_full_page_is_deterministic_and_binary_round_trip_is_canonical(self):
        first = self.record(); second = self.record()
        self.assertEqual(first, second)
        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(len(first.k_bytes), 4 * 2 * 256)
        self.assertEqual(len(first.v_bytes), 4 * 2 * 256)
        self.assertEqual(len(first.digest_sha256), 64)
        metadata = json.loads(first.metadata_bytes)
        self.assertEqual(metadata['scope'], 'CPU_REFERENCE_ONLY')
        self.assertEqual(metadata['shape'], [4, 2, 256])
        self.assertEqual(metadata['request_id'], 'request-a')
        self.assertEqual(metadata['allocation_epoch'], 7)
        self.assertEqual(metadata['logical_page'], 9)
        self.assertEqual(metadata['layer_id'], 3)

        parsed = self.api.parse_kv_page_record(first.to_bytes())
        self.assertEqual(parsed, first)
        decoded_key, decoded_value = self.api.decode_kv_page(
            parsed, self.identity(), logical_page=9, layer_id=3)
        source_key, source_value = self.tensors()
        self.assertEqual(decoded_key.dtype, self.torch.bfloat16)
        self.assertEqual(decoded_key.device.type, 'cpu')
        self.assertEqual(tuple(decoded_key.shape), (4, 2, 256))
        self.assertTrue(self.torch.isfinite(decoded_key).all().item())
        self.assertTrue(self.torch.isfinite(decoded_value).all().item())
        self.assertLessEqual(
            (decoded_key.float() - source_key.float()).abs().max().item(),
            first.k_scale_f32 * 32,
        )
        self.assertLessEqual(
            (decoded_value.float() - source_value.float()).abs().max().item(),
            first.v_scale_f32 * 32,
        )

    def test_zero_and_tail_pages_use_finite_positive_scales(self):
        torch = self.torch
        zeros = torch.zeros((2, 2, 256), dtype=torch.bfloat16)
        record = self.api.encode_kv_page(
            self.identity(), 0, 47, zeros, zeros.clone())
        self.assertEqual(record.shape, (2, 2, 256))
        self.assertEqual(record.valid_tokens, 2)
        self.assertEqual(record.k_scale_f32, 1.0)
        self.assertEqual(record.v_scale_f32, 1.0)
        self.assertEqual(set(record.k_bytes), {0})
        key, value = self.api.decode_kv_page(
            record, self.identity(), logical_page=0, layer_id=47)
        self.assertTrue(self.torch.equal(key, zeros))
        self.assertTrue(self.torch.equal(value, zeros))

    def test_maximum_finite_and_independent_k_v_scales_remain_finite(self):
        torch = self.torch
        maximum = torch.full(
            (4, 2, 256), torch.finfo(torch.bfloat16).max,
            dtype=torch.bfloat16,
        )
        smaller = torch.ones((4, 2, 256), dtype=torch.bfloat16)
        record = self.api.encode_kv_page(
            self.identity(), 1, 7, maximum, smaller)
        self.assertGreater(record.k_scale_f32, record.v_scale_f32)
        key, value = self.api.decode_kv_page(
            record, self.identity(), logical_page=1, layer_id=7)
        self.assertTrue(torch.isfinite(key).all().item())
        self.assertTrue(torch.isfinite(value).all().item())
        self.assertEqual(key.abs().max().item(), torch.finfo(torch.bfloat16).max)

    def test_dtype_shape_nonfinite_and_unbounded_coordinates_are_rejected(self):
        torch = self.torch; key, value = self.tensors()
        bad_pairs = (
            (key.float(), value),
            (key[:, :1], value[:, :1]),
            (key[:, :, :128], value[:, :, :128]),
            (torch.zeros((5, 2, 256), dtype=torch.bfloat16),
             torch.zeros((5, 2, 256), dtype=torch.bfloat16)),
            (key, value[:3]),
        )
        for bad_key, bad_value in bad_pairs:
            with self.subTest(shape=tuple(bad_key.shape), dtype=str(bad_key.dtype)), \
                    self.assertRaises((TypeError, ValueError)):
                self.api.encode_kv_page(
                    self.identity(), 0, 3, bad_key, bad_value)
        for special in (float('nan'), float('inf'), float('-inf')):
            bad = key.clone(); bad[0, 0, 0] = special
            with self.subTest(special=special), self.assertRaises(ValueError):
                self.api.encode_kv_page(self.identity(), 0, 3, bad, value)
        for page, layer in ((-1, 3), (32769, 3), (0, 0), (0, 48), (True, 3)):
            with self.subTest(page=page, layer=layer), self.assertRaises(ValueError):
                self.api.encode_kv_page(
                    self.identity(), page, layer, key, value)

    def test_prefix_sharing_wrong_component_and_stale_identity_are_rejected(self):
        key, value = self.tensors()
        for identity in (
            self.identity(prefix='shared-prefix'),
            self.Identity('request-a', None, 7, 'QSA_INDEX'),
        ):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.api.encode_kv_page(identity, 0, 3, key, value)

        record = self.record()
        for expected, page, layer in (
            (self.identity(request='request-b'), 9, 3),
            (self.identity(epoch=8), 9, 3),
            (self.identity(), 8, 3),
            (self.identity(), 9, 7),
            (self.identity(prefix='shared-prefix'), 9, 3),
        ):
            with self.subTest(expected=expected, page=page, layer=layer), \
                    self.assertRaises(ValueError):
                self.api.decode_kv_page(
                    record, expected, logical_page=page, layer_id=layer)

    def test_payload_metadata_digest_scale_and_truncation_tampering_fail_closed(self):
        record = self.record(); blob = record.to_bytes()
        mutations = (
            dataclasses.replace(record, k_bytes=bytes([record.k_bytes[0] ^ 1]) + record.k_bytes[1:]),
            dataclasses.replace(record, metadata_bytes=record.metadata_bytes + b' '),
            dataclasses.replace(record, digest_sha256='0' * 64),
            dataclasses.replace(record, k_scale_f32=0.0),
            dataclasses.replace(record, v_scale_f32=float('inf')),
        )
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.api.decode_kv_page(
                    changed, self.identity(), logical_page=9, layer_id=3)
        for changed_blob in (
            blob[:-1], blob + b'\x00',
            bytes([blob[0] ^ 1]) + blob[1:],
            blob[:-33] + bytes([blob[-33] ^ 1]) + blob[-32:],
        ):
            with self.subTest(length=len(changed_blob)), self.assertRaises(ValueError):
                self.api.parse_kv_page_record(changed_blob)

    def test_forged_nan_payload_is_rejected_even_with_matching_digest(self):
        record = self.record()
        nan_byte = int(self.torch.tensor(
            [float('nan')], dtype=self.torch.float8_e4m3fn,
        ).view(self.torch.uint8).item())
        forged_k = bytes([nan_byte]) + record.k_bytes[1:]
        forged = dataclasses.replace(
            record,
            k_bytes=forged_k,
            digest_sha256=self.api._digest(
                record.metadata_bytes, forged_k, record.v_bytes),
        )
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.api.decode_kv_page(
                forged, self.identity(), logical_page=9, layer_id=3)

    def test_decode_uses_explicit_cpu_owned_storage(self):
        record = self.record(tokens=2, logical_page=0, layer_id=47)
        with mock.patch.object(
                self.torch, 'frombuffer',
                side_effect=AssertionError('decode must not alias a byte buffer')):
            key, value = self.api.decode_kv_page(
                record, self.identity(), logical_page=0, layer_id=47)
        self.assertEqual(key.device.type, 'cpu')
        self.assertEqual(value.device.type, 'cpu')

    def test_metadata_bound_and_exact_integer_version_apply_before_serialization(self):
        key, value = self.tensors(tokens=1)
        oversized_identity = self.identity(request='r' * 5000)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.api.encode_kv_page(
                oversized_identity, 0, 3, key[:1], value[:1])

        record = self.record(tokens=1, logical_page=0, layer_id=3)
        with self.assertRaisesRegex(ValueError, 'format version'):
            dataclasses.replace(record, format_version=True).to_bytes()

        metadata = self.api._metadata_bytes(
            oversized_identity, record.logical_page, record.layer_id,
            record.shape, record.k_scale_f32, record.v_scale_f32)
        oversized = dataclasses.replace(
            record,
            identity=oversized_identity,
            metadata_bytes=metadata,
            digest_sha256=self.api._digest(
                metadata, record.k_bytes, record.v_bytes),
        )
        with self.assertRaisesRegex(ValueError, 'metadata'):
            oversized.to_bytes()


if __name__ == '__main__':
    unittest.main()
