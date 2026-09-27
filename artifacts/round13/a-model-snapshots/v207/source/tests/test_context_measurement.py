"""CPU tests for exact live-pool measurement paths and profile comparison."""
from types import SimpleNamespace as NS
import unittest

from migration_bridge.round9.runtime import context_measurement as measurement
from migration_bridge.round9.runtime import context_profile as profile


class FakeTensor:
    _BYTES = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.int64': 8}
    _NEXT_PTR = 4096

    def __init__(self, shape, dtype, *, nbytes=None, device='cpu', storage=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._numel = 1
        for value in self.shape:
            self._numel *= value
        self._element_size = self._BYTES[dtype]
        self.nbytes = (self._numel * self._element_size
                       if nbytes is None else nbytes)
        if storage is None:
            storage = FakeStorage(self._NEXT_PTR, self.nbytes)
            FakeTensor._NEXT_PTR += max(self.nbytes, 64) + 64
        self._storage = storage

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size

    def data_ptr(self):
        return self._storage.data_ptr()

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return True

    def untyped_storage(self):
        return self._storage


class FakeStorage:
    def __init__(self, pointer, nbytes):
        self.pointer = pointer
        self.bytes = nbytes

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.bytes


def footprint():
    context = profile.ContextGeometry(
        model_layers=48, qsa_layers=12, qsa_kv_heads=2, qsa_head_dim=256,
        qsa_kv_element_bytes=2, qsa_index_heads=1, qsa_index_head_dim=128,
        qsa_index_element_bytes=2, qsa_compress_ratio=4, page_size=4,
        request_slots=1, reserved_token_slots=4, page_metadata_bytes=16,
        position_element_bytes=8, gdn_bytes_per_request=115458048,
        ple_bytes_per_request=184336)
    geometry = profile.Qwen4TP1Geometry(
        context, 12, 36, (10240, 3), (48, 128, 128), 115458048,
        (1, 10240, 9), (2,), 184336, 1, 1, 2, 2048)
    return profile.derive_qwen4_tp1_footprint(4096, geometry)


def runner_for(current, *, device='cpu'):
    tensors = {
        expected.name: FakeTensor(expected.shape, expected.dtype, device=device)
        for expected in current.expected_tensors
    }
    mamba_cache = NS(conv=[tensors['gdn_conv']],
                      temporal=tensors['gdn_temporal'])
    mamba_pool = NS(size=1, mamba_cache=mamba_cache)
    req_pool = NS(
        mamba_pool=mamba_pool,
        short_conv_pool=NS(size=1, conv_state=tensors['ple_conv']),
        ngram_pool=NS(size=1, context=tensors['ple_history']))
    kv_pool = NS(
        full_kv_pool=NS(kv_buffer=tensors['ordinary_kv']),
        qsa_compressed_keys=tensors['qsa_compressed'],
        qsa_raw_keys=tensors['qsa_raw'],
        qsa_rope_positions=tensors['qsa_positions'],
        full_layer_nums=12, request_capacity=1, mamba_pool=mamba_pool)
    return NS(req_to_token_pool=req_pool, token_to_kv_pool=kv_pool), tensors


class ContextMeasurementTests(unittest.TestCase):
    def test_exact_eight_live_pool_tensors_match_tp1_profile(self):
        expected = footprint()
        runner, _ = runner_for(expected)
        result = measurement.measure_and_validate_qwen4_tp1_pools(
            runner, expected)
        self.assertEqual(len(result.tensors), 8)
        self.assertEqual(result.total_bytes, 335308864)
        self.assertEqual(result.expected_floor_bytes, 335308864)
        self.assertEqual(result.devices, ('cpu',))
        self.assertIsNone(result.required_device)
        self.assertIn('DEVICE_ONLY_RECORDED', result.scope)
        self.assertEqual(set(result.by_name()), set(measurement.TENSOR_ATTRIBUTE_CHAINS))
        for item in result.tensors:
            reference = {value.name: value for value in expected.expected_tensors}[item.name]
            self.assertEqual((item.shape, item.dtype, item.nbytes),
                             (reference.shape, reference.dtype, reference.nbytes))
            self.assertEqual(item.device, 'cpu')
            self.assertEqual(item.data_ptr, item.storage_data_ptr)
            self.assertEqual(item.storage_nbytes, item.nbytes)
            self.assertEqual(item.attribute_chain,
                             measurement.TENSOR_ATTRIBUTE_CHAINS[item.name])

    def test_real_pool_topology_and_attribute_chains_fail_closed(self):
        expected = footprint()
        runner, _ = runner_for(expected)
        runner.token_to_kv_pool.mamba_pool = NS(size=1)
        with self.assertRaisesRegex(ValueError, 'same MambaPool'):
            measurement.measure_and_validate_qwen4_tp1_pools(runner, expected)

        runner, _ = runner_for(expected)
        runner.req_to_token_pool.mamba_pool.mamba_cache.conv.append(
            FakeTensor((1,), 'torch.bfloat16'))
        with self.assertRaisesRegex(ValueError, 'exactly one GDN conv'):
            measurement.measure_and_validate_qwen4_tp1_pools(runner, expected)

        runner, _ = runner_for(expected)
        del runner.token_to_kv_pool.qsa_compressed_keys
        with self.assertRaisesRegex(ValueError, 'qsa_compressed_keys'):
            measurement.measure_and_validate_qwen4_tp1_pools(runner, expected)

    def test_shape_dtype_and_runtime_nbytes_each_fail_closed(self):
        expected = footprint()
        for name, replacement in (
            ('ordinary_kv', FakeTensor((2, 12, 1025, 4, 2, 256),
                                       'torch.bfloat16')),
            ('qsa_raw', FakeTensor((12, 4, 1, 128), 'torch.float32')),
            ('ple_history', FakeTensor((2, 2), 'torch.int64', nbytes=31)),
        ):
            runner, tensors = runner_for(expected)
            old = tensors[name]
            if name == 'ordinary_kv':
                runner.token_to_kv_pool.full_kv_pool.kv_buffer = replacement
            elif name == 'qsa_raw':
                runner.token_to_kv_pool.qsa_raw_keys = replacement
            else:
                runner.req_to_token_pool.ngram_pool.context = replacement
            self.assertIsNot(old, replacement)
            with self.subTest(name=name), self.assertRaises(ValueError):
                measurement.measure_and_validate_qwen4_tp1_pools(runner, expected)

    def test_category_alias_capacity_and_footprint_type_fail_closed(self):
        expected = footprint()
        runner, tensors = runner_for(expected)
        shared = tensors['qsa_positions'].untyped_storage()
        runner.req_to_token_pool.ngram_pool.context = FakeTensor(
            (2, 2), 'torch.int64', storage=shared)
        with self.assertRaisesRegex(ValueError, 'share underlying storage'):
            measurement.measure_and_validate_qwen4_tp1_pools(runner, expected)

        runner, _ = runner_for(expected)
        runner.req_to_token_pool.ngram_pool.size = True
        with self.assertRaisesRegex(ValueError, 'positive exact int'):
            measurement.measure_and_validate_qwen4_tp1_pools(runner, expected)

        runner, _ = runner_for(expected)
        with self.assertRaises(TypeError):
            measurement.measure_and_validate_qwen4_tp1_pools(runner, object())

    def test_required_device_is_explicit_and_cpu_cannot_claim_npu_scope(self):
        expected = footprint()
        runner, _ = runner_for(expected)
        with self.assertRaisesRegex(ValueError, 'required npu'):
            measurement.measure_and_validate_qwen4_tp1_pools(
                runner, expected, require_device='npu')
        with self.assertRaises(ValueError):
            measurement.measure_and_validate_qwen4_tp1_pools(
                runner, expected, require_device='npu:0')
        with self.assertRaises(ValueError):
            measurement.measure_and_validate_qwen4_tp1_pools(
                runner, expected, require_device='cpu')


if __name__ == '__main__':
    unittest.main()
