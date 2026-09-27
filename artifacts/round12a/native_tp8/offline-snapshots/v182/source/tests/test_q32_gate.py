"""Strict CPU-only validation for new Q32 device and physical receipts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


class Q32GateTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import q32_gate
        return q32_gate

    def physical(self):
        return {
            'resident_capacity': 32,
            'tensors': {
                'w13_weight': {
                    'shape': [32, 2560, 1280],
                    'dtype': 'torch.bfloat16',
                    'bytes': 209715200,
                },
                'w2_weight': {
                    'shape': [32, 640, 2560],
                    'dtype': 'torch.bfloat16',
                    'bytes': 104857600,
                },
            },
            'total_bytes': 314572800,
        }

    def gate(self):
        same_hash = 'a' * 64
        migrations = []
        for generation, logical in enumerate((511, 0, 511), start=1):
            migrations.append({
                'generation': generation,
                'moves': [[0, logical]],
                'all32_processed_experts_exact': True,
                'pinned_mask_pointer_stable': True,
                'formats_after': {'w13_weight': 29, 'w2_weight': 29},
            })
        return {
            'schema': 1,
            'resident_capacity': 32,
            'status': 'PASS_Q32_REAL_KTEP_LOCAL_MERGE',
            'storage': 'BF16',
            'cpu_workers': 128,
            'source_identity': {'revision': 'frozen-test'},
            'bank_sha256': 'bank-test',
            'native_sha256': 'c' * 64,
            'npu_weight_formats': {'w13_weight': 29, 'w2_weight': 29},
            'physical_capacity': self.physical(),
            'batch_copy_probe': {
                'experts': 32,
                'reverse_and_restore_exact': True,
                'max_device_staging_bytes': 200 * 1024**2,
                'staging_scope': 'EXPLICIT_PROJECTION_TENSOR_ONLY',
            },
            'measured_migration': {
                'peak_extra_allocated_bytes': 700 * 1024**2,
                'allocator_limit_bytes': 2 * 1024**3,
                'samples': 2,
            },
            'dynamic_migrations': migrations,
            'rows': [
                {
                    'tokens': tokens,
                    'kind': kind,
                    'errors': {
                        'cpu_partial': 0.0,
                        'npu_owned_partial': 0.001,
                        'routed_merge': 0.002,
                    },
                    'hashes': [same_hash] * 3,
                }
                for tokens in (1, 10, 32)
                for kind in ('mixed', 'resident_only', 'repeated_cpu')
            ],
        }

    def write_and_check(self, directory, gate=None, **expected):
        api = self.api()
        path = Path(directory) / 'q32-gate.json'
        path.write_text(json.dumps(gate or self.gate()))
        arguments = dict(
            storage='BF16', workers=128,
            source_identity={'revision': 'frozen-test'},
            layer0_sha256='bank-test',
            native_sha256='c' * 64,
        )
        arguments.update(expected)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return api.require_q32_device_gate(path, digest, **arguments)

    def test_valid_gate_binds_q32_geometry_identity_and_numeric_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.write_and_check(directory)
        self.assertEqual(result['resident_capacity'], 32)
        self.assertEqual(result['physical_capacity']['total_bytes'], 314572800)

    def test_wrong_capacity_geometry_dtype_or_byte_totals_are_rejected(self):
        mutations = (
            ('resident_capacity', 64),
            ('physical_capacity.resident_capacity', 64),
            ('physical_capacity.tensors.w13_weight.shape', [64, 2560, 1280]),
            ('physical_capacity.tensors.w2_weight.dtype', 'torch.float16'),
            ('physical_capacity.tensors.w13_weight.bytes', 1),
            ('physical_capacity.total_bytes', 1),
            ('batch_copy_probe.max_device_staging_bytes', 400 * 1024**2),
            ('batch_copy_probe.staging_scope', 'TOTAL_OPERATOR_WORKSPACE'),
            ('measured_migration.peak_extra_allocated_bytes', 1024**3 + 1),
            ('measured_migration.allocator_limit_bytes', 3 * 1024**3 + 1),
            ('measured_migration.samples', 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            for dotted, value in mutations:
                gate = self.gate()
                target = gate
                parts = dotted.split('.')
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                with self.subTest(field=dotted), self.assertRaises(RuntimeError):
                    self.write_and_check(directory, gate)

    def test_wrong_sha_source_profile_or_q64_evidence_is_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q32-gate.json'
            gate = self.gate()
            path.write_text(json.dumps(gate))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            arguments = dict(
                storage='BF16', workers=128,
                source_identity={'revision': 'frozen-test'},
                layer0_sha256='bank-test',
                native_sha256='c' * 64,
            )
            with self.assertRaises(RuntimeError):
                api.require_q32_device_gate(path, '0' * 64, **arguments)
            for changed in (
                {'storage': 'Q8_0'}, {'workers': 64},
                {'source_identity': {'revision': 'other'}},
                {'layer0_sha256': 'other-bank'},
                {'native_sha256': 'd' * 64},
            ):
                with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                    api.require_q32_device_gate(path, digest, **(arguments | changed))

            gate['status'] = 'PASS_REAL_KTEP_LOCAL_MERGE'
            with self.assertRaises(RuntimeError):
                self.write_and_check(directory, gate)
            gate = self.gate()
            gate['dynamic_migrations'][0]['all32_processed_experts_exact'] = False
            gate['dynamic_migrations'][0]['all64_processed_experts_exact'] = True
            with self.assertRaises(RuntimeError):
                self.write_and_check(directory, gate)

    def test_migration_sequence_nz29_and_nine_repeatable_cases_are_required(self):
        gate = self.gate()
        mutations = []
        changed = copy.deepcopy(gate)
        changed['dynamic_migrations'][1]['moves'] = [[1, 0]]
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed['dynamic_migrations'][2]['formats_after']['w2_weight'] = 2
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed['npu_weight_formats']['w13_weight'] = 2
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed['rows'].pop()
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed['rows'][0]['errors']['routed_merge'] = 0.011
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed['rows'][0]['hashes'][2] = 'b' * 64
        mutations.append(changed)

        with tempfile.TemporaryDirectory() as directory:
            for number, changed in enumerate(mutations):
                with self.subTest(mutation=number), self.assertRaises(RuntimeError):
                    self.write_and_check(directory, changed)

    def test_all48_physical_records_are_exact_and_sum_to_14400_mib(self):
        api = self.api()
        records = [dict(layer=layer, **copy.deepcopy(self.physical())) for layer in range(48)]
        summary = api.validate_q32_physical_layers(records)
        self.assertEqual(summary, {
            'schema': 1,
            'layers': 48,
            'resident_capacity': 32,
            'per_layer_bytes': 314572800,
            'total_bytes': 15099494400,
        })

        for mutation in ('missing', 'duplicate', 'shape', 'extra'):
            changed = copy.deepcopy(records)
            if mutation == 'missing':
                changed.pop()
            elif mutation == 'duplicate':
                changed[-1]['layer'] = 0
            elif mutation == 'shape':
                changed[7]['tensors']['w13_weight']['shape'][0] = 64
            else:
                changed[7]['unexpected'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                api.validate_q32_physical_layers(changed)


if __name__ == '__main__':
    unittest.main()
