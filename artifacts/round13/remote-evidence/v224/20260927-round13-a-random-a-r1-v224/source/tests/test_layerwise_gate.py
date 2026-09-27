"""CPU-only contract tests for the real layerwise BF16 transport receipt."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


_MIB = 1024**2


class LayerwiseGateTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import layerwise_gate
        return layerwise_gate

    def source_sha256(self):
        return {
            'layerwise.py': '1' * 64,
            'dynamic_adapter.py': '2' * 64,
            'expert_source.py': '3' * 64,
        }

    def gate(self):
        sweep = []
        for layer in range(48):
            hashes = {
                'gate_sha256': hashlib.sha256(f'gate-{layer}'.encode()).hexdigest(),
                'up_sha256': hashlib.sha256(f'up-{layer}'.encode()).hexdigest(),
                'down_sha256': hashlib.sha256(f'down-{layer}'.encode()).hexdigest(),
            }
            sweep.append({
                'layer': layer,
                'logical_expert': (layer * 11 + 7) % 512,
                **hashes,
                'combined_sha256': hashlib.sha256(
                    ''.join(hashes.values()).encode('ascii')
                ).hexdigest(),
            })
        cases = []
        for tokens in (1, 10, 32):
            for kind in ('one_group', 'four_block_multigroup'):
                cases.append({
                    'tokens': tokens,
                    'kind': kind,
                    'groups': 1 if kind == 'one_group' or tokens == 1 else 2,
                    'relative_l2': 0.001,
                    'execution_peak_extra_bytes': 64 * _MIB,
                    'output_sha256': hashlib.sha256(
                        f'{tokens}-{kind}'.encode()
                    ).hexdigest(),
                    'relative_l2_tolerance': 0.01,
                    'finite': True,
                    'all_top10_routes_executed': True,
                    'last_group_original_weights_exact': True,
                    'private_adapter_copy_only': True,
                    'template_residents_unchanged': True,
                })
        return {
            'schema': 1,
            'status': 'PASS_LAYERWISE_BF16_TRANSPORT',
            'scope': 'LAYER0_SYNTHETIC_REAL_NPU_NOT_FULL_MODEL_OR_SCHEDULER',
            'capacity': 32,
            'resident_capacity': 32,
            'max_group_rows': 32,
            'geometry': {
                'hidden': 2560,
                'intermediate': 640,
                'experts': 512,
                'layers': 48,
                'topk': 10,
                'dtype': 'torch.bfloat16',
                'w13_shape': [32, 2560, 1280],
                'w2_shape': [32, 640, 2560],
                'private_bank_bytes': 314572800,
            },
            'storage': 'BF16',
            'fixture_layers': [0],
            'allocator_limit_bytes': 3 * 1024**3,
            'constructor_peak_extra_bytes': 300 * _MIB,
            'execution_peak_extra_bytes': 128 * _MIB,
            'source_sha256': self.source_sha256(),
            'native_sha256': '4' * 64,
            'source_identity': {'revision': 'original-model-test'},
            'source_layer_identity_sweep': sweep,
            'numeric_cases': cases,
            'abort_probe': {
                'inflight_abort_synchronized': True,
                'backend_reusable_after_inflight_abort': True,
                'public_abort_reuse': True,
            },
            'unused_slot_probe': {
                'loaded_unused_slots': 22,
                'unchanged_after_execution': True,
                'numerically_ignored': True,
            },
            'original_resident_weights_and_maps_exact': True,
            'private_shadow_has_no_resident_weight_alias': True,
            'copied_original_bf16_bytes': 1024,
        }

    def write_and_check(self, directory, gate=None, **expected):
        path = Path(directory) / 'layerwise-gate.json'
        path.write_text(json.dumps(gate or self.gate()))
        arguments = {
            'source_identity': {'revision': 'original-model-test'},
            'native_sha256': '4' * 64,
            'source_sha256': self.source_sha256(),
        }
        arguments.update(expected)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.api().require_layerwise_device_gate(
            path, digest, **arguments
        )

    def test_valid_receipt_returns_runtime_resource_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.write_and_check(directory)
        self.assertEqual(result['constructor_peak_extra_bytes'], 300 * _MIB)
        self.assertEqual(result['execution_peak_extra_bytes'], 128 * _MIB)
        self.assertEqual(result['max_group_rows'], 32)

    def test_receipt_sha_original_model_native_and_all_sources_are_bound(self):
        gate = self.gate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layerwise-gate.json'
            path.write_text(json.dumps(gate))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            args = {
                'source_identity': {'revision': 'original-model-test'},
                'native_sha256': '4' * 64,
                'source_sha256': self.source_sha256(),
            }
            with self.assertRaises(RuntimeError):
                self.api().require_layerwise_device_gate(path, '0' * 64, **args)
            changes = (
                {'source_identity': {'revision': 'different-model'}},
                {'native_sha256': '5' * 64},
                {'source_sha256': self.source_sha256() | {'layerwise.py': '6' * 64}},
                {'source_sha256': {'runtime/layerwise.py': '1' * 64}},
            )
            for changed in changes:
                with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                    self.api().require_layerwise_device_gate(
                        path, digest, **(args | changed)
                    )

    def test_q32_pending_or_placeholder_receipts_are_rejected(self):
        mutations = (
            ('schema', 0),
            ('status', 'PASS_Q32_REAL_KTEP_LOCAL_MERGE'),
            ('status', 'ADMITTED_LAYERWISE_BF16_TRANSPORT_PENDING'),
            ('resident_capacity', 64),
            ('capacity', 64),
            ('max_group_rows', 31),
            ('fixture_layers', []),
            ('constructor_peak_extra_bytes', 0),
            ('execution_peak_extra_bytes', 0),
            ('allocator_limit_bytes', 43_008 * _MIB + 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            for field, value in mutations:
                gate = self.gate()
                gate[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    self.write_and_check(directory, gate)

    def test_geometry_and_all48_source_layer_identity_sweep_are_exact(self):
        mutations = []
        for field, value in (
            ('hidden', 4096), ('intermediate', 768), ('experts', 256),
            ('layers', 47), ('topk', 8), ('dtype', 'torch.float16'),
            ('w13_shape', [32, 2560, 640]), ('private_bank_bytes', 1),
        ):
            gate = self.gate()
            gate['geometry'][field] = value
            mutations.append(gate)
        for mutate in ('missing', 'wrong_layer', 'wrong_expert', 'bad_hash',
                       'bad_combined', 'duplicate_combined'):
            gate = self.gate()
            if mutate == 'missing':
                gate['source_layer_identity_sweep'].pop()
            elif mutate == 'wrong_layer':
                gate['source_layer_identity_sweep'][7]['layer'] = 8
            elif mutate == 'wrong_expert':
                gate['source_layer_identity_sweep'][7]['logical_expert'] = 9
            elif mutate == 'bad_hash':
                gate['source_layer_identity_sweep'][7]['up_sha256'] = 'not-a-sha'
            elif mutate == 'bad_combined':
                gate['source_layer_identity_sweep'][7]['combined_sha256'] = 'a' * 64
            else:
                gate['source_layer_identity_sweep'][7]['combined_sha256'] = \
                    gate['source_layer_identity_sweep'][6]['combined_sha256']
            mutations.append(gate)
        with tempfile.TemporaryDirectory() as directory:
            for number, gate in enumerate(mutations):
                with self.subTest(mutation=number), self.assertRaises(RuntimeError):
                    self.write_and_check(directory, gate)

    def test_six_numeric_cases_require_complete_top10_finite_bf16_evidence(self):
        mutations = []
        for action in ('missing', 'duplicate', 'error', 'wrong_tolerance',
                       'not_finite', 'nan', 'zero_peak', 'bad_hash',
                       'incomplete_top10', 'resident_changed'):
            gate = self.gate()
            row = gate['numeric_cases'][0]
            if action == 'missing':
                gate['numeric_cases'].pop()
            elif action == 'duplicate':
                gate['numeric_cases'][1]['kind'] = 'one_group'
            elif action == 'error':
                row['relative_l2'] = 0.011
            elif action == 'wrong_tolerance':
                row['relative_l2_tolerance'] = 0.011
            elif action == 'not_finite':
                row['finite'] = False
            elif action == 'nan':
                row['relative_l2'] = float('nan')
            elif action == 'zero_peak':
                row['execution_peak_extra_bytes'] = 0
            elif action == 'bad_hash':
                row['output_sha256'] = 'bad'
            elif action == 'incomplete_top10':
                row['all_top10_routes_executed'] = False
            else:
                row['template_residents_unchanged'] = False
            mutations.append(gate)
        with tempfile.TemporaryDirectory() as directory:
            for number, gate in enumerate(mutations):
                with self.subTest(mutation=number), self.assertRaises(RuntimeError):
                    self.write_and_check(directory, gate)

    def test_abort_unused_slots_and_private_ownership_evidence_are_mandatory(self):
        dotted_fields = (
            'abort_probe.inflight_abort_synchronized',
            'abort_probe.backend_reusable_after_inflight_abort',
            'abort_probe.public_abort_reuse',
            'unused_slot_probe.unchanged_after_execution',
            'unused_slot_probe.numerically_ignored',
            'original_resident_weights_and_maps_exact',
            'private_shadow_has_no_resident_weight_alias',
        )
        with tempfile.TemporaryDirectory() as directory:
            for dotted in dotted_fields:
                gate = self.gate()
                target = gate
                parts = dotted.split('.')
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = False
                with self.subTest(field=dotted), self.assertRaises(RuntimeError):
                    self.write_and_check(directory, gate)
            gate = self.gate()
            gate['unused_slot_probe']['loaded_unused_slots'] = 21
            with self.assertRaises(RuntimeError):
                self.write_and_check(directory, gate)
            gate = self.gate()
            gate['copied_original_bf16_bytes'] = 0
            with self.assertRaises(RuntimeError):
                self.write_and_check(directory, gate)


if __name__ == '__main__':
    unittest.main()
