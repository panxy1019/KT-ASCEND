import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
import sys

from migration_bridge.round9.runtime.candidate_config import (
    runtime_arguments,
    validate_kt_config,
    Candidate,
)
from migration_bridge.round9.runtime.framework_hooks import (
    collect_q32_physical_layers,
    framework_before,
    prepare_masks,
)
from migration_bridge.round9.runtime.dynamic_boundary import _framework_capacity, _require_capacity_gate
from migration_bridge.round9.runtime.candidate_server import require_capacity_device_gate


PROFILE = dict(workers=128, storage='BF16', placement='static', numa_nodes=[2, 3, 6, 7])


class Values:
    def __init__(self, values, dtype, device):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = dtype
        self.device = NS(type=device)
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(self.values)
    def sum(self): return sum(self.values)


class Tensor:
    def __init__(self, shape, elements, device='npu'):
        self.shape = shape
        self.dtype = 'torch.bfloat16'
        self.device = NS(type=device)
        self._elements = elements
    def numel(self): return self._elements
    def element_size(self): return 2


class Method:
    pass


def method(layer, capacity=32):
    value = Method()
    selected = list(range(capacity))
    logical_to_gpu = list(range(capacity)) + [-1] * (512 - capacity)
    value.kt_config = NS(layer_idx=layer)
    mask = [True] * capacity + [False] * (512 - capacity)
    value.wrapper = NS(num_experts=512,
                       gpu_experts_mask=Values(mask, 'torch.bool', 'cpu'))
    value.num_gpu_experts = capacity
    value.gpu_experts_mask = Values(mask, 'torch.bool', 'cpu')
    value.gpu_experts_mask_cuda = Values(mask, 'torch.bool', 'npu')
    value.logical_to_gpu_index = Values(logical_to_gpu, 'torch.int32', 'cpu')
    value.logical_to_gpu_index_cuda = Values(logical_to_gpu, 'torch.int32', 'npu')
    value.gpu_index_to_logical = Values(selected, 'torch.int32', 'cpu')
    value.w13_weight = Tensor((capacity, 2560, 1280), capacity * 2560 * 1280)
    value.w2_weight = Tensor((capacity, 640, 2560), capacity * 640 * 2560)
    return value


class Q32IntegrationTests(unittest.TestCase):
    def config(self, capacity):
        return NS(gpu_experts_mask=Values([True] * capacity + [False] * (512 - capacity),
                                          'torch.bool', 'cpu'), cpuinfer_threads=128,
                  threadpool_count=4, numa_nodes=[2, 3, 6, 7], method='LLAMAFILE',
                  max_deferred_experts_per_token=0, kt_enable_dynamic_expert_update=False,
                  gpu_prefill_token_threshold=0)

    def test_candidate_identity_stays_four_fields_and_capacity_is_separate(self):
        candidate = Candidate.from_dict(PROFILE)
        self.assertEqual(runtime_arguments(candidate)['kt_num_gpu_experts'], 64)
        self.assertEqual(runtime_arguments(candidate, resident_capacity=32)['kt_num_gpu_experts'], 32)
        self.assertEqual(validate_kt_config(self.config(64), PROFILE, native_bf16_tail=True), candidate)
        self.assertEqual(validate_kt_config(self.config(32), PROFILE, native_bf16_tail=True,
                                            resident_capacity=32), candidate)
        with self.assertRaises(RuntimeError):
            validate_kt_config(self.config(32), PROFILE, native_bf16_tail=True)
        for invalid in (0, 31, 48, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime_arguments(candidate, resident_capacity=invalid)

    def test_prepare_masks_returns_explicit_capacity_and_preserves_off_default(self):
        self.assertEqual(prepare_masks(NS(), NS()), 64)
        framework = dict(schema=1, resident_capacity=32, route_collection=False, frequency=None)
        args = NS(kt_num_gpu_experts=32, kt_gpu_experts_ratio=None,
                  kt_expert_placement_strategy='uniform', tp_size=1,
                  kt_enable_dynamic_expert_update=False)
        with patch('migration_bridge.round9.runtime.framework_hooks.load_framework', return_value=framework):
            self.assertEqual(prepare_masks(args, NS()), 32)

    def test_all48_physical_q32_layers_have_exact_maps_geometry_and_bytes(self):
        layers = [NS(quant_method=method(i)) for i in range(48)]
        document = collect_q32_physical_layers(NS(modules=lambda: layers), Method)
        self.assertEqual(document['summary']['layers'], 48)
        self.assertEqual(document['summary']['total_bytes'], 48 * 314572800)
        self.assertEqual(document['layers'][0]['tensors']['w13_weight']['shape'], [32, 2560, 1280])
        self.assertEqual(document['maps'][0]['gpu_index_to_logical'], list(range(32)))
        self.assertEqual(document['devices'][0],
                         {'layer': 0, 'w13_weight': 'npu', 'w2_weight': 'npu'})
        layers[17].quant_method.gpu_index_to_logical = Values(
            list(reversed(range(32))), 'torch.int32', 'cpu')
        with self.assertRaises(RuntimeError):
            collect_q32_physical_layers(NS(modules=lambda: layers), Method)

    def test_physical_record_rejects_npu_mask_map_or_cpu_weight(self):
        mutations = (
            lambda value: setattr(value, 'gpu_experts_mask_cuda',
                Values([False] + [True] * 32 + [False] * 479, 'torch.bool', 'npu')),
            lambda value: setattr(value, 'logical_to_gpu_index_cuda',
                Values([1] + list(range(1, 32)) + [-1] * 480, 'torch.int32', 'npu')),
            lambda value: setattr(value, 'w13_weight',
                Tensor((32, 2560, 1280), 32 * 2560 * 1280, device='cpu')),
        )
        for mutate in mutations:
            layers = [NS(quant_method=method(i)) for i in range(48)]
            mutate(layers[11].quant_method)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                collect_q32_physical_layers(NS(modules=lambda: layers), Method)

    def test_first_scheduler_batch_records_q32_even_without_route_collection(self):
        framework = dict(schema=1, resident_capacity=32, route_collection=False, frequency=None)
        scheduler = NS()
        with patch('migration_bridge.round9.runtime.framework_hooks.load_framework', return_value=framework), \
             patch('migration_bridge.round9.runtime.framework_hooks.write_q32_physical') as write:
            framework_before(scheduler, object())
            framework_before(scheduler, object())
        write.assert_called_once_with(scheduler)

    def test_framework_installer_passes_capacity_and_gates_before_model_binding(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        sys.path.insert(0, str(tools))
        try:
            from install_candidate_overlay import candidate_model_source
            from install_framework_overlay import patch_model
            baseline = (Path(__file__).resolve().parents[1] /
                        'sglang_overlay/python/sglang/srt/models/qwen4_exp.py').read_text()
            source = patch_model(candidate_model_source(baseline))
        finally:
            sys.path.remove(str(tools))
        compile(source, 'qwen4_exp.py', 'exec')
        self.assertIn('round9_resident_capacity = prepare_masks(args, kt_ep_wrapper)', source)
        self.assertIn('resident_capacity=round9_resident_capacity', source)
        gate = source.index('validate_q32_model_gate(round9_resident_capacity, candidate, bank_identity)')
        binding = source.index('bind_model_profile(candidate_path, layer_id, bank_identity)')
        self.assertLess(gate, binding)

    def test_server_and_dynamic_boundary_select_q32_validator_with_native_sha(self):
        candidate = Candidate.from_dict(PROFILE)
        identity = dict(source_identity={'model_path': '/models'}, layer0_sha256='b' * 64)
        binding = dict(bank_identity=identity)
        with patch('migration_bridge.round9.runtime.q32_gate.require_q32_device_gate') as dynamic_gate:
            _require_capacity_gate(32, candidate, binding, '/gate', 'a' * 64,
                                   native_sha256='c' * 64)
        self.assertEqual(dynamic_gate.call_args.kwargs['native_sha256'], 'c' * 64)

        q32 = NS(require_q32_device_gate=unittest.mock.Mock())
        modules = {
            'sglang': NS(), 'sglang.srt': NS(), 'sglang.srt.layers': NS(),
            'sglang.srt.layers.round9': NS(), 'sglang.srt.layers.round9.q32_gate': q32,
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'ROUND9_DYNAMIC_DEVICE_GATE': '/gate'}):
            require_capacity_device_gate(32, candidate, identity,
                                         {'device_gate': 'a' * 64}, 'c' * 64)
        self.assertEqual(q32.require_q32_device_gate.call_args.kwargs['native_sha256'], 'c' * 64)

    def test_default_q64_dynamic_path_still_uses_historical_validator(self):
        candidate = Candidate.from_dict(PROFILE | {'placement': 'request_boundary'})
        identity = dict(source_identity={'model_path': '/models'}, layer0_sha256='b' * 64)
        binding = dict(bank_identity=identity)
        with patch('migration_bridge.round9.runtime.candidate_config.require_dynamic_device_gate') as old_gate, \
             patch('migration_bridge.round9.runtime.q32_gate.require_q32_device_gate') as q32_gate:
            _require_capacity_gate(64, candidate, binding, '/gate', 'a' * 64)
        old_gate.assert_called_once()
        q32_gate.assert_not_called()

    def test_candidate_only_overlay_without_framework_module_defaults_q64(self):
        import builtins
        original = builtins.__import__
        def without_framework(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'framework_config' and level == 1:
                error = ModuleNotFoundError('candidate-only overlay')
                error.name = 'migration_bridge.round9.runtime.framework_config'
                raise error
            return original(name, globals, locals, fromlist, level)
        with patch('builtins.__import__', side_effect=without_framework):
            self.assertEqual(_framework_capacity(), 64)


if __name__ == '__main__': unittest.main()
