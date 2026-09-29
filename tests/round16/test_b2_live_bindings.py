from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from round16.b.selective_loader import plan_q32
from round16.b.selective_preallocation_installer import ROUTED_EXPERTS_Q32_API
from round16_integration.b2_live_bindings import (
    ASCEND_QSA_CACHE_RESHAPE_API,
    ASCEND_QSA_CACHE_SPLIT_API,
    ASCEND_QSA_CACHE_SHAPE_API,
    ASCEND_QSA_MULTIHEAD_STORE_API,
    HOST_PLE_BRIDGE_API,
    PhysicalQ32ExpertMapManager,
    build_live_q32_owner_class,
    install_ascend_qsa_cache_reshape,
    install_ascend_qsa_cache_split,
    install_ascend_qsa_cache_shape,
    install_ascend_qsa_multihead_store,
    install_host_ple_bridge,
    make_ascend_expert_selector,
)


class _Tensor:
    def __init__(self, values):
        self.values = tuple(values)


class _Torch:
    int32 = "int32"

    @staticmethod
    def tensor(values, dtype=None):
        assert dtype == "int32"
        return _Tensor(values)

    @staticmethod
    def device(value):
        return value


class _DeviceTensor:
    def __init__(self, value, device):
        self.value = value
        self.device = device

    def detach(self):
        return self

    def to(self, *, device, non_blocking):
        assert non_blocking is False
        return _DeviceTensor(self.value, device)


class _CacheTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def unsqueeze(self, dimension):
        assert dimension == 0
        return _CacheTensor((1, *self.shape))

    def permute(self, *dimensions):
        return _CacheTensor(tuple(self.shape[index] for index in dimensions))


class _SliceTensor(_CacheTensor):
    def __init__(self, shape, *, name):
        super().__init__(shape)
        self.name = name
        self.indices = []

    def __getitem__(self, index):
        self.indices.append(index)
        return SimpleNamespace(parent=self.name, index=index)


class _PleOffloadLayer:
    @classmethod
    def get_target_device(cls):
        del cls
        return "npu"

    def forward(self, hidden_states, input_ids, *args, **kwargs):
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)


class _PleInstance(_PleOffloadLayer):
    def forward_impl(self, hidden_states, input_ids, query_start, context):
        self.observed = (
            hidden_states.device,
            input_ids.device,
            query_start.device,
            context.device,
        )
        return _DeviceTensor("embedding", "cpu")


class _PostLoadPleOffloadLayer:
    def forward(self, hidden_states, input_ids, *args, **kwargs):
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)


class _PostLoadPleEmbedding:
    def __init__(self):
        self.child = SimpleNamespace()

    def modules(self):
        return (self, self.child)


class _NativeOwner:
    def __init__(
        self,
        layer_name,
        params_dtype,
        moe_config,
        quant_config,
        expert_map_manager,
        **kwargs,
    ):
        del params_dtype, quant_config
        self.layer_name = layer_name
        self.moe_config = moe_config
        self.quant_config = None
        self.expert_map_manager = expert_map_manager
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.quant_method = SimpleNamespace(name="construction")
        self.w13_weight = SimpleNamespace(shape=(32, 8, 16), dtype="bfloat16")
        self.w2_weight = SimpleNamespace(shape=(32, 16, 8), dtype="bfloat16")
        for name, value in kwargs.items():
            setattr(self, name, value)

    def _get_quant_method(self, prefix, quant_config, moe_config):
        del prefix, quant_config, moe_config
        return SimpleNamespace()


class _AscendMethod:
    def __init__(self, config):
        self.config = config


class LiveQ32OwnerTests(unittest.TestCase):
    def test_ascend_qsa_cache_split_restores_omitted_page_axis(self):
        module = SimpleNamespace(
            _split_qsa_kv_cache=lambda cache, head_size: cache,
        )
        install_ascend_qsa_cache_split(qsa_module=module)
        install_ascend_qsa_cache_split(qsa_module=module)
        key, value = module._split_qsa_kv_cache(
            (
                _CacheTensor((3072, 1, 256)),
                _CacheTensor((3072, 1, 256)),
            ),
            256,
        )
        self.assertEqual(key.shape, (1, 3072, 1, 256))
        self.assertEqual(value.shape, (1, 3072, 1, 256))
        self.assertEqual(module.round16_cache_split_api, ASCEND_QSA_CACHE_SPLIT_API)

    def test_ascend_qsa_cache_split_rejects_other_three_dimensional_shape(self):
        module = SimpleNamespace(
            _split_qsa_kv_cache=lambda cache, head_size: cache,
        )
        install_ascend_qsa_cache_split(qsa_module=module)
        with self.assertRaisesRegex(RuntimeError, "invalid shape"):
            module._split_qsa_kv_cache(
                (_CacheTensor((3072, 2, 256)), _CacheTensor((3072, 2, 256))),
                256,
            )

    def test_ascend_qsa_cache_split_permutates_hnd_views_without_copy(self):
        module = SimpleNamespace(
            _split_qsa_kv_cache=lambda cache, head_size: cache,
        )
        install_ascend_qsa_cache_split(qsa_module=module)
        key, value = module._split_qsa_kv_cache(
            (
                _CacheTensor((1, 1, 3072, 256)),
                _CacheTensor((1, 1, 3072, 256)),
            ),
            256,
        )
        self.assertEqual(key.shape, (1, 3072, 1, 256))
        self.assertEqual(value.shape, (1, 3072, 1, 256))

    def test_ascend_qsa_cache_split_accepts_tp1_two_head_nhd_views(self):
        module = SimpleNamespace(
            _split_qsa_kv_cache=lambda cache, head_size: cache,
        )
        install_ascend_qsa_cache_split(qsa_module=module)
        key, value = module._split_qsa_kv_cache(
            (
                _CacheTensor((102, 3072, 2, 256)),
                _CacheTensor((102, 3072, 2, 256)),
            ),
            256,
        )
        self.assertEqual(key.shape, (102, 3072, 2, 256))
        self.assertEqual(value.shape, (102, 3072, 2, 256))

    def test_ascend_qsa_multihead_store_writes_each_head_once(self):
        calls = []
        key_cache = _SliceTensor((102, 3072, 2, 256), name="key-cache")
        value_cache = _SliceTensor((102, 3072, 2, 256), name="value-cache")
        key = _SliceTensor((5, 2, 256), name="key")
        value = _SliceTensor((5, 2, 256), name="value")

        def store_rows(cache, slot_mapping, rows):
            calls.append((cache, slot_mapping, rows))

        module = SimpleNamespace(
            _split_qsa_kv_cache=lambda cache, head_size: cache,
            qsa_store_cache_rows=store_rows,
        )

        class QSAImpl:
            head_size = 256
            num_kv_heads = 2

            def do_kv_cache_update(
                self, layer, key, value, kv_cache, slot_mapping
            ):
                del self, layer, key, value, kv_cache, slot_mapping
                raise AssertionError("the tuple path must use the multi-head store")

        install_ascend_qsa_multihead_store(
            qsa_impl_cls=QSAImpl,
            qsa_module=module,
        )
        install_ascend_qsa_multihead_store(
            qsa_impl_cls=QSAImpl,
            qsa_module=module,
        )
        slots = object()
        QSAImpl().do_kv_cache_update(
            object(), key, value, (key_cache, value_cache), slots
        )

        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call[1] is slots for call in calls))
        self.assertEqual(
            [call[0].parent for call in calls],
            ["key-cache", "value-cache", "key-cache", "value-cache"],
        )
        self.assertEqual(
            [call[2].parent for call in calls],
            ["key", "value", "key", "value"],
        )
        self.assertEqual(
            QSAImpl.round16_multihead_store_api,
            ASCEND_QSA_MULTIHEAD_STORE_API,
        )

    def test_ascend_qsa_multihead_store_rejects_row_head_mismatch(self):
        module = SimpleNamespace(
            _split_qsa_kv_cache=lambda cache, head_size: cache,
            qsa_store_cache_rows=lambda *args: None,
        )

        class QSAImpl:
            head_size = 256
            num_kv_heads = 2

            def do_kv_cache_update(self, *args):
                del self, args

        install_ascend_qsa_multihead_store(
            qsa_impl_cls=QSAImpl,
            qsa_module=module,
        )
        with self.assertRaisesRegex(RuntimeError, "rows do not match"):
            QSAImpl().do_kv_cache_update(
                object(),
                _SliceTensor((5, 1, 256), name="key"),
                _SliceTensor((5, 1, 256), name="value"),
                (
                    _SliceTensor((102, 3072, 2, 256), name="key-cache"),
                    _SliceTensor((102, 3072, 2, 256), name="value-cache"),
                ),
                object(),
            )


    def test_ascend_qsa_cache_reshape_restores_single_block_axis(self):
        class QSAImpl:
            pass

        qsa_layer = SimpleNamespace(
            impl=QSAImpl(),
            num_kv_heads=1,
            head_dim=256,
        )
        other_layer = SimpleNamespace(impl=object())

        class Runner:
            def __init__(self):
                self.compilation_config = SimpleNamespace(
                    static_forward_context={
                        "qsa.attn": qsa_layer,
                        "other": other_layer,
                    }
                )

            def _reshape_kv_cache_tensors(self, *args, **kwargs):
                del args, kwargs
                return {
                    "qsa.attn": (
                        _CacheTensor((3072, 1, 256)),
                        _CacheTensor((3072, 1, 256)),
                    ),
                    "other": "unchanged",
                }

        install_ascend_qsa_cache_reshape(
            model_runner_cls=Runner,
            qsa_impl_cls=QSAImpl,
        )
        install_ascend_qsa_cache_reshape(
            model_runner_cls=Runner,
            qsa_impl_cls=QSAImpl,
        )
        result = Runner()._reshape_kv_cache_tensors()
        self.assertEqual(result["qsa.attn"][0].shape, (1, 3072, 1, 256))
        self.assertEqual(result["qsa.attn"][1].shape, (1, 3072, 1, 256))
        self.assertEqual(result["other"], "unchanged")
        self.assertEqual(
            Runner.round16_qsa_cache_reshape_api,
            ASCEND_QSA_CACHE_RESHAPE_API,
        )

    def test_ascend_qsa_cache_shape_uses_split_native_contract(self):
        class Backend:
            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                del args, kwargs
                return (7, 1, 16, 512)

        install_ascend_qsa_cache_shape(qsa_backend_cls=Backend)
        install_ascend_qsa_cache_shape(qsa_backend_cls=Backend)
        self.assertEqual(
            Backend.get_kv_cache_shape(7, 16, 1, 256),
            (2, 7, 16, 1, 256),
        )
        self.assertEqual(
            Backend.round16_cache_shape_api,
            ASCEND_QSA_CACHE_SHAPE_API,
        )
        with self.assertRaisesRegex(ValueError, "positive integers"):
            Backend.get_kv_cache_shape(0, 16, 1, 256)

    def test_host_ple_bridge_keeps_embedding_on_cpu_and_returns_to_npu(self):
        install_host_ple_bridge(
            torch_module=_Torch,
            ple_offload_cls=_PleOffloadLayer,
        )
        install_host_ple_bridge(
            torch_module=_Torch,
            ple_offload_cls=_PleOffloadLayer,
        )
        ple = _PleInstance()
        result = ple.forward(
            _DeviceTensor("hidden", "npu:0"),
            _DeviceTensor("ids", "npu:0"),
            _DeviceTensor("starts", "npu:0"),
            _DeviceTensor("context", "npu:0"),
        )
        self.assertEqual(_PleOffloadLayer.get_target_device(), "cpu")
        self.assertEqual(_PleOffloadLayer.round16_host_ple_api, HOST_PLE_BRIDGE_API)
        self.assertEqual(ple.observed, ("cpu", "cpu", "cpu", "cpu"))
        self.assertEqual(result.device, "npu:0")

    def test_host_ple_postload_keeps_only_marked_subtree_on_cpu(self):
        observed = []

        @contextmanager
        def device_loading_context(module, target_device):
            observed.append((module, target_device))
            yield module

        patch_module = SimpleNamespace(
            device_loading_context=device_loading_context,
        )
        install_host_ple_bridge(
            torch_module=_Torch,
            ple_offload_cls=_PostLoadPleOffloadLayer,
            ple_embedding_cls=_PostLoadPleEmbedding,
            postload_patch_module=patch_module,
        )
        embedding = _PostLoadPleEmbedding()
        normal = SimpleNamespace()
        with patch_module.device_loading_context(embedding, "npu:0"):
            pass
        with patch_module.device_loading_context(embedding.child, "npu:0"):
            pass
        with patch_module.device_loading_context(normal, "npu:0"):
            pass
        self.assertEqual(
            [target for _, target in observed],
            ["cpu", "cpu", "npu:0"],
        )

    def test_physical_manager_keeps_logical_domain(self):
        plan = plan_q32(tuple(range(32)))
        manager = PhysicalQ32ExpertMapManager(_Torch, plan.logical_to_physical)
        self.assertEqual(manager.global_num_experts, 512)
        self.assertEqual(manager.local_num_experts, 32)
        self.assertEqual(manager.map_global_to_local(0), 0)
        self.assertEqual(manager.map_global_to_local(31), 31)
        self.assertEqual(manager.map_global_to_local(32), -1)

    def test_ascend_selector_preserves_logical_512_contract(self):
        observed = {}
        registrations = []

        def ascend_selector(**kwargs):
            observed.update(kwargs)
            return "weights", "ids"

        def enable_custom_op():
            registrations.append("registered")
            return True

        selector = make_ascend_expert_selector(
            ascend_selector,
            enable_custom_op=enable_custom_op,
        )
        owner = SimpleNamespace(
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
        )
        runner = SimpleNamespace(
            routed_experts=owner,
            moe_config=SimpleNamespace(experts_per_token=10),
            _quant_method=SimpleNamespace(topk_indices_dtype="int32"),
        )
        self.assertEqual(
            selector(
                runner=runner,
                hidden_states="hidden",
                router_logits="logits",
                input_ids="ids-in",
            ),
            ("weights", "ids"),
        )
        self.assertEqual(observed["num_experts"], 512)
        self.assertEqual(observed["num_logical_experts"], 512)
        self.assertEqual(observed["top_k"], 10)
        self.assertEqual(observed["input_ids"], "ids-in")
        self.assertFalse(observed["mix_placement"])
        selector(
            runner=runner,
            hidden_states="hidden-2",
            router_logits="logits-2",
            input_ids="ids-in-2",
        )
        self.assertEqual(registrations, ["registered"])

    def test_ascend_selector_rejects_failed_custom_op_registration(self):
        selector = make_ascend_expert_selector(
            lambda **kwargs: kwargs,
            enable_custom_op=lambda: False,
        )
        owner = SimpleNamespace(
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
        )
        runner = SimpleNamespace(
            routed_experts=owner,
            moe_config=SimpleNamespace(experts_per_token=10),
            _quant_method=SimpleNamespace(topk_indices_dtype="int32"),
        )
        with self.assertRaisesRegex(RuntimeError, "registration was unavailable"):
            selector(
                runner=runner,
                hidden_states="hidden",
                router_logits="logits",
                input_ids="ids-in",
            )

    def test_live_owner_switches_to_ascend_before_load_and_sets_comm(self):
        plan = plan_q32(tuple(range(32)))
        setups = []
        cls = build_live_q32_owner_class(
            native_routed_experts_cls=_NativeOwner,
            torch_module=_Torch,
            ascend_method_cls=_AscendMethod,
            setup_moe_comm_method=lambda config: setups.append(config),
            plan=plan,
        )
        logical = SimpleNamespace(
            num_experts=512,
            num_logical_experts=512,
            num_local_experts=512,
        )
        owner = cls(
            "language_model.model.layers.0.mlp.experts",
            "bf16",
            logical,
            None,
            expert_map_manager=object(),
            round16_selective_plan=plan,
            round16_resident_logical_ids=plan.resident_logical_ids,
            round16_logical_to_physical=plan.logical_to_physical,
            round16_logical_expert_count=512,
            round16_physical_expert_count=32,
        )
        self.assertEqual(cls.round16_q32_api_identity, ROUTED_EXPERTS_Q32_API)
        self.assertTrue(owner.runtime_runner_binding_verified)
        self.assertEqual(owner.moe_config.num_local_experts, 32)
        self.assertIsInstance(owner.quant_method, _AscendMethod)
        self.assertEqual(setups, [owner.moe_config])


if __name__ == "__main__":
    unittest.main()
