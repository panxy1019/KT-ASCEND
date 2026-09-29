from __future__ import annotations

import importlib
from types import SimpleNamespace
import unittest


def _module():
    try:
        return importlib.import_module("round16.b.source_bound_q32_adapter")
    except ModuleNotFoundError as error:
        raise AssertionError("round16.b.source_bound_q32_adapter must exist") from error


class _Parameter:
    def __init__(self, shape):
        self.shape = tuple(shape)


class _NativeQuantMethod:
    supports_eplb = True

    def __init__(self, events, *, fail=False, wrong_shape=False):
        self.events = events
        self.fail = fail
        self.wrong_shape = wrong_shape

    def maybe_roundup_sizes(self, hidden_size, intermediate_size, params_dtype, parallel):
        return hidden_size, intermediate_size

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        self.events.append(("allocate", num_experts, extra_weight_attrs["global_num_experts"]))
        count = 31 if self.wrong_shape else num_experts
        layer.w13_weight = _Parameter((count, 16, hidden_size))
        layer.w2_weight = _Parameter((count, hidden_size, 8))
        if self.fail:
            raise RuntimeError("injected native allocation failure")


class _NativeRoutedExperts:
    events = []
    quant_fail = False
    quant_wrong_shape = False

    def __init__(
        self,
        layer_name,
        params_dtype,
        moe_config,
        quant_config,
        expert_map_manager,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        **kwargs,
    ):
        del kwargs
        self.layer_name = layer_name
        self.params_dtype = params_dtype
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.expert_map_manager = expert_map_manager
        self.ckpt_gate_proj_name = ckpt_gate_proj_name
        self.ckpt_down_proj_name = ckpt_down_proj_name
        self.ckpt_up_proj_name = ckpt_up_proj_name
        self.global_num_experts = moe_config.num_experts
        self.local_num_experts = moe_config.num_local_experts
        self.quant_method = self._get_quant_method(layer_name, quant_config, moe_config)
        self.quant_method.create_weights(
            layer=self,
            num_experts=moe_config.num_local_experts,
            hidden_size=moe_config.hidden_dim,
            intermediate_size_per_partition=moe_config.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
            global_num_experts=moe_config.num_experts,
        )

    def _get_quant_method(self, prefix, quant_config, moe_config):
        del prefix, quant_config, moe_config
        return _NativeQuantMethod(
            self.events,
            fail=self.quant_fail,
            wrong_shape=self.quant_wrong_shape,
        )

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        physical = self.expert_map_manager.map_global_to_local(expert_id)
        self.events.append(("load", weight_name, shard_id, expert_id, physical))
        return physical != -1 if return_success else None


class _Manager:
    def __init__(self, mapping):
        self.mapping = tuple(mapping)
        self.global_num_experts = 512
        self.local_num_experts = 32

    def map_global_to_local(self, expert_id):
        return self.mapping[expert_id]


class SourceBoundQ32AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        _NativeRoutedExperts.events = []
        _NativeRoutedExperts.quant_fail = False
        _NativeRoutedExperts.quant_wrong_shape = False

    def _build(self, resident=tuple(range(32)), *, prepared_mapper=None):
        module = _module()
        loader = importlib.import_module("round16.b.selective_loader")
        plan = loader.plan_q32(resident)
        events = _NativeRoutedExperts.events
        rollback_calls = []

        def prepare(moe_config, manager, selected_plan):
            self.assertIs(selected_plan, plan)
            events.append(("prepare", moe_config.num_local_experts))
            physical_config = SimpleNamespace(**vars(moe_config))
            physical_config.num_local_experts = 32
            physical_manager = _Manager(selected_plan.logical_to_physical)
            if prepared_mapper is not None:
                physical_manager.map_global_to_local = prepared_mapper
            return module.PreparedQ32State(
                moe_config=physical_config,
                expert_map_manager=physical_manager,
                logical_to_physical=selected_plan.logical_to_physical,
                rollback_token="prepared",
            )

        prepare.round16_api_identity = module.Q32_PHYSICAL_CONFIG_PREPARER_API

        def rollback(owner, state):
            rollback_calls.append((owner, state.rollback_token))
            for name in ("w13_weight", "w2_weight"):
                if hasattr(owner, name):
                    delattr(owner, name)

        rollback.round16_api_identity = module.Q32_ALLOCATION_ROLLBACK_API
        routed_cls = module.build_source_bound_q32_routed_experts_class(
            _NativeRoutedExperts,
            physical_config_preparer=prepare,
            allocation_rollback=rollback,
        )
        return module, plan, routed_cls, rollback_calls

    def test_prepared_manager_mapper_must_exactly_match_all_512_plan_entries(self):
        cases = {
            "swapped": lambda expert_id: (
                1 if expert_id == 0 else 0 if expert_id == 1 else expert_id
            ) if expert_id < 32 else -1,
            "none": lambda _expert_id: None,
            "identity": lambda expert_id: expert_id,
        }
        for name, mapper in cases.items():
            with self.subTest(name=name):
                self.setUp()
                _module_value, plan, routed_cls, _rollback_calls = self._build(
                    prepared_mapper=mapper
                )
                with self.assertRaisesRegex(ValueError, "mapper"):
                    self._construct(routed_cls, plan)
                self.assertFalse(
                    any(event[0] == "allocate" for event in _NativeRoutedExperts.events)
                )

    @staticmethod
    def _construct(routed_cls, plan):
        config = SimpleNamespace(
            num_experts=512,
            num_logical_experts=512,
            num_local_experts=512,
            hidden_dim=8,
            intermediate_size_per_partition=4,
        )
        manager = _Manager((-1,) * 512)
        manager.local_num_experts = 512
        return routed_cls(
            "model.layers.0.mlp.experts",
            "bf16",
            config,
            None,
            expert_map_manager=manager,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            round16_selective_plan=plan,
            round16_resident_logical_ids=plan.resident_logical_ids,
            round16_logical_to_physical=plan.logical_to_physical,
            round16_logical_expert_count=512,
            round16_physical_expert_count=32,
        )

    def test_prepares_physical32_before_native_create_weights_and_keeps_logical512(self) -> None:
        module, plan, routed_cls, rollback_calls = self._build()
        owner = self._construct(routed_cls, plan)

        self.assertEqual(
            _NativeRoutedExperts.events[:2],
            [("prepare", 512), ("allocate", 32, 512)],
        )
        self.assertEqual(owner.local_num_experts, 32)
        self.assertEqual(owner.global_num_experts, 512)
        self.assertEqual(owner.round16_upstream_moe_config.num_local_experts, 512)
        self.assertIsNot(owner.moe_config, owner.round16_upstream_moe_config)
        self.assertEqual(
            owner.round16_upstream_expert_map_manager.local_num_experts,
            512,
        )
        self.assertEqual(
            owner.round16_upstream_expert_map_manager.mapping,
            (-1,) * 512,
        )
        self.assertIsNot(
            owner.expert_map_manager,
            owner.round16_upstream_expert_map_manager,
        )
        self.assertEqual(owner.w13_weight.shape[0], 32)
        self.assertEqual(owner.w2_weight.shape[0], 32)
        self.assertEqual(
            owner.round16_source_bound_api_identity,
            module.SOURCE_BOUND_Q32_ROUTED_EXPERTS_API,
        )
        self.assertFalse(owner.runtime_runner_binding_verified)
        self.assertNotEqual(
            getattr(owner, "round16_q32_api_identity", None),
            module.ROUTED_EXPERTS_Q32_API,
        )
        self.assertEqual(rollback_calls, [])

    def test_global_to_local_checkpoint_mapping_uses_w13_alias_and_w2(self) -> None:
        _module, plan, routed_cls, _rollback_calls = self._build()
        owner = self._construct(routed_cls, plan)
        mapping = owner.get_round16_checkpoint_bindings()

        self.assertEqual(len(mapping), 32 * 3)
        first = [entry for entry in mapping if entry.logical_expert_id == 0]
        self.assertEqual([entry.shard_id for entry in first], ["w1", "w2", "w3"])
        self.assertEqual(first[0].parameter_name, "experts.w13_weight")
        self.assertEqual(first[2].parameter_name, "experts.w13_weight")
        self.assertEqual(first[1].parameter_name, "experts.w2_weight")
        self.assertIn(".gate_proj.", first[0].checkpoint_name)
        self.assertIn(".down_proj.", first[1].checkpoint_name)
        self.assertIn(".up_proj.", first[2].checkpoint_name)
        self.assertEqual(first[0].physical_expert_id, 0)

        loaded = owner.weight_loader(
            owner.w13_weight,
            object(),
            first[0].checkpoint_name,
            "w1",
            0,
            return_success=True,
        )
        skipped = owner.weight_loader(
            owner.w13_weight,
            object(),
            "experts.100.gate_proj.weight",
            "w1",
            100,
            return_success=True,
        )
        self.assertTrue(loaded)
        self.assertFalse(skipped)
        self.assertEqual(plan.logical_to_physical[100], -1)

    def test_native_allocation_error_rolls_back_once_and_preserves_primary(self) -> None:
        _module, plan, routed_cls, rollback_calls = self._build()
        _NativeRoutedExperts.quant_fail = True
        with self.assertRaisesRegex(RuntimeError, "native allocation failure"):
            self._construct(routed_cls, plan)
        self.assertEqual(len(rollback_calls), 1)
        self.assertEqual(rollback_calls[0][1], "prepared")

    def test_wrong_shape_or_tampered_installer_args_fail_closed(self) -> None:
        module, plan, routed_cls, rollback_calls = self._build()
        _NativeRoutedExperts.quant_wrong_shape = True
        with self.assertRaisesRegex(RuntimeError, "physical expert dimension"):
            self._construct(routed_cls, plan)
        self.assertEqual(len(rollback_calls), 1)

        config = SimpleNamespace(
            num_experts=512,
            num_logical_experts=512,
            num_local_experts=512,
            hidden_dim=8,
            intermediate_size_per_partition=4,
        )
        manager = _Manager((-1,) * 512)
        with self.assertRaisesRegex(ValueError, "immutable Q32 installer arguments"):
            routed_cls(
                "layer",
                "bf16",
                config,
                None,
                expert_map_manager=manager,
                round16_selective_plan=plan,
                round16_resident_logical_ids=plan.resident_logical_ids,
                round16_logical_to_physical=plan.logical_to_physical,
                round16_logical_expert_count=512,
                round16_physical_expert_count=31,
            )
        self.assertEqual(_NativeRoutedExperts.events[-1][0], "allocate")

    def test_unverified_preparer_or_rollback_identity_is_rejected_before_use(self) -> None:
        module = _module()

        def callback(*args):
            raise AssertionError("unverified callback must not run")

        with self.assertRaisesRegex(ValueError, "preparer API identity"):
            module.build_source_bound_q32_routed_experts_class(
                _NativeRoutedExperts,
                physical_config_preparer=callback,
                allocation_rollback=callback,
            )


if __name__ == "__main__":
    unittest.main()
