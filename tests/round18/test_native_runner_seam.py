from __future__ import annotations

from dataclasses import dataclass
import unittest

from round16.b.selective_loader import plan_q32
from round18.native_runner_seam import (
    ModelScopedNativePartialInstaller,
    bind_q32_map_to_native_runner,
    execution_map_from_owner,
    rebind_native_physical_quant_method,
    split_topk_occurrences,
)


class _Weight:
    def __init__(self, rows: int) -> None:
        self.shape = (rows, 8, 4)


class _Manager:
    def __init__(self, mapping: tuple[int, ...]) -> None:
        self._expert_map = mapping


class _Owner:
    def __init__(self, plan) -> None:
        self.round16_selective_plan = plan
        self.round16_logical_to_physical = plan.logical_to_physical
        self.round16_physical_expert_count = plan.physical_expert_count
        self.expert_map_manager = _Manager(plan.logical_to_physical)
        self.w13_weight = _Weight(32)
        self.w2_weight = _Weight(32)
        self.moe_config = _Config(num_local_experts=32)
        self.quant_method = object()


@dataclass
class _Config:
    num_experts: int = 512
    num_logical_experts: int = 512
    num_local_experts: int = 512
    ep_size: int = 1


class _PhysicalMethod:
    def __init__(self, moe_config, tid2eid=None) -> None:
        self.moe_config = moe_config
        self.tid2eid = tid2eid


class AscendMoERunner:
    def __init__(self, owner: _Owner) -> None:
        self.routed_experts = owner
        self.moe_config = _Config()
        self.dynamic_eplb = False
        self._expert_map = None
        self.global_expert_map = None
        self.log2phy = None
        self.global_redundant_expert_num = 0
        self.tid2eid = None

    @property
    def _quant_method(self):
        return self.routed_experts.quant_method


class _QwenModule:
    pass


class NativeRunnerSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = plan_q32(tuple(range(32)))

    def test_bind_preserves_native_runner_class_and_logical_capacity(self) -> None:
        owner = _Owner(self.plan)
        runner = AscendMoERunner(owner)
        original_class = type(runner)

        receipt = bind_q32_map_to_native_runner(runner, self.plan)

        self.assertIs(type(runner), original_class)
        self.assertEqual(runner.moe_config.num_experts, 512)
        self.assertEqual(runner.moe_config.num_local_experts, 512)
        self.assertEqual(tuple(runner._expert_map), self.plan.logical_to_physical)
        self.assertEqual(
            tuple(owner.expert_map_manager._expert_map),
            self.plan.logical_to_physical,
        )
        self.assertEqual(receipt.runner_class, "AscendMoERunner")
        self.assertEqual(receipt.physical_slots, 32)

    def test_bind_rejects_dynamic_eplb_and_non_q32_storage(self) -> None:
        owner = _Owner(self.plan)
        runner = AscendMoERunner(owner)
        runner.dynamic_eplb = True
        with self.assertRaisesRegex(ValueError, "dynamic EPLB"):
            bind_q32_map_to_native_runner(runner, self.plan)

        runner.dynamic_eplb = False
        owner.w2_weight = _Weight(512)
        with self.assertRaisesRegex(ValueError, "physical expert dimension"):
            bind_q32_map_to_native_runner(runner, self.plan)

    def test_execution_map_accepts_frozen_public_manager_field(self) -> None:
        owner = _Owner(self.plan)
        owner.expert_map_manager.expert_map = owner.expert_map_manager._expert_map
        del owner.expert_map_manager._expert_map
        execution_map = execution_map_from_owner(owner, self.plan)
        self.assertEqual(tuple(execution_map), self.plan.logical_to_physical)

    def test_model_scoped_installer_does_not_supply_runner_override(self) -> None:
        qwen = _QwenModule()
        calls = []
        runtime_bindings = []

        def physical_runtime_binding(runner):
            runtime_bindings.append(runner)
            return rebind_native_physical_quant_method(
                runner,
                ascend_method_cls=_PhysicalMethod,
                setup_moe_comm_method=lambda config: setattr(
                    runner, "physical_comm_config", config
                ),
            )

        def ascend_factory(*args, **kwargs):
            calls.append((args, dict(kwargs)))
            self.assertNotIn("runner_cls", kwargs)
            self.assertNotIn("runner_args", kwargs)
            owner_cls = kwargs["routed_experts_cls"]
            owner = owner_cls(kwargs["routed_experts_args"]["round16_selective_plan"])
            return AscendMoERunner(owner)

        qwen.FusedMoE = ascend_factory

        class Owner(_Owner):
            pass

        installer = ModelScopedNativePartialInstaller(
            qwen_module=qwen,
            ascend_fused_moe=ascend_factory,
            routed_experts_cls=Owner,
            physical_runtime_binding=physical_runtime_binding,
            expected_layer_count=2,
        )
        with installer.activate(plan=self.plan) as acknowledge:
            first = qwen.FusedMoE(prefix="layers.0.experts")
            second = qwen.FusedMoE(prefix="layers.1.experts")
            self.assertTrue(acknowledge(object()))

        self.assertIs(qwen.FusedMoE, ascend_factory)
        self.assertEqual(len(calls), 2)
        self.assertIs(type(first), AscendMoERunner)
        self.assertIs(type(second), AscendMoERunner)
        self.assertEqual(runtime_bindings, [first, second])
        self.assertEqual(first.physical_comm_config.num_local_experts, 32)
        self.assertIsInstance(first._quant_method, _PhysicalMethod)

    def test_physical_runtime_rejects_logical_owner_config(self) -> None:
        owner = _Owner(self.plan)
        owner.moe_config.num_local_experts = 512
        runner = AscendMoERunner(owner)
        with self.assertRaisesRegex(ValueError, "logical512/global512/local32"):
            rebind_native_physical_quant_method(
                runner,
                ascend_method_cls=_PhysicalMethod,
                setup_moe_comm_method=lambda _config: None,
            )

    def test_split_keeps_original_weights_without_renormalization(self) -> None:
        ids = ((0, 32, 1, 33), (40, 2, 3, 41))
        weights = ((0.41, 0.31, 0.17, 0.11), (0.40, 0.30, 0.20, 0.10))
        split = split_topk_occurrences(ids, weights, self.plan)

        self.assertEqual(split.resident_logical_ids[0], (0, -1, 1, -1))
        self.assertEqual(split.resident_physical_ids[0], (0, -1, 1, -1))
        self.assertEqual(split.cpu_logical_ids[0], (-1, 32, -1, 33))
        self.assertEqual(split.resident_weights[0], (0.41, 0.0, 0.17, 0.0))
        self.assertEqual(split.cpu_weights[0], (0.0, 0.31, 0.0, 0.11))
        self.assertEqual(
            tuple(
                a + b
                for a, b in zip(
                    split.resident_weights[0], split.cpu_weights[0], strict=True
                )
            ),
            weights[0],
        )


if __name__ == "__main__":
    unittest.main()
