from __future__ import annotations

import importlib
import math
import unittest

from round16.common.contracts import RouteBatch


def _routing_module():
    try:
        return importlib.import_module("round16.b.routing_bridge")
    except ModuleNotFoundError as error:
        raise AssertionError("round16.b.routing_bridge must exist") from error


def _row(first: int, second: int = -1) -> list[int]:
    return [first, second, -1, -1, -1, -1, -1, -1, -1, -1]


def _weights(first: float, second: float = 0.0) -> list[float]:
    return [first, second, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _ObservedSequence:
    def __init__(self, values) -> None:
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.values)


class RoutingBridgeTests(unittest.TestCase):
    def test_reuses_frozen_top10_and_separates_all_three_id_spaces(self) -> None:
        module = _routing_module()
        frozen = RouteBatch.freeze([_row(9, 300)], [_weights(0.6, 0.4)])

        plan = module.plan_routes(frozen, resident_logical_ids=(300, 9))

        self.assertIs(plan.routes, frozen)
        self.assertEqual(plan.npu_routes[0].logical_expert_id, 9)
        self.assertEqual(plan.npu_routes[0].npu_slot, 1)
        self.assertIsNone(plan.npu_routes[0].cpu_logical_expert_id)
        self.assertEqual(plan.npu_routes[1].logical_expert_id, 300)
        self.assertEqual(plan.npu_routes[1].npu_slot, 0)
        self.assertEqual(plan.cpu_routes, ())

    def test_q0_q32_and_all_resident_split_without_padding_routes(self) -> None:
        module = _routing_module()
        ids = [_row(4, 40), _row(7, 400)]
        weights = [_weights(0.25, 0.75), _weights(0.5, 0.5)]

        q0 = module.plan_routes(ids, weights, resident_logical_ids=())
        q32 = module.plan_routes(ids, weights, resident_logical_ids=tuple(range(32)))
        qall = module.plan_routes(ids, weights, resident_logical_ids=tuple(range(512)))

        self.assertEqual((len(q0.npu_routes), len(q0.cpu_routes)), (0, 4))
        self.assertEqual((len(q32.npu_routes), len(q32.cpu_routes)), (2, 2))
        self.assertEqual((len(qall.npu_routes), len(qall.cpu_routes)), (4, 0))
        self.assertEqual(
            [route.cpu_logical_expert_id for route in q32.cpu_routes],
            [40, 400],
        )
        self.assertTrue(all(route.logical_expert_id != -1 for route in q0.cpu_routes))

    def test_merge_applies_route_weights_shared_and_scaling_exactly_once(self) -> None:
        module = _routing_module()
        plan = module.plan_routes(
            [_row(1, 2)],
            [_weights(0.25, 0.75)],
            resident_logical_ids=(1,),
        )
        npu = {plan.npu_routes[0].key: (4.0, 8.0)}
        cpu = {plan.cpu_routes[0].key: (8.0, 4.0)}

        merged = module.merge_route_outputs(
            plan,
            npu,
            cpu,
            shared_output=((1.0, 1.0),),
            scaling=0.5,
        )

        self.assertEqual(merged, ((4.5, 3.5),))

    def test_merge_rejects_an_unfrozen_combine_policy(self) -> None:
        module = _routing_module()
        plan = module.plan_routes(
            [_row(1)],
            [_weights(1.0)],
            resident_logical_ids=(1,),
        )

        with self.assertRaisesRegex(ValueError, "combine policy"):
            module.merge_route_outputs(
                plan,
                {plan.npu_routes[0].key: (1.0,)},
                {},
                combine_policy="scale_everything",
            )

    def test_empty_shared_output_still_validates_row_count(self) -> None:
        module = _routing_module()
        plan = module.plan_routes(
            [_row(1)],
            [_weights(1.0)],
            resident_logical_ids=(),
            valid_rows=(False,),
        )

        with self.assertRaisesRegex(ValueError, "shared output row count"):
            module.merge_route_outputs(plan, {}, {}, shared_output=[])

        empty = module.plan_routes([], [], resident_logical_ids=())
        self.assertEqual(
            module.merge_route_outputs(empty, {}, {}, shared_output=[]),
            (),
        )

    def test_merge_rejects_non_finite_routed_scaling_factor(self) -> None:
        module = _routing_module()
        plan = module.plan_routes(
            [_row(1)],
            [_weights(1.0)],
            resident_logical_ids=(1,),
        )
        output = {plan.npu_routes[0].key: (2.0,)}

        for scaling in (math.nan, math.inf, -math.inf):
            with self.subTest(scaling=scaling):
                with self.assertRaisesRegex(ValueError, "finite"):
                    module.merge_route_outputs(plan, output, {}, scaling=scaling)

    def test_rejects_invalid_index_and_skips_invalid_rows(self) -> None:
        module = _routing_module()
        with self.assertRaisesRegex(ValueError, "logical expert"):
            module.plan_routes([_row(512)], [_weights(1.0)], resident_logical_ids=())
        with self.assertRaisesRegex(ValueError, "padding sentinel"):
            module.plan_routes([_row(-2)], [_weights(1.0)], resident_logical_ids=())

        plan = module.plan_routes(
            [_row(3), _row(3)],
            [_weights(1.0), _weights(1.0)],
            resident_logical_ids=(),
            valid_rows=(False, True),
        )
        self.assertEqual([route.token_index for route in plan.cpu_routes], [1])

    def test_same_expert_on_multiple_tokens_has_distinct_occurrence_keys(self) -> None:
        module = _routing_module()
        plan = module.plan_routes(
            [_row(40), _row(40), _row(40)],
            [_weights(1.0), _weights(1.0), _weights(1.0)],
            resident_logical_ids=(),
        )

        self.assertEqual(len({route.key for route in plan.cpu_routes}), 3)

    def test_resident_ids_reject_coercible_non_integers_before_route_inputs(self) -> None:
        module = _routing_module()

        for invalid_id in (0.9, False, "0"):
            with self.subTest(invalid_id=invalid_id):
                ids = _ObservedSequence([_row(1)])
                weights = _ObservedSequence([_weights(1.0)])
                with self.assertRaisesRegex(ValueError, "resident.*integers"):
                    module.plan_routes(
                        ids,
                        weights,
                        resident_logical_ids=(invalid_id,),
                    )
                self.assertEqual(ids.iterations, 0)
                self.assertEqual(weights.iterations, 0)

    def test_raw_route_ids_reject_coercible_non_integers_before_weights(self) -> None:
        module = _routing_module()

        for invalid_id in (0.9, True, "0"):
            with self.subTest(invalid_id=invalid_id):
                weights = _ObservedSequence([_weights(1.0)])
                with self.assertRaisesRegex(ValueError, "route logical expert ids.*integers"):
                    module.plan_routes(
                        [_row(invalid_id)],
                        weights,
                        resident_logical_ids=(),
                    )
                self.assertEqual(weights.iterations, 0)

    def test_frozen_route_and_logical_count_identities_are_strict_integers(self) -> None:
        module = _routing_module()
        frozen_weights = (tuple(_weights(1.0)),)

        for invalid_id in (0.9, True):
            with self.subTest(invalid_id=invalid_id):
                # contract-v2 validates immutable RouteBatch contents at the
                # shared boundary, before the routing bridge can observe them.
                with self.assertRaisesRegex(TypeError, "expert ids must be integers"):
                    RouteBatch(
                        (tuple(_row(invalid_id)),),
                        frozen_weights,
                        (True,),
                    )

        for logical_count in (512.0, True, "512"):
            with self.subTest(logical_count=logical_count):
                with self.assertRaisesRegex(ValueError, "logical_expert_count.*integer"):
                    module.plan_routes(
                        [_row(1)],
                        [_weights(1.0)],
                        resident_logical_ids=(),
                        logical_expert_count=logical_count,
                    )


if __name__ == "__main__":
    unittest.main()
