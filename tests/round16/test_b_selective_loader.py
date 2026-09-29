from __future__ import annotations

import importlib
import unittest


def _loader_module():
    try:
        return importlib.import_module("round16.b.selective_loader")
    except ModuleNotFoundError as error:
        raise AssertionError("round16.b.selective_loader must exist") from error


class _Bank:
    def __init__(self, rows, shape, *, on_read=None) -> None:
        self._rows = rows
        self.shape = shape
        self._on_read = on_read
        self.reads: list[tuple[int, ...]] = []

    def read_experts(self, expert_ids):
        selected = tuple(expert_ids)
        self.reads.append(selected)
        if self._on_read is not None:
            self._on_read()
        return tuple(self._rows[index] for index in selected)

    def __iter__(self):
        raise AssertionError("a selective loader must never iterate the full bank")


class _Allocator:
    def __init__(
        self,
        fail_on_write: str | None = None,
        fail_on_release: frozenset[str] = frozenset(),
    ) -> None:
        self.fail_on_write = fail_on_write
        self.fail_on_release = fail_on_release
        self.allocations: list[dict] = []
        self.released: list[dict] = []
        self.release_calls: list[str] = []
        self.write_calls: list[str] = []

    def allocate(self, name, shape):
        allocation = {"name": name, "shape": tuple(shape), "data": None}
        self.allocations.append(allocation)
        return allocation

    def write(self, allocation, data):
        self.write_calls.append(allocation["name"])
        if allocation["name"] == self.fail_on_write:
            raise RuntimeError("injected write failure")
        allocation["data"] = data

    def release(self, allocation):
        self.release_calls.append(allocation["name"])
        if allocation["name"] in self.fail_on_release:
            raise RuntimeError(f"injected release failure: {allocation['name']}")
        self.released.append(allocation)


class _SameObjectAllocator(_Allocator):
    def __init__(self) -> None:
        super().__init__()
        self.shared = {"name": "shared", "shape": None, "data": None}

    def allocate(self, name, shape):
        self.shared["shape"] = tuple(shape)
        self.allocations.append(self.shared)
        return self.shared


class SelectiveLoaderTests(unittest.TestCase):
    def test_q32_plan_precedes_allocation_and_keeps_512_logical_ids(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))

        self.assertEqual(plan.logical_expert_count, 512)
        self.assertEqual(plan.physical_expert_count, 32)
        self.assertEqual(len(plan.logical_to_physical), 512)
        self.assertEqual(plan.logical_to_physical[31], 31)
        self.assertEqual(plan.logical_to_physical[32], -1)
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            module.plan_q32(tuple(range(512)))

    def test_resident_ids_reject_coercible_non_integers_before_allocation(self) -> None:
        module = _loader_module()
        source = _Bank([(index,) for index in range(512)], (512, 1))

        for invalid_id in (0.9, False, "0"):
            with self.subTest(invalid_id=invalid_id):
                resident = list(range(32))
                resident[0] = invalid_id
                allocator = _Allocator()
                with self.assertRaisesRegex(ValueError, "resident.*integers"):
                    plan = module.plan_q32(resident)
                    module.SelectiveExpertLoader(allocator).load(
                        plan,
                        {"first": source},
                    )
                self.assertEqual(allocator.allocations, [])

    def test_source_shape_rejects_coercible_non_integers_before_allocation(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))

        for shape in ((512.0, 1), (512, 1.9), (512, True), (512, "1")):
            with self.subTest(shape=shape):
                allocator = _Allocator()
                source = _Bank([(index,) for index in range(512)], shape)
                with self.assertRaisesRegex(ValueError, "shape.*integers"):
                    module.SelectiveExpertLoader(allocator).load(
                        plan,
                        {"first": source},
                    )
                self.assertEqual(allocator.allocations, [])

    def test_loads_only_selected_rows_fuses_gate_up_and_preserves_aliases(self) -> None:
        module = _loader_module()
        selected = tuple(range(0, 64, 2))
        plan = module.plan_q32(selected)
        gate = _Bank([(index, index + 0.5) for index in range(512)], (512, 2))
        up = _Bank([(index + 1000,) for index in range(512)], (512, 1))
        down = _Bank([(index + 2000,) for index in range(512)], (512, 1))
        allocator = _Allocator()

        loaded = module.SelectiveExpertLoader(allocator).load(
            plan,
            {"gate": gate, "up": up, "down": down, "down_alias": down},
            fused_gate_up=("gate", "up", "w13"),
            aliases={"w13_alias": "w13"},
        )

        self.assertEqual([item["shape"][0] for item in allocator.allocations], [32, 32])
        self.assertEqual(gate.reads, [selected])
        self.assertEqual(up.reads, [selected])
        self.assertEqual(down.reads, [selected])
        self.assertEqual(loaded["w13"]["data"][0], (0, 0.5, 1000))
        self.assertIs(loaded["w13_alias"], loaded["w13"])
        self.assertIs(loaded["down_alias"], loaded["down"])

    def test_exception_rolls_back_every_unique_allocation(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))
        first = _Bank([(index,) for index in range(512)], (512, 1))
        second = _Bank([(index,) for index in range(512)], (512, 1))
        allocator = _Allocator(fail_on_write="second")

        with self.assertRaisesRegex(RuntimeError, "injected"):
            module.SelectiveExpertLoader(allocator).load(
                plan,
                {"first": first, "second": second},
            )

        self.assertEqual(
            {allocation["name"] for allocation in allocator.released},
            {"first", "second"},
        )
        self.assertTrue(all(allocation["shape"][0] == 32 for allocation in allocator.allocations))

    def test_fused_gate_up_source_aliases_fail_before_allocation(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))
        gate = _Bank([(index,) for index in range(512)], (512, 1))
        up = _Bank([(index + 1000,) for index in range(512)], (512, 1))

        cases = (
            {"gate": gate, "up": gate},
            {"gate": gate, "gate_alias": gate, "up": up},
        )
        for sources in cases:
            with self.subTest(names=tuple(sources)):
                allocator = _Allocator()
                with self.assertRaisesRegex(ValueError, "fused.*alias"):
                    module.SelectiveExpertLoader(allocator).load(
                        plan,
                        sources,
                        fused_gate_up=("gate", "up", "w13"),
                    )
                self.assertEqual(allocator.allocations, [])

    def test_name_ownership_conflicts_fail_before_allocation(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))
        gate = _Bank([(index,) for index in range(512)], (512, 1))
        up = _Bank([(index + 1000,) for index in range(512)], (512, 1))
        down = _Bank([(index + 2000,) for index in range(512)], (512, 1))
        first = _Bank([(index + 3000,) for index in range(512)], (512, 1))
        second = _Bank([(index + 4000,) for index in range(512)], (512, 1))

        cases = (
            (
                {"gate": gate, "up": up, "down": down},
                ("gate", "up", "down"),
                None,
            ),
            (
                {"first": first, "second": second},
                None,
                {"first": "second"},
            ),
        )
        for sources, fused_gate_up, aliases in cases:
            with self.subTest(sources=tuple(sources), aliases=aliases):
                allocator = _Allocator()
                with self.assertRaisesRegex(ValueError, "name ownership conflict"):
                    module.SelectiveExpertLoader(allocator).load(
                        plan,
                        sources,
                        fused_gate_up=fused_gate_up,
                        aliases=aliases,
                    )
                self.assertEqual(allocator.allocations, [])

    def test_rollback_attempts_every_release_and_preserves_primary_error(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))
        sources = {
            name: _Bank([(index,) for index in range(512)], (512, 1))
            for name in ("first", "second", "third")
        }
        allocator = _Allocator(
            fail_on_write="third",
            fail_on_release=frozenset(("first", "third")),
        )

        with self.assertRaisesRegex(RuntimeError, "injected write failure") as caught:
            module.SelectiveExpertLoader(allocator).load(plan, sources)

        self.assertEqual(allocator.release_calls, ["third", "second", "first"])
        cleanup = caught.exception.__cause__
        self.assertIsInstance(cleanup, module.RollbackCleanupError)
        self.assertEqual(len(cleanup.errors), 2)
        self.assertEqual(
            {str(error) for error in cleanup.errors},
            {
                "injected release failure: first",
                "injected release failure: third",
            },
        )

    def test_manual_selection_plan_is_validated_before_allocation(self) -> None:
        module = _loader_module()
        resident = tuple(range(32))
        valid_mapping = tuple(range(32)) + (-1,) * 480
        source = _Bank([(index,) for index in range(512)], (512, 1))

        duplicate_physical = list(valid_mapping)
        duplicate_physical[0] = 1
        illegal_physical = list(valid_mapping)
        illegal_physical[40] = 32
        nonresident_mapping = list(valid_mapping)
        nonresident_mapping[40] = 0
        cases = (
            (resident, valid_mapping[:-1]),
            (resident, tuple(duplicate_physical)),
            (resident, tuple(illegal_physical)),
            (resident, tuple(nonresident_mapping)),
            ((0,) * 32, valid_mapping),
        )

        for bad_resident, bad_mapping in cases:
            with self.subTest(resident=bad_resident, mapping=bad_mapping):
                allocator = _Allocator()
                with self.assertRaisesRegex(ValueError, "selection plan"):
                    plan = module.SelectiveLoadPlan(bad_resident, bad_mapping)
                    module.SelectiveExpertLoader(allocator).load(
                        plan,
                        {"first": source},
                    )
                self.assertEqual(allocator.allocations, [])

    def test_plan_deep_freezes_caller_lists_before_source_reads(self) -> None:
        module = _loader_module()
        resident = list(range(32))
        mapping = list(range(32)) + [-1] * 480
        selected = tuple(resident)
        plan = module.SelectiveLoadPlan(resident, mapping)

        def mutate_plan_inputs():
            resident.reverse()
            mapping[:] = [-1] * 512

        first = _Bank(
            [(index,) for index in range(512)],
            (512, 1),
            on_read=mutate_plan_inputs,
        )
        second = _Bank([(index + 1000,) for index in range(512)], (512, 1))

        module.SelectiveExpertLoader(_Allocator()).load(
            plan,
            {"first": first, "second": second},
        )

        self.assertIs(type(plan.resident_logical_ids), tuple)
        self.assertIs(type(plan.logical_to_physical), tuple)
        self.assertEqual(first.reads, [selected])
        self.assertEqual(second.reads, [selected])

    def test_load_snapshots_source_mapping_before_any_source_read(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))
        original_second = _Bank(
            [(index + 1000,) for index in range(512)],
            (512, 1),
        )
        replacement_second = _Bank(
            [(index + 9000,) for index in range(512)],
            (512, 1),
        )
        sources = {}

        def replace_second_source():
            sources["second"] = replacement_second

        first = _Bank(
            [(index,) for index in range(512)],
            (512, 1),
            on_read=replace_second_source,
        )
        sources.update(first=first, second=original_second)

        loaded = module.SelectiveExpertLoader(_Allocator()).load(plan, sources)

        self.assertEqual(original_second.reads, [tuple(range(32))])
        self.assertEqual(replacement_second.reads, [])
        self.assertEqual(loaded["second"]["data"][0], (1000,))

    def test_allocator_object_reuse_fails_before_second_write_and_releases_once(self) -> None:
        module = _loader_module()
        plan = module.plan_q32(tuple(range(32)))
        sources = {
            name: _Bank([(index,) for index in range(512)], (512, 1))
            for name in ("first", "second")
        }
        allocator = _SameObjectAllocator()

        with self.assertRaisesRegex(RuntimeError, "allocator ownership conflict"):
            module.SelectiveExpertLoader(allocator).load(plan, sources)

        self.assertEqual(len(allocator.allocations), 2)
        self.assertEqual(allocator.write_calls, ["shared"])
        self.assertEqual(allocator.release_calls, ["shared"])


if __name__ == "__main__":
    unittest.main()
