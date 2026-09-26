import ast
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from migration_bridge.round9.runtime.nd_parent_update import update_parent


class FakeTensor:
    def __init__(
        self,
        rows,
        *,
        fmt=2,
        contiguous=True,
        pointer=None,
        stride=None,
        shape=None,
        base=None,
    ):
        self.rows = list(rows)
        self.shape = (len(self.rows),) if shape is None else tuple(shape)
        self.fmt = fmt
        self._base = base
        self._contiguous = contiguous
        self._pointer = id(self) if pointer is None else pointer
        self._stride = (1,) if stride is None else tuple(stride)
        self.index_copy_calls = []
        self.copy_calls = []
        self.clone_calls = 0

    def index_copy_(self, dim, index, staged):
        self.index_copy_calls.append((dim, index, staged))
        return self

    def reshape(self, *shape):
        return FakeIndex2D(self.rows, shape, fmt=self.fmt)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return FakeTensor(self.rows, fmt=self.fmt, contiguous=True)

    def clone(self):
        self.clone_calls += 1
        return FakeTensor(self.rows, fmt=self.fmt, contiguous=True)

    def data_ptr(self):
        return self._pointer

    def stride(self):
        return self._stride

    def copy_(self, source):
        self.copy_calls.append(source)
        self.rows[:] = source.rows
        return self


class PartialCopyFailureTensor(FakeTensor):
    def copy_(self, source):
        self.copy_calls.append(source)
        self.rows[0] = source.rows[0]
        raise RuntimeError("partial copy")


class FakeIndex2D:
    def __init__(self, rows, shape, *, fmt):
        self.rows = list(rows)
        self.shape = shape
        self.fmt = fmt


class FakeProvider:
    def __init__(self, *, cast_alias=False, cast_shape=None, cast_view=False):
        self.cast_alias = cast_alias
        self.cast_shape = cast_shape
        self.cast_view = cast_view
        self.cast_calls = []
        self.scatter_calls = []
        self.cast_result = None

    @staticmethod
    def get_format(tensor):
        return tensor.fmt

    def format_cast(self, tensor, fmt):
        self.cast_calls.append((tensor, fmt))
        if self.cast_alias:
            self.cast_result = FakeTensor(
                tensor.rows,
                fmt=fmt,
                contiguous=True,
                pointer=tensor.data_ptr(),
                shape=self.cast_shape,
                base=object() if self.cast_view else None,
            )
        else:
            self.cast_result = FakeTensor(
                tensor.rows,
                fmt=fmt,
                contiguous=True,
                shape=self.cast_shape,
                base=object() if self.cast_view else None,
            )
        return self.cast_result

    def scatter(self, destination, indices, updates):
        self.scatter_calls.append((destination, indices, updates))
        for slot, value in zip(indices.rows, updates.rows):
            destination.rows[slot] = value
        return destination


def enabled_update(destination, index, staged, slots, provider, **kwargs):
    return update_parent(
        destination,
        index,
        staged,
        enabled=True,
        private_shadow_only=True,
        host_slots=slots,
        slot_count=len(destination.rows),
        update_count=len(slots),
        get_format=provider.get_format,
        format_cast=provider.format_cast,
        scatter_nd_update=provider.scatter,
        **kwargs,
    )


class NdParentUpdateTests(unittest.TestCase):
    def test_dynamic_adapter_wires_only_private_candidate_at_nz_batch_call(self):
        adapter_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "migration_bridge/round9/runtime/dynamic_adapter.py"
        )
        tree = ast.parse(adapter_path.read_text())
        copy_experts = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "copy_experts"
        )
        candidate_if = next(
            node
            for node in ast.walk(copy_experts)
            if isinstance(node, ast.If)
            and "_round9_private_parent_update" in ast.unparse(node.test)
        )

        self.assertEqual(
            ast.unparse(candidate_if.test),
            "getattr(self, '_round9_private_parent_update', False) is True",
        )
        on_source = "\n".join(ast.unparse(node) for node in candidate_if.body)
        self.assertIn("update_parent(destination, index, staged", on_source)
        self.assertIn("enabled=True", on_source)
        self.assertIn("private_shadow_only=True", on_source)
        self.assertIn("host_slots=tuple((slot for slot, _ in moves))", on_source)
        self.assertIn("slot_count=destination.shape[0]", on_source)
        self.assertIn("update_count=len(moves)", on_source)
        self.assertIn(
            "host_phase_costs=getattr(self, '_round9_parent_update_phase_costs', None)",
            on_source,
        )
        self.assertEqual(
            [ast.unparse(node) for node in candidate_if.orelse],
            ["destination.index_copy_(0, index, staged)"],
        )

    def test_default_off_is_exact_original_index_copy(self):
        class OriginalParent:
            def __init__(self):
                self.calls = []

            def index_copy_(self, dim, index, staged):
                self.calls.append((dim, index, staged))
                return self

            def __getattr__(self, name):
                raise AssertionError("OFF touched candidate state: " + name)

        destination = OriginalParent()
        index = object()
        staged = object()
        result = update_parent(destination, index, staged)
        self.assertIs(result, destination)
        self.assertEqual(destination.calls, [(0, index, staged)])

    def test_on_updates_requested_slots_and_preserves_unused_slots(self):
        destination = FakeTensor([0, 1, 2, 3], fmt=29, stride=(4, 1))
        index = FakeTensor([3, 1], fmt=2, contiguous=False)
        staged = FakeTensor([30, 10], fmt=2)
        provider = FakeProvider()

        result = enabled_update(destination, index, staged, (3, 1), provider)

        self.assertIs(result, destination)
        self.assertEqual(destination.rows, [0, 10, 2, 30])
        self.assertEqual(provider.scatter_calls[0][1].shape, (-1, 1))
        self.assertEqual(destination.index_copy_calls, [])

    def test_noncontiguous_index_is_only_reshaped_for_scatter(self):
        destination = FakeTensor([0, 1, 2, 3], fmt=29)
        index = FakeTensor([2, 0], fmt=2, contiguous=False)
        staged = FakeTensor([20, 0], fmt=2)
        provider = FakeProvider()

        enabled_update(destination, index, staged, (2, 0), provider)

        reshaped = provider.scatter_calls[0][1]
        self.assertEqual(reshaped.rows, [2, 0])
        self.assertEqual(reshaped.shape, (-1, 1))

    def test_cast_alias_is_cloned_before_scatter(self):
        destination = FakeTensor([0, 1, 2], fmt=29)
        provider = FakeProvider(cast_alias=True)

        enabled_update(
            destination,
            FakeTensor([1], fmt=2),
            FakeTensor([10], fmt=2),
            (1,),
            provider,
        )

        scatter_destination = provider.scatter_calls[0][0]
        self.assertIsNot(scatter_destination, destination)
        self.assertNotEqual(scatter_destination.data_ptr(), destination.data_ptr())
        self.assertEqual(provider.cast_result.clone_calls, 1)

    def test_cast_view_is_cloned_to_an_owning_nd_parent(self):
        destination = FakeTensor([0, 1, 2], fmt=29)
        provider = FakeProvider(cast_view=True)

        enabled_update(
            destination,
            FakeTensor([1], fmt=2),
            FakeTensor([10], fmt=2),
            (1,),
            provider,
        )

        self.assertEqual(provider.cast_result.clone_calls, 1)
        self.assertIsNone(provider.scatter_calls[0][0]._base)

    def test_cast_shape_mismatch_fails_before_scatter_and_copy(self):
        destination = FakeTensor([0, 1, 2], fmt=29)
        provider = FakeProvider(cast_shape=(2,))

        with self.assertRaisesRegex(RuntimeError, "shape"):
            enabled_update(
                destination,
                FakeTensor([1], fmt=2),
                FakeTensor([10], fmt=2),
                (1,),
                provider,
            )

        self.assertEqual(provider.scatter_calls, [])
        self.assertEqual(destination.copy_calls, [])

    def test_wrong_tensor_format_fails_before_cast_or_scatter(self):
        provider = FakeProvider()
        destination = FakeTensor([0, 1], fmt=2)

        with self.assertRaisesRegex(RuntimeError, "NZ29 destination"):
            enabled_update(
                destination,
                FakeTensor([1], fmt=2),
                FakeTensor([10], fmt=2),
                (1,),
                provider,
            )

        self.assertEqual(provider.cast_calls, [])
        self.assertEqual(provider.scatter_calls, [])

    def test_host_slots_rejected_before_tensor_or_provider_access(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError("validation touched tensor: " + name)

        provider = FakeProvider()
        invalid = ((1, 1), (4,), (1, 2))
        counts = ((4, 2), (4, 1), (4, 1))
        for slots, (slot_count, update_count) in zip(invalid, counts):
            with self.subTest(slots=slots):
                with self.assertRaises(ValueError):
                    update_parent(
                        Untouchable(),
                        Untouchable(),
                        Untouchable(),
                        enabled=True,
                        private_shadow_only=True,
                        host_slots=slots,
                        slot_count=slot_count,
                        update_count=update_count,
                        get_format=lambda _: self.fail("format lookup before slots validation"),
                        format_cast=provider.format_cast,
                        scatter_nd_update=provider.scatter,
                    )

    def test_on_requires_explicit_private_shadow_permission(self):
        with self.assertRaisesRegex(RuntimeError, "private shadow"):
            update_parent(
                FakeTensor([0], fmt=29),
                FakeTensor([0], fmt=2),
                FakeTensor([1], fmt=2),
                enabled=True,
                private_shadow_only=False,
                host_slots=(0,),
                slot_count=1,
                update_count=1,
            )

    def test_partial_copy_exception_propagates_without_fallback(self):
        destination = PartialCopyFailureTensor([0, 1], fmt=29)
        provider = FakeProvider()

        with self.assertRaisesRegex(RuntimeError, "partial copy"):
            enabled_update(
                destination,
                FakeTensor([1], fmt=2),
                FakeTensor([10], fmt=2),
                (1,),
                provider,
            )

        self.assertEqual(destination.index_copy_calls, [])
        self.assertEqual(len(provider.scatter_calls), 1)

    def test_object_pointer_stride_and_format_are_preserved(self):
        destination = FakeTensor(
            [0, 1, 2], fmt=29, pointer=123456, stride=(99, 1, 9)
        )
        pointer = destination.data_ptr()
        stride = destination.stride()
        provider = FakeProvider()

        result = enabled_update(
            destination,
            FakeTensor([2], fmt=2),
            FakeTensor([20], fmt=2),
            (2,),
            provider,
        )

        self.assertIs(result, destination)
        self.assertEqual(destination.data_ptr(), pointer)
        self.assertEqual(destination.stride(), stride)
        self.assertEqual(destination.fmt, 29)

    def test_host_phase_costs_do_not_require_a_sync(self):
        destination = FakeTensor([0, 1], fmt=29)
        provider = FakeProvider()
        costs = {}
        ticks = iter(range(20))

        enabled_update(
            destination,
            FakeTensor([1], fmt=2),
            FakeTensor([10], fmt=2),
            (1,),
            provider,
            host_phase_costs=costs,
            clock_ns=lambda: next(ticks),
        )

        self.assertEqual(
            set(costs),
            {"validate_host_ns", "format_cast_host_ns", "scatter_host_ns", "copy_host_ns"},
        )
        self.assertTrue(all(value == 1 for value in costs.values()))


if __name__ == "__main__":
    unittest.main()
