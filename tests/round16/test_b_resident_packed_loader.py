from __future__ import annotations

import unittest

from round16.b.resident_packed_loader import (
    load_selected_packed_rows,
    planned_row_reads,
    validate_packed_source_metadata,
)
from round16.b.selective_loader import plan_q32


class _Row:
    def __init__(self, label: str) -> None:
        self.label = label

    def __getitem__(self, item):
        return _Row(f"{self.label}[{item.start}:{item.stop}]")


class _Selected:
    def __init__(self, shape, name: str) -> None:
        self.shape = shape
        self.name = name

    def __getitem__(self, item):
        return _Row(f"{self.name}[{item}]")


class _Parameter:
    def __init__(self, reject=None) -> None:
        self.calls = []
        self.reject = reject

        def loader(**kwargs):
            self.calls.append(kwargs)
            return kwargs["shard_id"] != self.reject

        self.weight_loader = loader


class _Owner:
    def __init__(self, reject=None) -> None:
        self.layer_name = "model.layers.0.mlp.experts"
        self.w13_weight = _Parameter(reject)
        self.w2_weight = _Parameter(reject)


class ResidentPackedLoaderTests(unittest.TestCase):
    def test_noncontiguous_plan_keeps_global_ids_and_physical_order(self):
        resident = tuple(range(511, 479, -1))
        plan = plan_q32(resident)
        owner = _Owner()

        coverage = load_selected_packed_rows(
            owner=owner,
            plan=plan,
            selected_gate_up=_Selected((32, 1280, 2560), "gate_up"),
            selected_down=_Selected((32, 2560, 640), "down"),
            gate_up_weight_name="model.layers.0.mlp.experts.gate_up_proj",
            down_weight_name="model.layers.0.mlp.experts.down_proj",
        )

        self.assertEqual(len(coverage), 96)
        self.assertEqual(planned_row_reads(plan)[0], (0, 511))
        self.assertEqual(planned_row_reads(plan)[-1], (31, 480))
        all_calls = owner.w13_weight.calls + owner.w2_weight.calls
        self.assertEqual({call["expert_id"] for call in all_calls}, set(resident))
        self.assertTrue(all(call["return_success"] for call in all_calls))
        self.assertEqual(
            [call["shard_id"] for call in owner.w13_weight.calls[:2]],
            ["w1", "w3"],
        )
        self.assertTrue(
            all(
                call["weight_name"] == "model.layers.0.mlp.experts.w13_weight"
                for call in owner.w13_weight.calls
            )
        )
        self.assertTrue(
            all(
                call["weight_name"] == "model.layers.0.mlp.experts.w2_weight"
                for call in owner.w2_weight.calls
            )
        )

    def test_rejects_full512_tensor_at_selected_boundary(self):
        with self.assertRaisesRegex(ValueError, "exactly Q32"):
            load_selected_packed_rows(
                owner=_Owner(),
                plan=plan_q32(tuple(range(32))),
                selected_gate_up=_Selected((512, 1280, 2560), "gate_up"),
                selected_down=_Selected((32, 2560, 640), "down"),
                gate_up_weight_name="experts.gate_up_proj",
                down_weight_name="experts.down_proj",
            )

    def test_loader_rejection_is_not_counted_as_coverage(self):
        with self.assertRaisesRegex(RuntimeError, "projection up"):
            load_selected_packed_rows(
                owner=_Owner(reject="w3"),
                plan=plan_q32(tuple(range(32))),
                selected_gate_up=_Selected((32, 1280, 2560), "gate_up"),
                selected_down=_Selected((32, 2560, 640), "down"),
                gate_up_weight_name="experts.gate_up_proj",
                down_weight_name="experts.down_proj",
            )

    def test_packed_source_metadata_requires_original_logical512(self):
        validate_packed_source_metadata(
            (512, 1280, 2560),
            (512, 2560, 640),
        )
        with self.assertRaisesRegex(ValueError, "logical512"):
            validate_packed_source_metadata(
                (32, 1280, 2560),
                (512, 2560, 640),
            )


if __name__ == "__main__":
    unittest.main()
