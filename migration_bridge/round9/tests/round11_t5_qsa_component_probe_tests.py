import unittest

import torch

from artifacts.round11.round11_t5_qsa_component_probe import (
    make_causal_selection,
    take_query_half,
)


class Round11T5QSAComponentProbeTests(unittest.TestCase):
    def test_take_query_half_does_not_return_unused_gate_view(self):
        projected = torch.arange(2 * 3 * 8).reshape(2, 3 * 8)
        query = take_query_half(projected, num_heads=3, head_dim=4)
        expected = projected.reshape(2, 3, 8)[..., :4]
        self.assertTrue(torch.equal(query, expected))
        self.assertEqual(tuple(query.shape), (2, 3, 4))

    def test_causal_selection_packs_each_active_prefix(self):
        selected = make_causal_selection(
            positions=torch.tensor([32, 33, 34]),
            context_tokens=35,
            width=40,
        )
        self.assertEqual(tuple(selected.shape), (3, 40))
        for row, position in zip(selected, (32, 33, 34)):
            self.assertEqual(row[: position + 1].tolist(), list(range(position + 1)))
            self.assertEqual(row[position + 1 :].tolist(), [-1] * (39 - position))


if __name__ == "__main__":
    unittest.main()
