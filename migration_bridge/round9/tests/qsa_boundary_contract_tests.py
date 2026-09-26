"""Small QSA contract cases. Torch tests SKIP when unavailable; no device setup.

A local fixture check is not a numerical/provider or allocator qualification.
Run directly with Python; if torch is installed, its CPU thread count is one.
"""
import importlib.util
import pathlib
import sys
import unittest

LENGTHS = (1, 3, 4, 2047, 2048, 2049, 2051, 2052)


def expected_tokens(length):
    # Independent fixture: positive strictly increasing score per completed block.
    complete, tail = divmod(length, 4)
    blocks = range(complete - 1, max(-1, complete - 513), -1)
    return [4 * block + offset for block in blocks for offset in range(4)] + list(
        range(4 * complete, 4 * complete + tail)
    )


def physical_slots(length):
    # Reversed page order, preserving within-page alignment; slot 0 is reserved.
    pages = (length + 3) // 4
    return [4 * (pages - logical // 4) + logical % 4 for logical in range(length)]


class FixtureContractTests(unittest.TestCase):
    def test_pressure_starts_at_2052_visible_tokens(self):
        for length in LENGTHS:
            with self.subTest(length=length):
                ids = expected_tokens(length)
                self.assertEqual(len(ids), min(length // 4, 512) * 4 + length % 4)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(0 <= i < length for i in ids))
                self.assertEqual(set(range(length)) - set(ids),
                                 set(range(4)) if length == 2052 else set())

    def test_page_permutation_preserves_alignment_and_identity(self):
        for length in LENGTHS:
            slots = physical_slots(length)
            self.assertEqual(len(set(slots)), length)
            self.assertTrue(all(s > 0 for s in slots))
            self.assertTrue(all(slots[i] % 4 == i % 4 for i in range(length)))
            pool = {slot: ('K', logical) for logical, slot in enumerate(slots)}
            self.assertEqual([pool[slots[i]] for i in expected_tokens(length)],
                             [('K', i) for i in expected_tokens(length)])


HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    from runtime.qsa import select_tokens, sparse_gqa


@unittest.skipUnless(HAS_TORCH, 'torch unavailable: numeric provider gate NOT_RUN')
class ProviderContractTests(unittest.TestCase):
    def test_boundaries_and_future_high_score_blocks(self):
        # Future blocks have the highest scores, so omitting causal masking fails.
        keys = torch.arange(1, 517, dtype=torch.float32).reshape(-1, 1)
        for length in LENGTHS:
            with self.subTest(length=length):
                ids = select_tokens(torch.ones(1, 1, 1), keys,
                                    torch.tensor([length - 1]))
                self.assertEqual(tuple(ids.shape), (1, 2051))
                self.assertEqual(ids[0][ids[0] >= 0].tolist(), expected_tokens(length))

    def test_ties_require_valid_blocks_not_a_specific_tie_order(self):
        ids = select_tokens(torch.ones(1, 1, 1), torch.ones(514, 1),
                            torch.tensor([2051]))[0]
        chosen = ids[ids >= 0].tolist()
        self.assertEqual(len(chosen), 2048)
        self.assertEqual(len(set(chosen)), 2048)
        self.assertTrue(all(0 <= token < 2052 for token in chosen))
        blocks = {token // 4 for token in chosen}
        self.assertEqual(len(blocks), 512)
        self.assertEqual(set(chosen), {4 * b + o for b in blocks for o in range(4)})

    def test_page_permutation_and_independent_attention_oracle(self):
        length = 2052
        selected = torch.tensor([expected_tokens(length)])
        # Tiny head dimensions deliberately bound CPU work; not model-shape proof.
        keys = torch.arange(length * 2, dtype=torch.float32).reshape(length, 1, 2) / length
        values = keys.flip(0).clone()
        slots = torch.tensor(physical_slots(length))
        kp = torch.zeros(int(slots.max()) + 1, 1, 2)
        vp = torch.zeros_like(kp)
        kp[slots], vp[slots] = keys, values
        q = torch.tensor([[[0.25, -0.125]]])
        got = sparse_gqa(q, kp[slots], vp[slots], selected)
        k = keys[selected[0], 0].double()
        v = values[selected[0], 0].double()
        scores = (k * q[0, 0].double()).sum(-1) / (2 ** 0.5)
        want = (scores.softmax(0)[:, None] * v).sum(0)
        torch.testing.assert_close(got[0, 0].double(), want, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
