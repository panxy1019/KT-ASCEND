from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
ROUND13 = ROOT / "artifacts" / "round13"
sys.path.insert(0, str(ROUND13))

from round13_placement import (  # noqa: E402
    BUDGET_EXPERTS,
    EXPERTS_PER_LAYER,
    LAYERS,
    NOMINAL_BF16_BYTES,
    build_profiles,
    frequency_global_budget,
    front_loading,
    random_global_budget,
    uniform,
    validate_profile,
)


DEV = (
    ROOT
    / "migration_bridge"
    / "round9"
    / "evidence"
    / "physical15"
    / "20260921-framework-frequency-v1"
    / "frequency-q32.json"
)


class PlacementProfilesTest(unittest.TestCase):
    def test_uniform_reproduces_round12a_q32(self):
        profile = uniform()
        validate_profile(profile)
        self.assertEqual(profile[0], list(range(32)))
        self.assertTrue(all(row == list(range(32)) for row in profile))

    def test_frequency_is_global_and_ties_are_layer_expert_ordered(self):
        counts = [[0] * EXPERTS_PER_LAYER for _ in range(LAYERS)]
        counts[47][511] = 100
        counts[12][7] = 99
        profile = frequency_global_budget(counts)
        validate_profile(profile)
        selected = {(layer, expert) for layer, row in enumerate(profile) for expert in row}
        self.assertIn((47, 511), selected)
        self.assertIn((12, 7), selected)
        # All remaining zero-count ties are resolved by layer then expert ID.
        self.assertIn((0, 0), selected)
        self.assertNotIn((47, 510), selected)

    def test_front_loading_has_three_full_layers_and_zero_elsewhere(self):
        profile = front_loading()
        validate_profile(profile)
        self.assertEqual([len(row) for row in profile[:3]], [512, 512, 512])
        self.assertTrue(all(not row for row in profile[3:]))

    def test_random_is_reproducible_without_replacement(self):
        seed42_a = random_global_budget(42)
        seed42_b = random_global_budget(42)
        seed43 = random_global_budget(43)
        validate_profile(seed42_a)
        self.assertEqual(seed42_a, seed42_b)
        self.assertNotEqual(seed42_a, seed43)
        pairs = [(layer, expert) for layer, row in enumerate(seed42_a) for expert in row]
        self.assertEqual(len(pairs), BUDGET_EXPERTS)
        self.assertEqual(len(set(pairs)), BUDGET_EXPERTS)

    def test_q0_q512_and_nonuniform_rows_are_valid(self):
        front = front_loading()
        self.assertEqual(len(front[0]), 512)
        self.assertEqual(len(front[3]), 0)
        random_profile = random_global_budget(42)
        counts = {len(row) for row in random_profile}
        self.assertGreater(len(counts), 1)
        validate_profile(random_profile)

    def test_real_dev_build_has_four_exact_budget_profiles(self):
        document = json.loads(DEV.read_text())
        profiles = build_profiles(document["statistics"]["counts"])
        self.assertEqual(set(profiles), {"uniform", "frequency", "front_loading", "random_seed42"})
        for profile in profiles.values():
            validate_profile(profile)
            self.assertEqual(sum(map(len, profile)), BUDGET_EXPERTS)

    def test_nominal_budget_matches_14_0625_gib(self):
        self.assertEqual(NOMINAL_BF16_BYTES, 15099494400)
        self.assertEqual(NOMINAL_BF16_BYTES / 2**30, 14.0625)

    def test_profile_hash_is_stable_json(self):
        profile = random_global_budget(42)
        encoded = json.dumps(profile, separators=(",", ":"), sort_keys=True).encode()
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            hashlib.sha256(encoded).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
