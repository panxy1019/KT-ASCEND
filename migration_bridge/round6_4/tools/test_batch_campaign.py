import unittest
from run_batch_campaign import stages


class CampaignTests(unittest.TestCase):
    def test_pilots_precede_long_runs(self):
        rows = list(stages())
        self.assertEqual(len(rows), 16)
        self.assertEqual([r[0] for r in rows[:4]], [2, 2, 4, 4])
        self.assertTrue(all(r[2] == "pilot" and r[3] == 1 for r in rows[:4]))
        for batch in (2, 4):
            for profile in ("D1", "D1_NODEFER"):
                chosen = [r for r in rows[4:] if r[:2] == (batch, profile)]
                self.assertEqual([r[2] for r in chosen], ["repeat", "restart1", "restart2"])
                self.assertEqual(chosen[0][3], 10)
                self.assertTrue(all(r[4] == "repeat" for r in chosen[1:]))


if __name__ == "__main__":
    unittest.main()
