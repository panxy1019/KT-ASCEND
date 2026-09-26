import unittest

from artifacts.round11.round11_t6_driver import build_schedule


class Round11T6DriverTests(unittest.TestCase):
    def setUp(self):
        names = {
            "canary_marker": 29,
            "canary_math391": 27,
            "canary_json": 28,
            "canary_quote": 31,
            "canary_transform": 25,
            "edge128": 128,
            "cancel128": 128,
        }
        self.fixture = {
            "cases": {
                name: {
                    "input_ids": list(range(length)),
                    "validator": {"kind": "exact_text", "expected": name},
                }
                for name, length in names.items()
            }
        }
        self.fixture["cases"]["edge128"]["validator"] = {
            "kind": "assistant_message_prefix",
            "expected_token_ids": [3897, 248046],
            "expected_text": "ACK",
        }
        self.fixture["cases"]["cancel128"]["validator"] = {"kind": "abort"}

    def test_every_block_has_same_three_performance_requests(self):
        for version, (mode, block) in enumerate(
            (("BASELINE", "A"), ("FINAL", "A"), ("FINAL", "B"), ("BASELINE", "B")),
            147,
        ):
            rows = build_schedule(self.fixture, mode=mode, block=block, version=version)
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                [(row["case_id"], row["phase"], row["decode_tokens"]) for row in rows[:4]],
                [
                    ("canary_math391", "warmup", None),
                    ("edge128", "performance", 32),
                    ("edge128", "performance", 64),
                    ("edge128", "performance", 128),
                ],
            )

    def test_exactly_two_blocks_have_cancel_and_reuse(self):
        schedules = [
            build_schedule(self.fixture, mode=mode, block=block, version=147 + index)
            for index, (mode, block) in enumerate(
                (("BASELINE", "A"), ("FINAL", "A"), ("FINAL", "B"), ("BASELINE", "B"))
            )
        ]
        self.assertEqual(sum(any(row["phase"] == "cancel" for row in rows) for rows in schedules), 2)
        self.assertEqual(sum(row["phase"] == "cancel" for rows in schedules for row in rows), 2)

    def test_all_five_canaries_are_covered_across_blocks(self):
        covered = set()
        for index, (mode, block) in enumerate(
            (("BASELINE", "A"), ("FINAL", "A"), ("FINAL", "B"), ("BASELINE", "B"))
        ):
            for row in build_schedule(self.fixture, mode=mode, block=block, version=147 + index):
                if row["case_id"].startswith("canary_"):
                    covered.add(row["case_id"])
        self.assertEqual(
            covered,
            {"canary_marker", "canary_math391", "canary_json", "canary_quote", "canary_transform"},
        )


if __name__ == "__main__":
    unittest.main()
