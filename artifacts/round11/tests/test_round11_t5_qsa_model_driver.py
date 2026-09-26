import unittest

from artifacts.round11.round11_t5_qsa_model_driver import (
    build_schedule,
    validate_output,
)


class Round11T5QSAModelDriverTests(unittest.TestCase):
    def setUp(self):
        self.fixture = {
            "cases": {
                "edge128": {"input_ids": list(range(128))},
                "canary_math391": {"input_ids": list(range(27))},
            }
        }

    def test_schedule_is_bounded_four_generation_unprofiled_block(self):
        schedule = build_schedule(self.fixture)
        self.assertEqual(len(schedule), 4)
        self.assertEqual(
            [(row["case_id"], row["phase"], row["decode_tokens"]) for row in schedule],
            [
                ("canary_math391", "warmup", 4),
                ("edge128", "measurement", 32),
                ("edge128", "measurement", 64),
                ("canary_math391", "drain", 4),
            ],
        )
        self.assertTrue(all(row["request"]["stream"] for row in schedule))

    def test_quality_is_bound_to_frozen_output_ids(self):
        schedule = build_schedule(self.fixture)
        self.assertTrue(validate_output(schedule[0], [18, 24, 16, 248046]))
        self.assertTrue(validate_output(schedule[1], [3897, 248046] + [1] * 30))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            validate_output(schedule[0], [18, 24, 15, 248046])
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            validate_output(schedule[1], [3897, 1] + [1] * 30)


if __name__ == "__main__":
    unittest.main()
