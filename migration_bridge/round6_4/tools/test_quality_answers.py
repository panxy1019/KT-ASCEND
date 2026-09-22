import unittest
from quality_answers import parse, score


class AnswerTests(unittest.TestCase):
    def test_numeric_normalization(self):
        self.assertTrue(score("gsm8k", "reasoning 123\n#### 1,000.00", "1000")["correct"])
        self.assertTrue(score("gsm8k", "#### -2.50", "-2.5")["correct"])
        self.assertIsNone(parse("gsm8k", "The calculation gives 25"))
        self.assertIsNone(parse("gsm8k", "#### 1,00"))
        self.assertIsNone(parse("gsm8k", "#### 2+2"))

    def test_no_code_or_ambiguous_letters(self):
        self.assertEqual(parse("cmmlu", " Ａ。\n"), "A")
        for text in ("A or B", "A because ...", "print('A')", "", "E"):
            self.assertIsNone(parse("cmmlu", text))

    def test_truncated_counts_wrong_even_with_answer(self):
        result = score("gsm8k", "#### 42", "42", truncated=True)
        self.assertFalse(result["correct"])
        self.assertTrue(result["truncated"])


if __name__ == "__main__":
    unittest.main()
