import unittest
from quality_answers_v2 import parse, score


class QualityV2Tests(unittest.TestCase):
    def test_base_model_numeric_answer(self):
        self.assertEqual(parse("gsm8k","48 + 24 = 72\nShe sold 72 clips."),"72")
        self.assertEqual(parse("gsm8k","The answer is $1,234.50."),"1234.5")
        self.assertEqual(parse("gsm8k","2-4 = -2"),"-2")
        self.assertIsNone(parse("gsm8k","I cannot answer."))

    def test_explicit_marker_remains_strict(self):
        self.assertEqual(parse("gsm8k","calculate 15\n#### 20"),"2E+1")
        self.assertIsNone(parse("gsm8k","#### 2+2"))

    def test_length_stop_never_passes(self):
        self.assertFalse(score("gsm8k","42", "42", truncated=True)["correct"])
        self.assertTrue(score("gsm8k","42", "42")["correct"])

    def test_multiple_choice_not_arbitrary_text(self):
        self.assertEqual(parse("cmmlu","Ａ。"),"A")
        self.assertIsNone(parse("cmmlu","A or B"))


if __name__ == "__main__":
    unittest.main()
