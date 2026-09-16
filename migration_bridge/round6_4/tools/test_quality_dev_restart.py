import copy
import unittest
from compare_quality_dev_restart import compare


class DevRestartTests(unittest.TestCase):
    def setUp(self):
        self.row={"r":dict(payload=dict(input_ids=[100000,1]),response=dict(
            output_ids=[2,3],text="answer",meta_info=dict(finish_reason=dict(type="stop"))))}

    def test_exact(self):
        self.assertEqual(compare(self.row,copy.deepcopy(self.row)),["r"])

    def test_token_change_fails(self):
        other=copy.deepcopy(self.row)
        other["r"]["response"]["output_ids"]=[2,4]
        with self.assertRaisesRegex(ValueError,"NONDETERMINISM"):
            compare(self.row,other)

    def test_input_change_fails(self):
        other=copy.deepcopy(self.row)
        other["r"]["payload"]["input_ids"]=[1]
        with self.assertRaisesRegex(ValueError,"PROTOCOL_DRIFT"):
            compare(self.row,other)


if __name__ == "__main__":
    unittest.main()
