import unittest
from freeze_quality_protocol import validate_dev


class DevFreezeTests(unittest.TestCase):
    def test_incomplete_cannot_freeze(self):
        with self.assertRaisesRegex(ValueError,"DEV_NOT_COMPLETE"):
            validate_dev({"status":"RUNNING"},{})

    def test_pilot_not_final_dev_review(self):
        report=dict(status="DEV_EXECUTION_COMPLETE_PENDING_PROTOCOL_REVIEW",protocol_v2=True,
                    primary_outputs_observed=False,profile="D1_NODEFER",rows=[])
        with self.assertRaisesRegex(ValueError,"32_PLUS_32"):
            validate_dev(report,{})

    def test_approval_requires_explained_protocol_review(self):
        report=dict(status="DEV_EXECUTION_COMPLETE_PENDING_PROTOCOL_REVIEW",protocol_v2=True,
                    primary_outputs_observed=False,profile="D1_NODEFER",
                    rows=[dict(benchmark=b) for b in ("gsm8k","cmmlu") for _ in range(32)])
        review=dict(status="APPROVED_FOR_PRE_PRIMARY_FREEZE",primary_outputs_observed=False)
        with self.assertRaisesRegex(ValueError,"MISSING_DEV_REVIEW"):
            validate_dev(report,review)


if __name__ == "__main__":
    unittest.main()
