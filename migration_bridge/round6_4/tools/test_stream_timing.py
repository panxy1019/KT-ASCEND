import json
import unittest
from unittest.mock import patch
from stream_timing import consume


class StreamTests(unittest.TestCase):
    def event(self, ids, rid="x"):
        return "data: "+json.dumps(dict(output_ids=ids,meta_info=dict(id=rid,finish_reason=dict(type="length"))))

    def test_cumulative_and_client_clock(self):
        stamps = iter([11.,12.,13.])
        with patch("stream_timing.validate") as check:
            got = consume([self.event([1]),self.event([1,2]),"data: [DONE]"], "x",2,10.,clock=lambda:next(stamps))
        check.assert_called_once()
        self.assertEqual(got["ttft_seconds"],1.)
        self.assertEqual(got["decode_tokens_per_second"],1.)
        self.assertEqual(got["seconds"],3.)

    def test_cross_rid_and_prefix_escape(self):
        with self.assertRaisesRegex(ValueError,"STREAM_RID"):
            consume([self.event([1],rid="other")],"x",2,0.)
        with self.assertRaisesRegex(ValueError,"STREAM_TOKEN_HISTORY"):
            consume([self.event([1]),self.event([2,3])],"x",2,0.)


if __name__ == "__main__":
    unittest.main()
