import unittest
from run_batch_capture import response_map


def response(rid):
    return dict(output_ids=[7]*64, meta_info=dict(id=rid, output_token_logprobs=[[-1.,7]]*64))


class BatchResponseTests(unittest.TestCase):
    def test_map_by_rid_not_response_order(self):
        self.assertEqual(response_map([response("b"),response("a")], ["a","b"],64), {"a":[7]*64,"b":[7]*64})

    def test_duplicate_unknown_and_incomplete_fail(self):
        for batch in ([response("a"),response("a")], [response("c"),response("a")], [response("a")]):
            with self.assertRaises(ValueError):
                response_map(batch,["a","b"],64)


if __name__ == "__main__":
    unittest.main()
