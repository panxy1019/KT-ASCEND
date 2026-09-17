import unittest
from performance_screen import summarize_events


def packet(ids):
    return dict(output_ids=ids,meta_info=dict(output_token_logprobs=[[-1.,i,None] for i in ids],
        finish_reason=dict(type='length')))


class Timing(unittest.TestCase):
    def test_distinguish_first_token_from_request_latency(self):
        result=summarize_events([(2.,packet([4])),(5.,packet([4,7,8,9]))],4)
        self.assertEqual(result['ttft_client_seconds'],2.)
        self.assertEqual(result['decode_tokens_per_second'],1.)

    def test_coalesced_stream_not_decode_benchmark(self):
        result=summarize_events([(2.,packet([4,7]))],2)
        self.assertIsNone(result['decode_tokens_per_second'])

    def test_nonfinite_rejected(self):
        p=packet([4]);p['meta_info']['output_token_logprobs'][0][0]=float('nan')
        with self.assertRaises(ValueError):summarize_events([(2.,p)],1)


if __name__=='__main__':unittest.main()
