import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from round10_t5_aggregate import ERROR_KEYS,validate


class T5AggregateTests(unittest.TestCase):
    def test_gdn_output_key_matches_real_probe_schema(self):
        self.assertIn("relative_l2",ERROR_KEYS)
        self.assertNotIn("canonical_output_relative_l2",ERROR_KEYS)

    def fixture(self):
        identity={"revision":"x"}
        grow={"dtype":"torch.float32","repeat_hashes":["h"]*3,**{key:0.0 for key in ERROR_KEYS}}
        qrow={"dtype":"torch.bfloat16","relative_l2":0.0,"index_query_relative_l2":0.0,
              "compressed_key_relative_l2":0.0,"repeat_hashes":["q"]*3}
        g={"status":"PASS_GDN_COMPONENT_ONLY","rows":[dict(grow) for _ in range(18)],
           "gates":{"float32":1e-4,"bfloat16":.03},"after_allocated":0,"after_reserved":0,
           "peak_allocated":1,"source_identity":identity}
        q={"status":"PASS_QSA_LOCAL_ONLY","rows":[dict(qrow) for _ in range(30)],
           "gates":{"float32":1e-4,"bfloat16":.03},"after_allocated":0,"after_reserved":0,
           "peak_allocated":2,"topk_stress":True,"signed_multihead_ranking":True,"source_identity":identity}
        checks={key:True for key in ("original_rows_exact","duplicates_exact","cross_shard_exact",
          "consumer_stream_order","slot_reuse","request_switch","cancel_late_completion","zero_credits_after_close")}
        p={"status":"PASS_PLE_ASYNC_ORIGINAL_ROWS","physical_device":12,"checks":checks,
           "cases":[{"exact":True}]*3,"peak_extra_bytes":3,"torch_version":"2.9","torch_npu_version":"2.9",
           "source_identity":identity}
        return g,q,p

    def test_accepts_complete_independent_receipts(self):
        self.assertEqual(validate(*self.fixture())["status"],"PASS_ROUND10_T5_COMPONENTS")

    def test_rejects_each_missing_semantic_gate(self):
        g,q,p=self.fixture();g["rows"][0]["decode_relative_l2"]=1.0
        with self.assertRaises(RuntimeError):validate(g,q,p)
        g,q,p=self.fixture();q["signed_multihead_ranking"]=False
        with self.assertRaises(RuntimeError):validate(g,q,p)
        g,q,p=self.fixture();p["checks"]["slot_reuse"]=False
        with self.assertRaises(RuntimeError):validate(g,q,p)


if __name__=='__main__':unittest.main()
