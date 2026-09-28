#!/usr/bin/env python3

from dataclasses import replace
import unittest

from round15_b_segment_policy import decide_resident_moe_graph


VALID = {
    "enabled": True,
    "forward_mode": "DECODE",
    "hidden_shape": (1, 2560),
    "topk_ids_shape": (1, 10),
    "topk_weights_shape": (1, 10),
    "hidden_dtype": "torch.bfloat16",
    "ids_dtype": "torch.int32",
    "weights_dtype": "torch.float32",
    "layer_idx": 0,
    "tp_size": 1,
    "num_gpu_experts": 32,
    "method_name": "UnquantizedFusedMoEMethod",
    "return_fp32_contribution": True,
    "with_bias": False,
    "max_running_requests": 1,
}


class SegmentPolicyTests(unittest.TestCase):
    def decide(self, **changes):
        values = dict(VALID)
        values.update(changes)
        return decide_resident_moe_graph(**values)

    def test_exact_decode_contract_is_admitted(self):
        decision = self.decide()
        self.assertTrue(decision.use_graph)
        self.assertEqual(
            decision.reason,
            "QUALIFIED_DECODE_BS1_Q32_BF16_NZ29_TOPK_IDS_INT32",
        )
        self.assertEqual(decision.key[0], 0)

    def test_one_token_extend_is_not_mistaken_for_decode(self):
        decision = self.decide(forward_mode="EXTEND")
        self.assertFalse(decision.use_graph)
        self.assertEqual(decision.reason, "FORWARD_MODE_EXTEND")

    def test_unknown_mode_fails_closed(self):
        self.assertFalse(self.decide(forward_mode=None).use_graph)

    def test_only_batch_one_hidden_geometry_is_admitted(self):
        self.assertFalse(self.decide(hidden_shape=(2, 2560)).use_graph)
        self.assertFalse(self.decide(hidden_shape=(1, 1280)).use_graph)

    def test_only_top10_route_geometry_is_admitted(self):
        self.assertFalse(self.decide(topk_ids_shape=(1, 8)).use_graph)
        self.assertFalse(self.decide(topk_weights_shape=(2, 10)).use_graph)

    def test_dtype_drift_falls_back(self):
        self.assertFalse(self.decide(hidden_dtype="torch.float16").use_graph)
        self.assertFalse(self.decide(ids_dtype="torch.int64").use_graph)
        self.assertFalse(self.decide(weights_dtype="torch.bfloat16").use_graph)

    def test_provider_and_precision_contract_are_exact(self):
        self.assertFalse(self.decide(method_name="Fp8MoEMethod").use_graph)
        self.assertFalse(self.decide(return_fp32_contribution=False).use_graph)
        self.assertFalse(self.decide(with_bias=True).use_graph)

    def test_topology_and_ownership_are_serial_tp1_q32(self):
        self.assertFalse(self.decide(tp_size=2).use_graph)
        self.assertFalse(self.decide(num_gpu_experts=42).use_graph)
        self.assertFalse(self.decide(max_running_requests=2).use_graph)

    def test_layer_range_is_bounded(self):
        self.assertFalse(self.decide(layer_idx=-1).use_graph)
        self.assertFalse(self.decide(layer_idx=48).use_graph)
        self.assertTrue(self.decide(layer_idx=47).use_graph)

    def test_disabled_candidate_is_eager(self):
        self.assertEqual(self.decide(enabled=False).reason, "DISABLED")


if __name__ == "__main__":
    unittest.main()
