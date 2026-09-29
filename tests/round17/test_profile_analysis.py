import unittest


class Round17ProfileAnalysisTests(unittest.TestCase):
    def test_steady_decode_drops_only_first_and_last_decode_batches(self):
        from round17.profile_analysis import select_steady_decode

        rows = [
            {"request_id": "x:4:W2_MEASUREMENT", "mode": "extend", "batch_sequence": 0},
            *[
                {"request_id": "x:4:W2_MEASUREMENT", "mode": "decode", "batch_sequence": i}
                for i in range(4, 11)
            ],
        ]
        selected = select_steady_decode(rows)
        self.assertEqual([row["batch_sequence"] for row in selected], [5, 6, 7, 8, 9])

    def test_qsa_exclusive_uses_union_of_nested_children(self):
        from round17.profile_analysis import qsa_exclusive_ns

        spans = [
            {"component": "QSA_ATTENTION", "layer": 3, "start_ns": 100, "end_ns": 300},
            {"component": "QSA_INPUT_PROJ", "layer": 3, "start_ns": 110, "end_ns": 150},
            {"component": "QSA_INDEXER", "layer": 3, "start_ns": 140, "end_ns": 210},
            {"component": "QSA_INDEX_PROJ", "layer": 3, "start_ns": 160, "end_ns": 190},
            {"component": "QSA_SPARSE_CORE", "layer": 3, "start_ns": 240, "end_ns": 280},
        ]
        # Child union is [110,210] plus [240,280] = 140 ns.
        self.assertEqual(qsa_exclusive_ns(spans), 60)

    def test_glue_intervals_follow_real_attention_and_expert_dependencies(self):
        from round17.profile_analysis import glue_intervals_ns

        spans = [
            {"component": "GDN", "layer": 0, "start_ns": 10, "end_ns": 30},
            {"component": "ROUTED_EXPERT", "layer": 0, "start_ns": 50, "end_ns": 70},
            {"component": "QSA_ATTENTION", "layer": 1, "start_ns": 100, "end_ns": 130},
            {"component": "ROUTED_EXPERT", "layer": 1, "start_ns": 160, "end_ns": 180},
            {"component": "DECODER_LAYER", "layer": 1, "start_ns": 90, "end_ns": 210},
        ]
        result = glue_intervals_ns(spans)
        self.assertEqual(result["attention_to_expert_ns"], 50)
        self.assertEqual(result["expert_to_next_attention_ns"], 30)
        self.assertEqual(result["last_expert_to_layer_end_ns"], 30)


if __name__ == "__main__":
    unittest.main()
