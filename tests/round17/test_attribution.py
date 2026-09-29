import unittest


class Round17AttributionTests(unittest.TestCase):
    def test_observe_preserves_result_and_records_host_interval(self):
        from round17.attribution import observe

        rows = []
        ticks = iter((100, 145))
        wrapped = observe(
            lambda value: value + 1,
            rows.append,
            "QSA_INPUT_PROJ",
            3,
            lambda: {"request_id": "r", "mode": "decode", "batch_sequence": 7},
            clock=lambda: next(ticks),
        )

        self.assertEqual(wrapped(4), 5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["duration_ns"], 45)
        self.assertEqual(rows[0]["component"], "QSA_INPUT_PROJ")
        self.assertEqual(rows[0]["layer"], 3)

    def test_exclusive_account_rejects_children_outside_parent(self):
        from round17.attribution import exclusive_account

        with self.assertRaisesRegex(ValueError, "outside parent"):
            exclusive_account(
                (100, 200),
                [(90, 110)],
            )

    def test_exclusive_account_merges_overlapping_children(self):
        from round17.attribution import exclusive_account

        result = exclusive_account(
            (100, 200),
            [(110, 140), (130, 160), (180, 190)],
        )
        self.assertEqual(result["parent_ns"], 100)
        self.assertEqual(result["covered_child_ns"], 60)
        self.assertEqual(result["exclusive_ns"], 40)

    def test_component_contract_contains_complete_qsa_and_glue_boundaries(self):
        from round17.attribution import COMPONENTS

        required = {
            "DECODER_LAYER",
            "HC_ATTN_MIX",
            "HC_ATTN_COMBINE",
            "HC_MLP_MIX",
            "MLP_PARENT",
            "HC_MLP_COMBINE",
            "QSA_ATTENTION",
            "QSA_INPUT_PROJ",
            "QSA_QK_NORM",
            "QSA_ROPE",
            "QSA_INDEXER",
            "QSA_INDEX_PROJ",
            "QSA_INDEX_Q_NORM",
            "QSA_INDEX_K_NORM",
            "QSA_SPARSE_CORE",
            "QSA_OUTPUT_PROJ",
        }
        self.assertTrue(required.issubset(COMPONENTS))

    def test_dynamic_observe_uses_active_qsa_layer_and_preserves_call(self):
        from round17.attribution import dynamic_observe

        rows = []
        ticks = iter((500, 580))
        state = {
            "active_qsa_layer": 23,
            "current": {
                "request_id": "r",
                "mode": "decode",
                "batch_sequence": 2,
            },
        }
        wrapped = dynamic_observe(
            lambda value: value * 2,
            rows.append,
            "QSA_SPARSE_CORE",
            lambda: state,
            clock=lambda: next(ticks),
        )
        self.assertEqual(wrapped(6), 12)
        self.assertEqual(rows[0]["layer"], 23)
        self.assertEqual(rows[0]["duration_ns"], 80)

    def test_dynamic_observe_is_noop_outside_qsa_parent(self):
        from round17.attribution import dynamic_observe

        rows = []
        wrapped = dynamic_observe(
            lambda value: value - 1,
            rows.append,
            "QSA_SPARSE_CORE",
            lambda: {"active_qsa_layer": None, "current": None},
        )
        self.assertEqual(wrapped(4), 3)
        self.assertEqual(rows, [])

    def test_shared_qsa_provider_is_wrapped_once_and_uses_active_layer(self):
        from round17.attribution import install_dynamic_observer

        class Shared:
            def forward(self, value):
                return value + 1

        shared = Shared()
        rows = []
        ticks = iter((10, 20, 30, 50))
        state = {
            "active_qsa_layer": 3,
            "current": {"request_id": "r", "mode": "decode", "batch_sequence": 1},
        }
        installed = install_dynamic_observer(
            shared,
            "forward",
            rows.append,
            "QSA_ROPE",
            lambda: state,
            clock=lambda: next(ticks),
        )
        repeated = install_dynamic_observer(
            shared,
            "forward",
            rows.append,
            "QSA_ROPE",
            lambda: state,
            clock=lambda: next(ticks),
        )
        self.assertTrue(installed)
        self.assertFalse(repeated)
        self.assertEqual(shared.forward(4), 5)
        state["active_qsa_layer"] = 7
        self.assertEqual(shared.forward(8), 9)
        self.assertEqual([row["layer"] for row in rows], [3, 7])


if __name__ == "__main__":
    unittest.main()
