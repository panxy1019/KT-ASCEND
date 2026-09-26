import unittest

from artifacts.round11.analyze_round11_t5_qsa_component import analyze_component


def row(layer, context, ratio, *, batch=32, rel=2e-5):
    return {
        "layer": layer,
        "context_tokens": context,
        "query_tokens": batch,
        "baseline_median_ms": 10.0,
        "candidate_median_ms": 10.0 * ratio,
        "candidate_over_baseline": ratio,
        "relative_l2": rel,
        "candidate_repeat_hashes": ["same", "same", "same"],
    }


class Round11T5QSAComponentAnalyzerTests(unittest.TestCase):
    def test_promotes_only_when_each_prefill_context_improves_and_decode_is_safe(self):
        rows = [
            row(layer, context, 0.70 + layer / 1000)
            for layer in (3, 23, 47)
            for context in (32, 64, 128)
        ]
        rows += [row(layer, 129, 1.02, batch=1) for layer in (3, 23, 47)]
        result = analyze_component({
            "status": "PASS_ROUND11_T5_QSA_COMPONENT",
            "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
            "rows": rows,
        })
        self.assertEqual(result["disposition"], "QSA_COMPONENT_PROMOTE_TO_REQUEST")
        self.assertEqual(set(result["prefill_context_median_ratio"]), {"32", "64", "128"})

    def test_rejects_numerical_drift_even_when_faster(self):
        rows = [
            row(layer, context, 0.5)
            for layer in (3, 23, 47)
            for context in (32, 64, 128)
        ]
        rows += [row(layer, 129, 0.9, batch=1) for layer in (3, 23, 47)]
        rows[0]["relative_l2"] = 0.002
        with self.assertRaisesRegex(ValueError, "numerical"):
            analyze_component({
                "status": "PASS_ROUND11_T5_QSA_COMPONENT",
                "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
                "rows": rows,
            })


if __name__ == "__main__":
    unittest.main()
