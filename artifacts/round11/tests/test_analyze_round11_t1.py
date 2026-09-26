import copy
import unittest

from artifacts.round11.analyze_round11_t1 import aggregate


def process_result(process, rows, *, aborts=0):
    return {
        "schema": 1,
        "status": "PASS_ROUND11_T1_PROCESS",
        "process": process,
        "generation_attempts": len(rows),
        "abort_controls": aborts,
        "automatic_retries": 0,
        "results": rows,
        "cancel_receipts": ([{"status": "PASS_CANCEL_REUSE"}] if aborts else []),
    }


def row(case, tokens, path, e2e, *, phase="extended_prefill", output=None):
    return {
        "case_id": case,
        "phase": phase,
        "state": "PASS",
        "e2e_seconds": e2e,
        "validated": {
            "output_ids": list(output or [3897, 248046]),
            "finite_output_logprobs": True,
        },
        "path": {
            "path": path,
            "request_total_tokens": tokens,
            "scheduler_chunks": [[start, min(32, tokens - start)] for start in range(0, tokens, 32)],
        },
    }


class AnalyzeRound11T1Tests(unittest.TestCase):
    def setUp(self):
        self.boundary = process_result(
            "BOUNDARY",
            [
                row("edge128", 128, "HYBRID", 26, phase="boundary"),
                row("edge129", 129, "LAYERWISE", 1100, phase="boundary"),
                row("edge130", 130, "LAYERWISE", 1110, phase="boundary"),
            ],
        )
        hybrid256_rows = [
            row(name, tokens, "HYBRID", 5, phase="canary", output=[index])
            for index, (name, tokens) in enumerate(
                [
                    ("canary_marker", 29),
                    ("canary_math391", 27),
                    ("canary_json", 28),
                    ("canary_quote", 31),
                    ("canary_transform", 25),
                ],
                start=1,
            )
        ]
        hybrid256_rows += [
            row("edge129", 129, "HYBRID", 27),
            row("edge130", 130, "HYBRID", 28),
            row("edge160", 160, "HYBRID", 30),
            row("edge192", 192, "HYBRID", 35),
            row("edge256", 256, "HYBRID", 44),
            row("edge256", 256, "HYBRID", 43, phase="repeatability"),
            row("edge192", 192, "HYBRID", 70, phase="decode_exact", output=list(range(32))),
            row("edge256", 256, "HYBRID", 120, phase="decode_exact", output=list(range(64))),
            row("cancel256", 256, "HYBRID", 2, phase="cancel"),
            row("canary_math391", 27, "HYBRID", 5, phase="reuse", output=[2]),
        ]
        next(item for item in hybrid256_rows if item["phase"] == "cancel")[
            "cancel_status"
        ] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
        self.hybrid256 = process_result("HYBRID256", hybrid256_rows, aborts=1)
        self.hybrid512 = process_result(
            "HYBRID512",
            [
                row("edge384", 384, "HYBRID", 50),
                row("edge512", 512, "HYBRID", 60),
                row("edge192", 192, "HYBRID", 180, phase="decode_exact", output=list(range(128))),
                row("edge512", 512, "HYBRID", 300, phase="decode_exact", output=list(range(256))),
            ],
        )
        self.hybrid1024 = process_result(
            "HYBRID1024",
            [row("edge768", 768, "HYBRID", 90), row("edge1024", 1024, "HYBRID", 130)],
        )

    def test_complete_hierarchy_reports_boundary_and_decode_qualification(self):
        result = aggregate(
            {
                "BOUNDARY": self.boundary,
                "HYBRID256": self.hybrid256,
                "HYBRID512": self.hybrid512,
                "HYBRID1024": self.hybrid1024,
            }
        )
        self.assertEqual(result["status"], "PASS_ROUND11_T1")
        self.assertEqual(result["hybrid_qualified_up_to_tokens"], 1024)
        self.assertEqual(result["decode_steps_qualified"], [32, 64, 128, 256])
        self.assertEqual(result["baseline_boundary"], {"128": "HYBRID", "129": "LAYERWISE", "130": "LAYERWISE"})
        self.assertLess(result["same_input_path_ratios"]["129"]["hybrid_over_layerwise"], 0.1)
        self.assertEqual(result["baseline_path_switch_at_tokens"], 129)
        self.assertEqual(result["crossover"], "NOT_FOUND_WITHIN_TESTED_RANGE")
        self.assertEqual(result["expanded_hybrid_crossover"], "NOT_FOUND_WITHIN_TESTED_RANGE")
        self.assertTrue(result["cancel_and_reuse_pass"])

    def test_resource_cutoff_at_1024_qualifies_hybrid_through_768(self):
        cutoff = process_result(
            "HYBRID1024",
            [row("edge768", 768, "HYBRID", 282)],
        )
        cutoff.update(
            status="PASS_ROUND11_T1_PROCESS_WITH_RESOURCE_CUTOFF",
            composite=True,
            generation_attempts=2,
            raw_source_status="FAIL",
            resource_cutoff_tokens=1024,
            hybrid_qualified_up_to_tokens=768,
            source_hashes={name: "a" * 64 for name in ("result", "supervisor", "monitor", "cleanup")},
            cleanup={"owned_container_removed": True, "physical12_after": "EMPTY"},
        )
        result = aggregate(
            {
                "BOUNDARY": self.boundary,
                "HYBRID256": self.hybrid256,
                "HYBRID512": self.hybrid512,
                "HYBRID1024": cutoff,
            }
        )
        self.assertEqual(result["status"], "PASS_ROUND11_T1_WITH_RESOURCE_CUTOFF")
        self.assertEqual(result["hybrid_qualified_up_to_tokens"], 768)
        self.assertEqual(result["resource_cutoff_tokens"], 1024)
        self.assertEqual(result["crossover"], "NOT_FOUND_WITHIN_QUALIFIED_RANGE")

    def test_incomplete_hierarchy_is_not_reported_as_qualified(self):
        with self.assertRaisesRegex(ValueError, "missing process evidence"):
            aggregate({"BOUNDARY": self.boundary, "HYBRID256": self.hybrid256})

    def test_failed_row_fails_closed(self):
        bad = copy.deepcopy(self.hybrid512)
        bad["results"][1]["state"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "non-passing row"):
            aggregate(
                {
                    "BOUNDARY": self.boundary,
                    "HYBRID256": self.hybrid256,
                    "HYBRID512": bad,
                    "HYBRID1024": self.hybrid1024,
                }
            )

    def test_repeatability_uses_identical_edge256_outputs(self):
        bad = copy.deepcopy(self.hybrid256)
        bad["results"][10]["validated"]["output_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "repeatability"):
            aggregate(
                {
                    "BOUNDARY": self.boundary,
                    "HYBRID256": bad,
                    "HYBRID512": self.hybrid512,
                    "HYBRID1024": self.hybrid1024,
                }
            )

    def test_composite_hybrid256_requires_explicit_physical_source_contract(self):
        composite = copy.deepcopy(self.hybrid256)
        composite.update(
            composite=True,
            physical_generation_attempts=17,
            physical_model_processes=2,
            composite_validation={
                "status": "PASS_STRICT_V109_PREFIX_PLUS_CANCEL_REPAIR",
                "immutable_failed_source_status": "FAIL",
                "failed_source_rows_used": 13,
                "repair_rows_used": 2,
                "source_hashes": {f"source_{index}": "a" * 64 for index in range(4)},
            },
        )
        documents = {
            "BOUNDARY": self.boundary,
            "HYBRID256": composite,
            "HYBRID512": self.hybrid512,
            "HYBRID1024": self.hybrid1024,
        }
        self.assertEqual(aggregate(documents)["status"], "PASS_ROUND11_T1")
        del composite["composite_validation"]["source_hashes"]
        with self.assertRaisesRegex(ValueError, "composite source contract"):
            aggregate(documents)


if __name__ == "__main__":
    unittest.main()
