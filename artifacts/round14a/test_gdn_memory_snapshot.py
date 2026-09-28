import unittest
from pathlib import Path

from gdn_memory_snapshot import (
    diff_active_blocks,
    match_live_tensors,
    plateau_verdict,
    snapshot_totals,
)


def block(address, size, state, frame, requested_size=None):
    return {
        "address": address,
        "size": size,
        "requested_size": size if requested_size is None else requested_size,
        "state": state,
        "frames": [
            {
                "filename": frame[0],
                "line": frame[1],
                "name": frame[2],
            }
        ],
    }


class SnapshotAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.before = {
            "segments": [
                {
                    "address": 1000,
                    "total_size": 1000,
                    "allocated_size": 300,
                    "active_size": 300,
                    "stream": 7,
                    "segment_type": "large",
                    "blocks": [
                        block(1000, 300, "active_allocated", ("baseline.py", 10, "build"), 280),
                        block(1300, 700, "inactive", ("", 0, "")),
                    ],
                }
            ],
            "device_traces": [[]],
        }
        self.after = {
            "segments": [
                self.before["segments"][0],
                {
                    "address": 4000,
                    "total_size": 800,
                    "allocated_size": 700,
                    "active_size": 800,
                    "stream": 11,
                    "segment_type": "large",
                    "blocks": [
                        block(4000, 500, "active_allocated", ("candidate.py", 44, "extend"), 480),
                        block(4500, 300, "active_awaiting_free", ("candidate.py", 51, "project"), 290),
                    ],
                },
            ],
            "device_traces": [[]],
        }

    def test_totals_separate_active_awaiting_and_inactive(self):
        totals = snapshot_totals(self.after)
        self.assertEqual(totals["segment_bytes"], 1800)
        self.assertEqual(totals["active_allocated_bytes"], 800)
        self.assertEqual(totals["active_awaiting_free_bytes"], 300)
        self.assertEqual(totals["inactive_bytes"], 700)
        self.assertEqual(totals["active_requested_bytes"], 1050)

    def test_diff_attributes_new_active_blocks_to_frames_and_streams(self):
        diff = diff_active_blocks(self.before, self.after)
        self.assertEqual(diff["new_active_bytes"], 800)
        self.assertEqual(diff["new_active_requested_bytes"], 770)
        self.assertEqual(diff["states"], {"active_allocated": 500, "active_awaiting_free": 300})
        self.assertEqual(diff["streams"], {"11": 800})
        self.assertEqual(diff["owners"]["candidate.py:44:extend"], 500)
        self.assertEqual(diff["owners"]["candidate.py:51:project"], 300)

    def test_live_tensor_ranges_match_containing_allocator_block(self):
        diff = diff_active_blocks(self.before, self.after)
        matches = match_live_tensors(
            diff["blocks"],
            [
                {"name": "core", "data_ptr": 4080, "storage_nbytes": 128, "shape": [2, 32]},
                {"name": "unrelated", "data_ptr": 9000, "storage_nbytes": 16, "shape": [1]},
            ],
        )
        self.assertEqual(matches[0]["tensor_names"], ["core"])
        self.assertEqual(matches[1]["tensor_names"], [])

    def test_plateau_requires_no_late_growth_and_bounded_span(self):
        stable = plateau_verdict([100, 140, 140, 140, 140, 140], warmup=1, tolerance_bytes=0)
        self.assertTrue(stable["pass"])
        growing = plateau_verdict([100, 140, 140, 180, 180, 220], warmup=1, tolerance_bytes=0)
        self.assertFalse(growing["pass"])
        self.assertEqual(growing["growth_after_warmup_bytes"], 80)


class ProbeSourceContractTest(unittest.TestCase):
    def test_staged_sigmoid_gate_has_an_explicit_import(self):
        source = Path(__file__).with_name("round14a_gdn_memory_probe.py").read_text()
        self.assertIn(
            "from sglang.srt.layers.round9.gdn import sigmoid_gated_rms_norm",
            source,
        )
        self.assertNotIn("torch.npu.empty_cache()", source)

    def test_paired_component_matches_production_inference_contract(self):
        source = Path(__file__).with_name("round14a_gdn_component_probe.py").read_text()
        self.assertGreaterEqual(source.count("@torch.inference_mode()"), 2)
        self.assertNotIn("torch.npu.empty_cache()", source)
        self.assertIn('"same_shape_cycles_required": 20', source)
        self.assertIn('"legal_global_cache_limit_bytes": 65536', source)

    def test_paired_component_has_frozen_blocks(self):
        source = Path(__file__).with_name("round14a_gdn_component_probe.py").read_text()
        self.assertIn("choices=(290, 291)", source)
        self.assertIn('(290, "A"), (291, "B")', source)


if __name__ == "__main__":
    unittest.main()
