import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("position", Path(__file__).resolve().parents[1] / "tools/validate_position.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row():
    return dict(position_id="smoke:0", request_id="r0", prompt_id="smoke",
                prompt_hash="a"*64, history_hash="b"*64, request_input_hash="c"*64,
                token_index=0, current_input_position=1, forward_id="pid:1", logits_sha256="d"*64,
                baseline_history_sha256="e"*64, logits_shape=[102400], mode="defer1",
                trace_complete=True, pending_drained=True,
                deferred_event_count_current_row=0, deferred_event_count_prefix=0,
                cpu_route_count_current_row=0, cpu_route_count_prefix=0,
                deferred_route_records=[], deferred_exposure="NO_EXPOSURE")


class PositionTests(unittest.TestCase):
    def test_no_exposure(self):
        self.assertEqual(module.validate(row()), "NO_EXPOSURE")

    def test_legacy_proxy_is_insufficient(self):
        value = row()
        value.update(deferred_event_count_current_row=1, cpu_route_count_current_row=1,
                     deferred_exposure="CURRENT_EXPOSURE", cpu_route_weights=[.1])
        with self.assertRaisesRegex(ValueError, "DEFERRED_ONLY"):
            module.validate(value)

    def test_missing_prefix_is_not_zero(self):
        value = row()
        del value["deferred_event_count_prefix"]
        with self.assertRaisesRegex(ValueError, "MISSING_FIELDS"):
            module.validate(value)

    def test_prefix_only(self):
        value = row()
        value.update(deferred_event_count_prefix=1, cpu_route_count_prefix=1,
                     deferred_exposure="PREFIX_ONLY_EXPOSURE")
        value["deferred_route_records"] = [dict(event_id="r0:7:0:16", request_id="r0",
            producer_layer=7, consumer_layer=8, expert_logical_id=16, topk_rank=5,
            token_position=0, scope="prefix", route_score=.1, normalized_route_weight=.2)]
        self.assertEqual(module.validate(value), "PREFIX_ONLY_EXPOSURE")
        broken = copy.deepcopy(value)
        broken["deferred_route_records"][0]["producer_layer"] = 8
        with self.assertRaisesRegex(ValueError, "NON_D1"):
            module.validate(broken)

    def test_pending_and_nonfinite_fail(self):
        value = row()
        value["pending_drained"] = False
        with self.assertRaisesRegex(ValueError, "PENDING"):
            module.validate(value)


if __name__ == "__main__":
    unittest.main()
