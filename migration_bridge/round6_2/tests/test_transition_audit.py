import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_deferred_transitions import validate_transition


def events(hit):
    rows = []
    for layer in (7, 8):
        if layer == 8 and hit:
            rows.append(dict(event="previous_pending_consumed", producer_layer=7, consumer_layer=8))
        rows.append(dict(event="pending_state", layer_idx=layer, pending_layers=[7] if hit and layer == 7 else []))
        for kind in ("sync", "sync_complete"):
            rows.append(dict(event=kind, layer_idx=layer, allow_pending=int(hit and layer == 7)))
    return rows


class TransitionTests(unittest.TestCase):
    def test_both_modes(self):
        self.assertTrue(validate_transition(events(False), 0))
        self.assertTrue(validate_transition(events(True), 4))

    def test_consumption_missing(self):
        rows = events(True)
        rows.pop(3)
        with self.assertRaisesRegex(ValueError, "EXACTLY_ONCE"):
            validate_transition(rows, 1)

    def test_consumption_duplicate(self):
        rows = events(True)
        rows.insert(3, copy.deepcopy(rows[3]))
        with self.assertRaisesRegex(ValueError, "EXACTLY_ONCE"):
            validate_transition(rows, 1)

    def test_pending_disagrees_with_routes(self):
        with self.assertRaisesRegex(ValueError, "PENDING_ROUTE"):
            validate_transition(events(False), 1)

    def test_reordered_sync(self):
        rows = events(True)
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaisesRegex(ValueError, "ORDER"):
            validate_transition(rows, 1)


if __name__ == "__main__":
    unittest.main()
