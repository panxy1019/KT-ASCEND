import copy
import unittest
from frozen_threshold import validate_ack, load_threshold


class ThresholdFreeze(unittest.TestCase):
    def setUp(self):
        self.selected = dict(status='FROZEN_NOT_HELDOUT_QUALIFIED', threshold=512,
            profile_sha256='a'*64, dev_sha256='b'*64,
            quality_used_for_selection=False, heldout_used_for_selection=False)
        self.ack = dict(local_candidate_commit='c'*40, threshold_freeze_sha256='d'*64,
            profile_sha256='a'*64, dev_sha256='b'*64)

    def check(self, selected=None, ack=None):
        validate_ack(self.selected if selected is None else selected,
            self.ack if ack is None else ack, 'd'*64, 'a'*64)

    def test_matching_explicit_commit_binding(self):
        self.check()

    def test_missing_ack_or_threshold(self):
        with self.assertRaisesRegex(ValueError, 'ACK_REQUIRED'):
            load_threshold(None, None, None)

    def test_uncommitted_or_short_commit_rejected(self):
        for value in ('', 'c'*9, 'not-a-sha'):
            ack = self.ack.copy(); ack['local_candidate_commit'] = value
            with self.assertRaisesRegex(ValueError, 'COMMIT_FREEZE'):self.check(ack=ack)

    def test_ack_artifact_drift_rejected(self):
        for field in ('threshold_freeze_sha256', 'profile_sha256', 'dev_sha256'):
            ack = self.ack.copy(); ack[field] = 'e'*64
            with self.assertRaises(ValueError):self.check(ack=ack)

    def test_threshold_state_or_profile_drift_rejected(self):
        for field, value in [('status', 'DEV_SCREEN_COMPLETE'), ('threshold', 1),
                             ('profile_sha256', 'e'*64), ('dev_sha256', '')]:
            selected = self.selected.copy(); selected[field] = value
            with self.assertRaises(ValueError):self.check(selected=selected)

    def test_quality_or_heldout_fitting_rejected(self):
        for field in ('quality_used_for_selection', 'heldout_used_for_selection'):
            for value in (True, None, 0):
                selected = copy.deepcopy(self.selected); selected[field] = value
                with self.assertRaisesRegex(ValueError, 'CONTAMINATION'):self.check(selected=selected)


if __name__ == '__main__':unittest.main()
