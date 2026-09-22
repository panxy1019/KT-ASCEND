import unittest
from run_static_phase import validate_ack


class Freeze(unittest.TestCase):
    def test_explicit_commit_and_both_artifact_hashes(self):
        validate_ack(dict(local_candidate_commit='a'*40,frequency_profile_sha256='b'*64,
                          frequency_activation_sha256='c'*64),'b'*64,'c'*64)

    def test_missing_commit_or_drift_fails(self):
        for ack in ({},dict(local_candidate_commit='a'*40,frequency_profile_sha256='b'*64,
                           frequency_activation_sha256='d'*64)):
            with self.assertRaises(ValueError):validate_ack(ack,'b'*64,'c'*64)
