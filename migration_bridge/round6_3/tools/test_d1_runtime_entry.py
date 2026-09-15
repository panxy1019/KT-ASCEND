import unittest
from run_d1_runtime import validate_entry, S6_SHA


class RuntimeEntryTests(unittest.TestCase):
    def values(self):
        return ({'corpus':'S6', 'prompts':[{}]*12, 'sha256':S6_SHA},
                {'version':'round6.3-v2-candidate','preregistration_sha256':'p'}, {'sha256':'p'}, 'a'*40)

    def test_new_version_explicit(self):
        validate_entry(*self.values())

    def test_old_version_not_relabelled(self):
        values = self.values()
        values[1]['version'] = 'round6.2-v2-candidate'
        with self.assertRaises(ValueError):
            validate_entry(*values)

    def test_commit_required(self):
        values = self.values()
        with self.assertRaises(ValueError):
            validate_entry(*values[:3], 'pending')


if __name__ == '__main__':
    unittest.main()
