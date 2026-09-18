import unittest
from verified_bf16_cache import VerifiedExpertCache


class CacheTests(unittest.TestCase):
    def test_reuses_only_same_binding_and_expert(self):
        c = VerifiedExpertCache(('source', 0), 2, 16)
        calls = []
        def build():
            calls.append(1)
            return object()
        first = c.get(('source', 0), 1, build)
        self.assertIs(first, c.get(('source', 0), 1, build))
        self.assertEqual(len(calls), 1)
        self.assertEqual(c.resident_bytes, 16)
        with self.assertRaisesRegex(ValueError, 'BINDING_CHANGED'):
            c.get(('source', 1), 1, build)

    def test_validation_failure_does_not_publish(self):
        c = VerifiedExpertCache('s', 64, 32)
        def fail():
            raise ValueError('lossy')
        with self.assertRaisesRegex(ValueError, 'lossy'):
            c.get('s', 0, fail)
        self.assertEqual(c.values, {})
        self.assertEqual(c.misses, 0)
        for expert in (-1, 64):
            with self.assertRaisesRegex(ValueError, 'EXPERT_INDEX'):
                c.get('s', expert, fail)


if __name__ == '__main__':
    unittest.main()
