import unittest
from verified_bf16_cache import VerifiedExpertCache


class CacheTests(unittest.TestCase):
    def test_port_is_opt_in_and_requires_staging(self):
        from bf16_context_compat import port_source
        from test_llamafile_staging import PIPELINE
        self.assertNotIn('cached_stager_class',port_source(PIPELINE,llamafile_staging=True))
        self.assertIn('cached_stager_class',port_source(PIPELINE,llamafile_staging=True,verified_cache=True))
        with self.assertRaisesRegex(ValueError,'REQUIRES_STAGING'):
            port_source(PIPELINE,verified_cache=True)

    def test_private_cache_provenance_binding(self):
        from runtime_selection import select_sglang_runtime
        root='/workspace/c64-dev/round7/cache/sglang-python'
        wrapper=root+'/sglang/srt/layers/moe/kt_ep_wrapper.py'
        helper=root+'/sglang/srt/layers/moe/verified_bf16_cache.py'
        p=dict(sglang_runtime_root=root,files={wrapper:{'sha256':'w'},helper:{'sha256':'c'}},
               bf16_context_port=dict(ported_sha256='w',verified_cache_helper_sha256='c'))
        self.assertEqual(select_sglang_runtime(dict(sglang_runtime_root=root),p),root)
        p['files'][helper]['sha256']='different'
        with self.assertRaisesRegex(ValueError,'PRIVATE_VERIFIED_CACHE_HASH_BINDING'):
            select_sglang_runtime(dict(sglang_runtime_root=root),p)

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
