import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from migration_bridge.round9.runtime.context_binding import (
    parse_context_probe, load_context_probe, context_pools_ready)


class ContextBindingTests(unittest.TestCase):
    def test_disabled_does_not_inspect_runner(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsNone(context_pools_ready(object()))

    def test_four_explicit_targets_and_strict_profile(self):
        for target in (4096, 16384, 65536, 131072):
            profile = dict(schema=1, context_tokens=target, model_config_sha256='a'*64)
            self.assertEqual(parse_context_probe(profile), profile)
        for override in ({'context_tokens': True}, {'context_tokens': 8192},
                         {'schema': True}, {'model_config_sha256': 'changed'},
                         {'native_fp8': True}):
            with self.assertRaises(ValueError):
                parse_context_probe(dict(profile, **override))

    def test_profile_hash_is_checked_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.json'
            profile = dict(schema=1, context_tokens=4096, model_config_sha256='a'*64)
            path.write_text(json.dumps(profile))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict('os.environ', {'ROUND9_CONTEXT_PROBE_PROFILE': str(path),
                    'ROUND9_CONTEXT_PROBE_SHA256': sha}, clear=True):
                self.assertEqual(load_context_probe(), profile)
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'changed'): load_context_probe()

    def test_wrong_resolved_capacity_fails_before_model_or_pool_read(self):
        profile = dict(schema=1, context_tokens=4096, model_config_sha256='a'*64)
        args = NS(tp_size=1, device='npu', max_running_requests=1,
                  max_mamba_cache_size=1, max_total_tokens=4101,
                  context_length=4098, page_size=4, disable_radix_cache=True,
                  disable_overlap_schedule=True, disable_cuda_graph=True)
        with patch('migration_bridge.round9.runtime.context_binding.load_context_probe',
                   return_value=profile):
            with self.assertRaisesRegex(RuntimeError, 'exact serial'):
                context_pools_ready(NS(server_args=args))
