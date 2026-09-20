import sys
import unittest
import os
import time
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import download_guard as guard

class DownloadCommandTest(unittest.TestCase):
    def test_finished_child_does_not_wait_for_resource_poll_interval(self):
        start = time.monotonic()
        with patch.object(guard, 'pause_reason', return_value=None), patch.object(guard, 'write_state'):
            guard.guarded([sys.executable, '-c', 'import time; time.sleep(0.1)'], 'TEST', dict(os.environ))
        self.assertLess(time.monotonic()-start, 2)

    def test_endpoint_override_preserves_proxy_and_rejects_unapproved_host(self):
        self.assertTrue(hasattr(guard, 'download_environment'))
        env = guard.download_environment({'KT_DOWNLOAD_ENDPOINT': 'https://huggingface.co',
                                         'HTTPS_PROXY': 'http://127.0.0.1:17891'})
        self.assertEqual(env['HF_ENDPOINT'], 'https://huggingface.co')
        self.assertEqual(env['HTTPS_PROXY'], 'http://127.0.0.1:17891')
        self.assertEqual(guard.download_environment({})['HF_ENDPOINT'], 'https://hf-mirror.com')
        with self.assertRaises(ValueError):
            guard.download_environment({'KT_DOWNLOAD_ENDPOINT': 'http://unapproved.invalid'})

    def test_serial_transport_keeps_revision_allowlist_and_cache(self):
        manifest = {'repo': 'Qwen/Qwen3.8-Flash-Next', 'files': [{'path': 'config.json'}]}
        command = guard.download_command(manifest)
        self.assertEqual(command, [guard.HF, 'download', 'Qwen/Qwen3.8-Flash-Next',
            'config.json', '--revision', guard.REV, '--local-dir', str(guard.MODEL),
            '--max-workers', '1'])

if __name__ == '__main__': unittest.main()
