import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import download_guard as guard

class DownloadCommandTest(unittest.TestCase):
    def test_serial_transport_keeps_revision_allowlist_and_cache(self):
        manifest = {'repo': 'Qwen/Qwen3.8-Flash-Next', 'files': [{'path': 'config.json'}]}
        command = guard.download_command(manifest)
        self.assertEqual(command, [guard.HF, 'download', 'Qwen/Qwen3.8-Flash-Next',
            'config.json', '--revision', guard.REV, '--local-dir', str(guard.MODEL),
            '--max-workers', '1'])

if __name__ == '__main__': unittest.main()
