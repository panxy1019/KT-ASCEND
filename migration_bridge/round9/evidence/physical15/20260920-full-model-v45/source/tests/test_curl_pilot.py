import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from curl_shard_pilot import validate_range


class RangeTest(unittest.TestCase):
    def test_exact_range_after_redirect(self):
        self.assertTrue(validate_range('HTTP/1.1 302 Found\r\n\r\nHTTP/2 206\r\nContent-Range: bytes 8-15/32\r\n\r\n', 8, 15, 32, 8))

    def test_reject_wrong_or_ignored_range(self):
        for header, size in [('HTTP/2 200\r\n', 32),
                             ('HTTP/2 206\r\nContent-Range: bytes 0-7/32\r\n', 8),
                             ('HTTP/2 206\r\nContent-Range: bytes 8-15/32\r\n', 7),
                             ('HTTP/2 206\r\nContent-Range: bytes 8-15/99\r\n', 8)]:
            with self.subTest(header=header, size=size):
                with self.assertRaises(ValueError):
                    validate_range(header, 8, 15, 32, size)
