import tempfile
import unittest
from pathlib import Path
from cgroup_memory import swap_bytes


class CgroupMemory(unittest.TestCase):
    def test_versions_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'memory').mkdir()
            (r/'memory/memory.stat').write_text('total_swap 0\n')
            self.assertEqual(swap_bytes(r),0)
            (r/'memory/memory.stat').write_text('total_swap 4096\n')
            self.assertEqual(swap_bytes(r),4096)
            (r/'memory/memory.stat').write_text('rss 100\n')
            with self.assertRaises(ValueError):swap_bytes(r)
            (r/'memory.swap.current').write_text('8192')
            self.assertEqual(swap_bytes(r),8192)
