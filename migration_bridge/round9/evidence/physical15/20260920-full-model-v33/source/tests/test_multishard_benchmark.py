import sys
import threading
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


class MultiShardTest(unittest.TestCase):
    def test_four_shards_run_concurrently_without_reordering(self):
        import multishard_benchmark as benchmark
        barrier = threading.Barrier(4)
        def fetch(row):
            barrier.wait(timeout=2)
            return row
        self.assertEqual(benchmark.run_shards(fetch, [1, 2, 3, 4], 4), [1, 2, 3, 4])

    def test_serial_baseline_retains_all_shards(self):
        import multishard_benchmark as benchmark
        self.assertEqual(benchmark.run_shards(lambda r: r*2, [1, 2, 3, 4], 1), [2, 4, 6, 8])
        with self.assertRaises(ValueError):
            benchmark.run_shards(lambda r: r, [1, 2, 3, 4], 8)
