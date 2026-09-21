import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

class RecoveryTest(unittest.TestCase):
    def test_range_uses_approved_endpoint_and_frozen_revision(self):
        import range_recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            def transport(cmd, phase, env):
                self.assertEqual(cmd[-1],
                    f'https://hf-mirror.com/Qwen/Qwen3.8-Flash-Next/resolve/{recovery.REV}/x')
                Path(cmd[cmd.index('--output')+1]).write_bytes(b'ab')
                Path(cmd[cmd.index('--dump-header')+1]).write_text(
                    'HTTP/1.1 206 OK\r\nContent-Range: bytes 0-1/2\r\n')
            with patch.object(recovery, 'guarded', side_effect=transport) as fetch:
                recovery.fetch_range(dict(path='x', size=2), 0,
                    {'HF_ENDPOINT': 'https://hf-mirror.com'}, Path(tmp))
                self.assertEqual(fetch.call_count, 1)
                with self.assertRaises(ValueError):
                    recovery.fetch_range(dict(path='x', size=2), 0,
                        {'HF_ENDPOINT': 'http://unapproved.invalid'}, Path(tmp))
                self.assertEqual(fetch.call_count, 1)

    def test_slow_progressing_range_gets_time_to_finish(self):
        import range_recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            def slow_transport(cmd, phase, env):
                # A 32MiB range at the observed 200KiB/s exceeds the old 90s cap.
                self.assertGreaterEqual(int(cmd[cmd.index('--max-time')+1]), 240)
                self.assertEqual(cmd[cmd.index('--speed-time')+1], '60')
                self.assertEqual(cmd[cmd.index('--speed-limit')+1], '1024')
                Path(cmd[cmd.index('--output')+1]).write_bytes(b'abcd')
                Path(cmd[cmd.index('--dump-header')+1]).write_text(
                    'HTTP/1.1 206 OK\r\nContent-Range: bytes 0-3/4\r\n')
            with patch.object(recovery, 'guarded', side_effect=slow_transport):
                result = recovery.fetch_range(dict(path='x', size=4), 0, {}, Path(tmp))
                self.assertEqual(result[2].read_bytes(), b'abcd')

    def test_eight_shard_wave_preserves_success_and_isolates_transport_failure(self):
        import range_recovery as recovery
        barrier = threading.Barrier(8)
        def fetch(row):
            barrier.wait(timeout=2)
            if row['path'] == '3':
                raise RuntimeError('transport')
        rows = [dict(path=str(i)) for i in range(8)]
        failures = recovery.shard_wave(fetch, rows, 8)
        self.assertEqual([r['path'] for r in failures], ['3'])
        with self.assertRaises(ValueError):
            recovery.shard_wave(fetch, rows+rows, 8)

    def test_integrity_failure_is_not_transport_retry(self):
        import range_recovery as recovery
        def fetch(row):
            raise ValueError('SHA mismatch')
        with self.assertRaises(ValueError):
            recovery.shard_wave(fetch, [dict(path='x')], 1)

    def test_tiny_tail_allows_redirect_but_validates_exact_body(self):
        import range_recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            def fake_transport(cmd, phase, env):
                self.assertEqual(int(cmd[cmd.index('--max-filesize')+1]), recovery.CHUNK)
                Path(cmd[cmd.index('--output')+1]).write_bytes(b'cd')
                Path(cmd[cmd.index('--dump-header')+1]).write_text(
                    'HTTP/1.1 302 Found\r\nContent-Length: 1000\r\n\r\n'
                    'HTTP/1.1 206 OK\r\nContent-Range: bytes 2-3/4\r\n')
            with patch.object(recovery, 'guarded', side_effect=fake_transport):
                result = recovery.fetch_range(dict(path='x', size=4), 2, {}, Path(tmp))
                self.assertEqual(result[2].read_bytes(), b'cd')

    def test_two_connections_overlap_but_results_stay_ordered(self):
        import range_recovery as recovery
        barrier = threading.Barrier(2)
        def fetch(value):
            barrier.wait(timeout=2)
            return value
        self.assertEqual(recovery.fetch_batch(fetch, [0, 1], 2), [0, 1])
        with self.assertRaises(ValueError):
            recovery.fetch_batch(fetch, [0, 1], 3)

    def test_only_validated_range_is_appended(self):
        import range_recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, part = root/'data', root/'part'
            target.write_bytes(b'ab')
            part.write_bytes(b'cd')
            with self.assertRaises(ValueError):
                recovery.append_range(target, part, 'HTTP/1.1 200 OK\r\n', 2, 3, 4)
            self.assertEqual(target.read_bytes(), b'ab')
            recovery.append_range(target, part, 'HTTP/1.1 206 OK\r\nContent-Range: bytes 2-3/4\r\n', 2, 3, 4)
            self.assertEqual(target.read_bytes(), b'abcd')

    def test_publication_requires_sha_and_never_overwrites(self):
        import range_recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, model = root/'stage', root/'model'
            stage.mkdir(); model.mkdir()
            (stage/'x').write_bytes(b'bad')
            row = dict(path='x', size=3, sha256=hashlib.sha256(b'yes').hexdigest())
            with self.assertRaises(ValueError):
                recovery.publish(stage, model, row)
            self.assertFalse((model/'x').exists())
            (stage/'x').write_bytes(b'yes')
            recovery.publish(stage, model, row)
            self.assertEqual((model/'x').read_bytes(), b'yes')
            with self.assertRaises(FileExistsError):
                recovery.publish(stage, model, row)
