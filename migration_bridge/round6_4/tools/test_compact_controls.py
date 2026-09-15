import io
import json
from pathlib import Path
import tempfile
import socket
import unittest
from unittest.mock import patch

import run_compact_controls as compact


class CompactTests(unittest.TestCase):
    def test_active_listener_is_not_reused(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            with self.assertRaises(OSError):
                compact.ensure_free_port(listener.getsockname()[1])

    def test_time_wait_does_not_block_restart(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            address = listener.getsockname()
            listener.listen(1)
            client.connect(address)
            connection, _ = listener.accept()
            connection.close()  # server side active close creates TIME_WAIT
            self.assertEqual(client.recv(1), b'')
        compact.ensure_free_port(address[1])

    def exercise(self, wrong_token=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'D1'
            class Server:
                pid = 123
                returncode = None
                def poll(self):
                    return self.returncode
                def send_signal(self, sig):
                    self.signal = sig
                def wait(self, timeout):
                    self.returncode = 1
                    (root / 'guard').mkdir()
                    (root / 'guard/result.json').write_text(json.dumps({'reason':'CANCELLED'}))
            server = Server()
            count = 0
            def response(request, timeout):
                nonlocal count
                count += 1
                payload = json.loads(request.data)
                budget = payload['sampling_params']['max_new_tokens']
                token = 8 if wrong_token and count == 2 else 7
                return io.StringIO(json.dumps(dict(output_ids=[token]*budget,
                    meta_info=dict(id=payload['rid'], output_token_logprobs=[[-1.0,token]]*budget))))
            selected = {pid:{'input_ids':[1]} for pid in compact.PROMPTS}
            with patch.object(compact, 'ensure_free_port'), \
                    patch.object(compact.subprocess, 'Popen', return_value=server), \
                    patch.object(compact.subprocess, 'run'), \
                    patch.object(compact.urllib.request, 'urlopen', side_effect=response):
                if wrong_token:
                    with self.assertRaisesRegex(ValueError, 'SAME_PATH_NONDETERMINISM'):
                        compact.one_profile('D1', root, selected, {})
                else:
                    result = compact.one_profile('D1', root, selected, {})
                    self.assertEqual(result['request_count'], 20)
                    self.assertEqual(result['status'], 'COMPACT_PASS')
            self.assertEqual(server.returncode, 1)
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['status'], 'FAIL' if wrong_token else 'COMPACT_PASS')
            self.assertEqual(len((root/'requests.jsonl').read_text().splitlines()), count)

    def test_complete_and_owned_cleanup(self):
        self.exercise()

    def test_first_mismatch_retained_and_owned_cleanup(self):
        self.exercise(wrong_token=True)

    def test_frozen_s6(self):
        path = Path(__file__).resolve().parents[2] / 'round6/corpora/s6.json'
        self.assertEqual(set(compact.check_corpus(path)), set(compact.PROMPTS))


if __name__ == '__main__':
    unittest.main()
