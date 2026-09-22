import json
from pathlib import Path
import subprocess
import sys
import unittest


class ReapingTests(unittest.TestCase):
    def test_own_orphaned_grandchild_is_reaped(self):
        code = '''
import json, os, time
from run_owned_process import enable_subreaper, reap_children
enable_subreaper()
pid = os.fork()
if pid == 0:
    child = os.fork()
    if child == 0:
        time.sleep(.05)
        os._exit(0)
    os._exit(0)
result = reap_children()
print(json.dumps(result))
'''
        answer = subprocess.check_output([sys.executable,"-c",code],
                                        cwd=Path(__file__).parent, text=True, timeout=15)
        result = json.loads(answer)
        self.assertEqual(result["status"],"OWNED_DESCENDANTS_REAPED")
        self.assertEqual(len(result["reaped"]),2)


if __name__ == "__main__":
    unittest.main()
