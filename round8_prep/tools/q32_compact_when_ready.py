"""Run only Q32 same-process compact after actual Q32 structure is observed."""
import json
import subprocess
import sys
import time
from pathlib import Path
import requests

root=Path('/workspace/round8')
for _ in range(180):
    try:
        if requests.get('http://127.0.0.1:18400/health',timeout=5).status_code==200:break
    except requests.RequestException:pass
    time.sleep(5)
else:raise RuntimeError('Q32_SERVER_NOT_READY')
rows=[json.loads(x) for x in (root/'q32-launch-v1/structure.jsonl').read_text().splitlines()]
assert {r['layer'] for r in rows}==set(range(40))
assert all(r['slots']==32 and r['shared_exactly_once_oracle'] for r in rows)
subprocess.run([sys.executable,str(root/'tools/q64_compact.py'),'--profile','q32','--phase','same'],check=True)
print('Q32_SAME_PROCESS_PASS_RESTART_IS_NEXT_NOT_QUALITY',flush=True)
