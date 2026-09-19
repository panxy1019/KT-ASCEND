"""Owned diagnostic chain: ready -> restart exact -> 32-prompt screen."""
import json
import argparse
import subprocess
import sys
import time
from pathlib import Path
import requests

root=Path('/workspace/round8')
p=argparse.ArgumentParser();p.add_argument('--profile',choices=['q64','q32'],default='q64');a=p.parse_args()
for _ in range(120):
    try:
        if requests.get('http://127.0.0.1:18400/health',timeout=5).status_code==200:break
    except requests.RequestException:pass
    time.sleep(5)
else:raise RuntimeError('Q64_RESTART_SERVER_NOT_READY')
subprocess.run([sys.executable,str(root/'tools/q64_compact.py'),'--profile',a.profile,'--phase','restart'],check=True)
assert json.loads((root/f'{a.profile}-compact-v1/restart-summary.json').read_text())['status']=='PASS'
print('RESTART_PASS_STARTING_32_DISTINCT_PROMPT_SCREEN',flush=True)
subprocess.run([sys.executable,str(root/'tools/q64_generation_screen.py'),'--profile',a.profile],check=True)
print('SYSTEM_SCREEN_COMPLETE_NOT_QUALITY_QUALIFIED',flush=True)
