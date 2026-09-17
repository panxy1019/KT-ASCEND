"""Exercise real command construction without starting a model process."""
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
from common import write
from serial_session import validate_launch

base=Path('/workspace/c64-dev/launcher-audit-v1');base.mkdir(exist_ok=False)
tools=Path(__file__).parent
results=[]
for tp in (1,2):
    for observer in (0,1):
        root=base/f'tp{tp}-observer{observer}';root.mkdir()
        env=dict(C64_TP=str(tp),KT_R67_CPU_SET='128-191' if tp==2 else '128-159',KT_R67_NPU_ID='0')
        with patch.dict(os.environ,env),patch.object(sys,'argv',['serial_command.py',str(root),
                '/workspace/results/r65-r67-serial-v5/profiles/R16.json',str(observer)]),patch('os.execv') as execute:
            runpy.run_path(str(tools/'serial_command.py'),run_name='__main__')
            execute.assert_called_once()
            result=validate_launch(root,bool(observer))
            results.append(dict(tp=tp,observer=bool(observer),result=result))
write(base/'result.json',dict(status='PASS',cases=results,model_executed=False))
print(json.dumps(dict(status='PASS',cases=len(results))))
