#!/workspace/results/venv-r6/bin/python
"""Explicit destination worker profile; preserve original token capacity."""
import json
import os
from pathlib import Path
import sys
from capacity_python_adapter import adapt_workers

workers = int(os.environ['KT_R64_CPU_WORKERS'])
assert sys.argv[1:3] == ['-m','sglang.launch_server']
argv = adapt_workers(sys.argv[1:], workers)
record = Path(os.environ['KT_R64_RESOLVED_COMMAND'])
assert str(record).startswith('/workspace/results/r64-')
with record.open('x') as output:
    json.dump(dict(original=sys.argv[1:], resolved=argv, cpu_workers=workers,
                   capacity_changed=False, production_code_changed=False),output,indent=2)
interpreter='/workspace/results/venv-r6/bin/python'
os.execv(interpreter,[interpreter,*argv])
