"""Reuse original capture integrity checks with scoped diagnostic launcher."""
import os
from pathlib import Path
import runpy
import subprocess
import sys

tools=Path('/workspace/r65-tools/migration_bridge/round6_5/tools')
sys.path.insert(0,str(tools))
original=subprocess.Popen
def scoped_popen(args,*extra,**kwargs):
    args=list(args)
    old=str(tools/'launch_static.sh')
    if old in args:
        args[args.index(old)]=str(Path(__file__).parent/'launch_diagnostic.sh')
    return original(args,*extra,**kwargs)
subprocess.Popen=scoped_popen
os.environ['KT_R65_STAGEWISE']='1'
runpy.run_path(str(tools/'capture_static_logits.py'),run_name='__main__')
