"""Round7 output-scope adapter; retain original process guard and subreaping."""
import ctypes
import json
import os
import runpy
import sys
import time
from worker_topology import output_root


def main():
    output=output_root(sys.argv[sys.argv.index('--output')+1])
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0):
        raise OSError('SUBREAPER_FAILED')
    try:
        runpy.run_path('/workspace/results/round6_2-prequalification/tools/run_owned_process.py',run_name='__main__')
    finally:
        reaped=[];deadline=time.monotonic()+15;complete=False
        while time.monotonic()<deadline:
            try:pid,status=os.waitpid(-1,os.WNOHANG)
            except ChildProcessError:
                complete=True;break
            if pid:reaped.append([pid,status])
            else:time.sleep(.1)
        if output.exists():
            (output/'reaping.json').write_text(json.dumps(dict(complete=complete,reaped=reaped))+'\n')
        if not complete:raise RuntimeError('OWNED_DESCENDANTS_NOT_REAPED')


if __name__=='__main__':main()
