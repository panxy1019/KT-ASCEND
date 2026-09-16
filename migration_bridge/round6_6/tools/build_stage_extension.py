"""Recompile only diagnostic bindings, retaining frozen production object files.

No install, no existing build edits. Source copy and patched header supplied externally.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    entries=json.loads((a.build/'compile_commands.json').read_text())
    row=next(x for x in entries if x['file'].endswith('/ext_bindings.cpp'))
    args=shlex.split(row['command'])
    args=[x.replace('/workspace/kt-src/kt-kernel',str(a.source)) for x in args]
    obj=a.output/'ext_bindings.cpp.o'
    args[args.index('-o')+1]=str(obj)
    args.insert(1,'-DKT_R66_STAGE_CAPTURE=1')
    link=shlex.split((a.build/'CMakeFiles/kt_kernel_ext.dir/link.txt').read_text())
    output=a.output/'kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'
    link[link.index('-o')+1]=str(output)
    link=[str(obj) if x=='CMakeFiles/kt_kernel_ext.dir/ext_bindings.cpp.o' else x for x in link]
    record=dict(compile=args,link=link,cwd=str(a.build),source=str(a.source),
        header_sha256=sha(a.source/'operators/llamafile/moe.hpp'),production_install_modified=False)
    (a.output/'commands.json').write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(args,cwd=a.build,check=True)
    subprocess.run(link,cwd=a.build,check=True)
    record['library_sha256']=sha(output)
    (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
