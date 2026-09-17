"""Relink the frozen extension, changing only the WorkerPool translation unit."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

root = Path('/workspace/c64/r8-c32-v2')
root.mkdir(exist_ok=True)
base = Path('/workspace/results/migration-cpu-build-v1/build')
build = root / 'build'
shutil.copytree(base, build, dirs_exist_ok=True)
original = Path('/workspace/results/venv-r6/lib/python3.11/site-packages/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so')
assert sha(original) == '0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895'
link = shlex.split((build/'CMakeFiles/kt_kernel_ext.dir/link.txt').read_text())
oi = link.index('-o') + 1
link[oi] = str(root/'original-relinked.so')
subprocess.run(link, cwd=build, check=True)
subprocess.run(['strip',str(root/'original-relinked.so')],check=True)
assert sha(root/'original-relinked.so') == sha(original), 'FROZEN_RELINK_NOT_IDENTICAL'
source = Path('/workspace/c64/v4/kt-kernel/cpu_backend')
flags = {}
for line in (build/'CMakeFiles/kt_kernel_ext.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v = line.split(' = ',1); flags[k] = shlex.split(v)
obj = build/'CMakeFiles/kt_kernel_ext.dir/cpu_backend/worker_pool.cpp.o'
compile_cmd = ['/usr/bin/g++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
               '-o',str(obj),'-c',str(source/'worker_pool.cpp')]
subprocess.run(compile_cmd,check=True)
output = root/original.name
link[oi] = str(output)
subprocess.run(link,cwd=build,check=True)
subprocess.run(['strip',str(output)],check=True)
record = dict(original_sha256=sha(original), relink_control_exact=True,
              candidate_sha256=sha(output), output=str(output),
              changed_translation_unit='cpu_backend/worker_pool.cpp',
              source_sha256={p.name:sha(p) for p in source.glob('worker_pool.*')},
              compile_command=compile_cmd,link_command=link,
              tp_count=1,workers=32,arithmetic_objects='UNCHANGED')
(root/'build-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
