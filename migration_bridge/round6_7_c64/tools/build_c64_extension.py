"""Private incremental build; original arithmetic objects are never overwritten."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

root=Path('/workspace/c64-dev')
base=Path('/workspace/results/migration-cpu-build-v1/build')
build=root/'native-build'
source=root/'source/kt-kernel'
shutil.copytree('/workspace/kt-src/kt-kernel',source,dirs_exist_ok=True)
overlay=root/'overlay/kt-kernel'
for rel in ('cpu_backend/worker_pool.cpp','cpu_backend/worker_pool.h','operators/moe-tp.hpp',
            'operators/llamafile/moe.hpp','operators/llamafile/tp_slices.hpp',
            'operators/llamafile/c64_test_capture.hpp','python/utils/llamafile.py'):
    shutil.copy2(overlay/rel,source/rel)
shutil.copytree(base,build,dirs_exist_ok=True)
flags={}
for line in (build/'CMakeFiles/kt_kernel_ext.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
commands=[]
for unit in ('ext_bindings.cpp','cpu_backend/worker_pool.cpp'):
    obj=build/'CMakeFiles/kt_kernel_ext.dir'/(unit+'.o')
    cmd=['g++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
         '-DKT_C64_TEST_CAPTURE=1','-c',str(source/unit),'-o',str(obj)]
    commands.append(cmd);subprocess.run(cmd,check=True)
runtime=root/'runtime/kt_kernel'
shutil.copytree('/workspace/results/venv-r6/lib/python3.11/site-packages/kt_kernel',runtime,dirs_exist_ok=True)
shutil.copy2(source/'python/utils/llamafile.py',runtime/'utils/llamafile.py')
link=shlex.split((build/'CMakeFiles/kt_kernel_ext.dir/link.txt').read_text())
out=runtime/'kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'
link[link.index('-o')+1]=str(out)
commands.append(link);subprocess.run(link,cwd=build,check=True)
subprocess.run(['strip',str(out)],check=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=dict(status='BUILT_NOT_QUALIFIED',native_sha256=sha(out),native_path=str(out),
            commands=commands,source_sha256={str(p.relative_to(overlay)):sha(p) for p in overlay.rglob('*') if p.is_file()})
(root/'c64-build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
