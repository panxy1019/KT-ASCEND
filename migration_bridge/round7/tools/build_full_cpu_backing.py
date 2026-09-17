"""Private incremental native build for the reproduced CPU-backing defect.

Keep the original source, objects, Python runtime and binary immutable. Reuse
unchanged original arithmetic objects; rebuild the header-dependent bindings.
This is a build result only, not qualification. Execute after owned serving reaps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--header', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.output.resolve().is_relative_to(Path('/workspace/c64-dev/round7'))
    a.output.mkdir(parents=True, exist_ok=False)
    original_source = Path('/workspace/c64-dev/source/kt-kernel')
    original_build = Path('/workspace/c64-dev/native-build')
    original_native = Path('/workspace/c64-dev/runtime/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so')
    assert sha(original_source/'operators/llamafile/moe.hpp') == '17a7e720166b18ee6fb57e4de8ffc683972df08ca98a56c7cd763f11c5a642eb'
    assert sha(original_native) == 'a5497fa085d704d8f67484b3936eaf448080f5d445899d88e4e76bea4305f077'
    source, build, runtime = (a.output/x for x in ('source', 'build', 'runtime'))
    shutil.copytree(original_source, source)
    shutil.copy2(a.header, source/'operators/llamafile/moe.hpp')
    shutil.copytree(original_build, build)
    shutil.copytree('/workspace/c64-dev/runtime', runtime)
    flags = {}
    for line in (build/'CMakeFiles/kt_kernel_ext.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            flags[k] = shlex.split(v)
    obj = build/'CMakeFiles/kt_kernel_ext.dir/ext_bindings.cpp.o'
    compile_cmd = ['g++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
        '-DKT_C64_TEST_CAPTURE=1', '-c', str(source/'ext_bindings.cpp'), '-o', str(obj)]
    link = shlex.split((build/'CMakeFiles/kt_kernel_ext.dir/link.txt').read_text())
    native = runtime/'kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'
    link[link.index('-o')+1] = str(native)
    record = dict(status='BUILDING', original_native_sha256=sha(original_native),
        original_header_sha256=sha(original_source/'operators/llamafile/moe.hpp'),
        repaired_header_sha256=sha(a.header), native_path=str(native),
        commands=[compile_cmd, link], scope='PRIVATE_CPU_BACKING_LOAD_FIX_NO_ARITHMETIC_CHANGE')
    def save():
        (a.output/'build.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        subprocess.run(compile_cmd, check=True)
        subprocess.run(link, cwd=build, check=True)
        subprocess.run(['strip', str(native)], check=True)
        assert sha(original_native) == record['original_native_sha256'], 'ORIGINAL_BINARY_MUTATED'
        assert sha(original_source/'operators/llamafile/moe.hpp') == record['original_header_sha256'], 'ORIGINAL_SOURCE_MUTATED'
        record.update(status='BUILT_NOT_QUALIFIED', native_sha256=sha(native))
    except BaseException as error:
        record.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()
    print(json.dumps(dict(status=record['status'], native_sha256=record['native_sha256'])))


if __name__ == '__main__':
    main()
