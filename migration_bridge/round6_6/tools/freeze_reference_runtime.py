"""Supplement independent CPU reference provenance; no install or NPU work."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    package=Path(torch.__file__).parent
    linked=subprocess.check_output(['ldd',str(package/'lib/libtorch_cpu.so')],text=True)
    files=set((package/'lib').glob('*.so*'))|set((package.parent/'torch.libs').glob('*.so*'))
    for line in linked.splitlines():
        for token in line.split():
            if token.startswith('/') and Path(token).is_file():files.add(Path(token).resolve())
    files=sorted(files)
    rows=[]
    for path in files:
        if not path.is_file():continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        rows.append(dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest()))
    report=dict(version='r66-reference-runtime-supplement-v2',time_unix=time.time(),
        python=platform.python_version(),machine=platform.machine(),torch_version=torch.__version__,
        torch_npu_version=importlib.metadata.version('torch-npu'),torch_package=str(package),
        torch_build_config=torch.__config__.show(),ldd_libtorch_cpu=linked,files=rows,
        canonical_implementation='CPU FP32 torch.nn.functional.linear and silu; explicit BF16 boundary controls separate',
        scope='Post-B0 build provenance supplement; not a replacement for original B0 result',
        production_backend_replaced=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(files=len(rows),torch_version=report['torch_version'],package=str(package))))


if __name__=='__main__':main()
