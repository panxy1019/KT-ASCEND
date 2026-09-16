"""Read-only runtime source/installed-package provenance; no imports or NPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/workspace/kt-src');child=root/'third_party/sglang'
    report=dict(time_unix=time.time(),source_read_only=True,repositories={},files=[])
    for name,path in [('parent',root),('sglang',child),('llama',root/'third_party/llama.cpp')]:
        def git(*args):return subprocess.check_output(['git','-C',str(path),*args]).decode()
        report['repositories'][name]=dict(head=git('rev-parse','HEAD').strip(),status=git('status','--short'),
            diff_sha256=hashlib.sha256(git('diff','--binary').encode()).hexdigest())
    installed=Path('/workspace/results/venv-r6/lib/python3.11/site-packages')
    paths=set((installed/'kt_kernel').rglob('*.py'))|set(installed.glob('**/*cpuinfer*.so'))
    for relative in ['python/sglang/srt/layers/moe/kt_ep_wrapper.py',
        'python/sglang/srt/models/deepseek_v2.py','python/sglang/srt/model_executor/model_runner.py',
        'python/sglang/srt/model_executor/hook_manager.py','python/sglang/srt/layers/moe/topk.py']:
        paths.add(child/relative)
    paths.update((child/'python/sglang/srt/debug_utils').glob('*.py'))
    paths.add(root/'kt-kernel/python/experts_base.py')
    paths.update((installed/'kt_kernel-0.7.0.post4.dist-info').glob('*'))
    for path in sorted(paths):
        if path.is_file():report['files'].append(hash_file(path))
    report['scope']='INSTALLED_KT_PYTHON_CPU_EXTENSION_AND_SGLANG_INTEGRATION_OVERLAYS'
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')


if __name__=='__main__':main()
