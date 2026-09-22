"""Offline A3 provenance/profile preparation, no inference or quality selection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    import torch
    import torch_npu
    root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
    paths=[Path('/workspace/c64-dev/runtime/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'),
        Path('/workspace/c64-dev/runtime/kt_kernel/utils/llamafile.py'),
        Path('/workspace/kt-src/third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py'),
        Path('/workspace/kt-src/third_party/sglang/python/sglang/srt/models/deepseek_v2.py'),
        Path('/workspace/kt-src/third_party/sglang/python/sglang/srt/server_args.py'),
        Path('/workspace/results/r65-r67-serial-v5/cmmlu.json'),
        Path('/workspace/results/r65-r67-serial-v5/R64/quality.json')]
    report=dict(time_unix=time.time(),local_base='bc2e5d53e05d97db83b40c09a6f80cddff0549ab',
        actual_parent='e3084d309d6f39b8a9f7185d36ac27e4b77526d7',
        actual_child='eaa58ccea4ee372b8b207dd16f398eafe3da37ff',
        model_revision='604d5664dddd88a0433dbae533b7fe9472482de0',
        torch=torch.__version__,torch_npu=torch_npu.__version__,python=sys.version,
        cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),
        physical_npu=2,logical_npu=0,container='kt-r67-c64-development-v1',
        host_modified=False,business_containers_modified=False,
        files={str(x):dict(sha256=sha(x),size=x.stat().st_size) for x in paths})
    native=paths[0]
    assert report['files'][str(native)]['sha256']=='a5497fa085d704d8f67484b3936eaf448080f5d445899d88e4e76bea4305f077'
    assert report['files'][str(paths[-2])]['sha256']=='bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04'
    old=json.loads(Path('/workspace/c64-dev/execution-freeze-v11.json').read_text())
    report['retained_execution_freeze']=old
    (root/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    gguf=Path('/workspace/results/r65-assets-v1/all-moe-f32.gguf')
    gguf_hash=sha(gguf)
    assert gguf_hash=='20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2'
    report['gguf']=dict(path=str(gguf),size=gguf.stat().st_size,sha256=gguf_hash)
    (root/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    profiles=root/'profiles'; profiles.mkdir(exist_ok=True)
    mask=torch.zeros((1,27,64),dtype=torch.int64)
    mask[:,1:,::8]=1
    # Dense layer0 is zero in the frequency input and is restored by KT after
    # top-K; it must never consume any of the 208 routed slots.
    torch.save(dict(logical_count=mask),profiles/'uniform.pt')
    data=dict(name='UNIFORM_R8_C64',total_resident=208,
        rows=[dict(layer=l,npu_ids=list(range(0,64,8)),cpu_ids=[e for e in range(64) if e%8]) for l in range(1,27)],
        activation_sha256=sha(profiles/'uniform.pt'))
    (profiles/'uniform.json').write_text(json.dumps(data,indent=2)+'\n')
    cfg=dict(workers=64,cpu_set='128-191',deferred=0,dynamic=False,threshold=0,
        recorder=True,observer=True,profile=str(profiles/'uniform.json'),activation_path=str(profiles/'uniform.pt'))
    (root/'uniform-dev-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED',gguf=gguf_hash,profile=str(profiles/'uniform.json'))),flush=True)


if __name__=='__main__': main()
