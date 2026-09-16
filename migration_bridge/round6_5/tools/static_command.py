"""Explicit static launch argv; no production patch or inherited Deferred env."""
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    root, profile, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    if not str(root).startswith('/workspace/results/r65-') or workers not in (16,32):
        raise ValueError('LAUNCH_SCOPE')
    data=json.loads(profile.read_text())
    for key in list(os.environ):
        if key.startswith(('KT_R6_', 'KT_R61_', 'KT_R62_', 'KT_R63_', 'KT_R64_', 'KT_DEBUG_')):
            os.environ.pop(key)
    from build_ratio_profiles import digest
    logical={k:v for k,v in data.items() if k not in ('canonical_sha256','pt_sha256')}
    if digest(logical)!=data['canonical_sha256'] or data['deferred']!=0 or data['dynamic']:
        raise ValueError('PROFILE_DRIFT')
    pt=profile.with_suffix('.pt')
    if hashlib.sha256(pt.read_bytes()).hexdigest()!=data['pt_sha256']:
        raise ValueError('PLACEMENT_BYTES_DRIFT')
    args=['-m','sglang.launch_server','--model-path','/workspace/models/DeepSeek-V2-Lite-604d5664',
          '--tokenizer-path','/workspace/results/r64-tokenizer-compat-v1',
          '--host','127.0.0.1','--port','31000','--device','npu','--tp-size','1',
          '--dtype','bfloat16','--context-length','2048','--max-total-tokens','4096',
          '--chunked-prefill-size','2048','--max-prefill-tokens','2048',
          '--mem-fraction-static','0.55','--max-running-requests','1','--random-seed','0',
          '--disable-cuda-graph','--disable-custom-all-reduce','--skip-server-warmup',
          '--weight-loader-disable-mmap','--attention-backend','ascend','--sampling-backend','pytorch',
          '--disable-radix-cache']
    if data['total_npu_experts']!=26*64:
        gguf=Path('/workspace/results/r65-assets-v1/all-moe-f32.gguf')
        manifest=json.loads(gguf.with_suffix('.gguf.manifest.json').read_text())
        if manifest['layers']!=list(range(1,27)) or gguf.stat().st_size!=manifest['gguf_size']:
            raise ValueError('GGUF_INCOMPLETE')
        args += ['--kt-weight-path',str(gguf),'--kt-method','LLAMAFILE',
                 '--kt-cpuinfer',str(workers),'--kt-threadpool-count','1','--kt-numa-nodes','0',
                 '--kt-gpu-experts-ratio',str(data['gpu_experts_ratio']),
                 '--kt-max-deferred-experts-per-token','0','--kt-expert-placement-strategy','frequency',
                 '--init-expert-location',str(pt)]
    (root/'resolved-command.json').write_text(json.dumps(dict(argv=args,profile=data,workers=workers,
                                                            environment={k:os.environ.get(k) for k in (
                                                                'ASCEND_RT_VISIBLE_DEVICES','SGLANG_APPLY_CONFIG_BACKUP',
                                                                'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH',
                                                                'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),indent=2)+'\n')
    interpreter='/workspace/results/venv-r6/bin/python'
    os.execv(interpreter,[interpreter,*args])


if __name__=='__main__':main()
