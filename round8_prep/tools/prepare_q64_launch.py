"""Freeze static Q64 command and actual imported artifact hashes; no server start."""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu
import kt_kernel
from kt_kernel.experts_base import generate_gpu_experts_masks


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path('/workspace/round8')
    out=root/'q64-launch-v2'
    out.mkdir(exist_ok=False)
    profile=json.loads((root/'q64_profile.json').read_text())
    assert profile['resident']==64 and len(profile['layers'])==40
    mask=torch.zeros(40,256,dtype=torch.bool)
    for row in profile['layers']:
        assert len(row['npu_ids'])==64
        assert set(row['npu_ids']).isdisjoint(row['cpu_ids'])
        assert set(row['npu_ids'])|set(row['cpu_ids'])==set(range(256))
        mask[row['layer'],row['npu_ids']]=True
    counts=mask.to(torch.int64).unsqueeze(0)
    assert torch.equal(generate_gpu_experts_masks(counts[0].float(),40*64),mask)
    torch.save({'logical_count':counts},out/'uniform-mask.pt')
    # Frequency loader is used only as an exact frozen-mask transport, no data fitting.
    args=[sys.executable,'-m','sglang.launch_server',
        '--model-path',str(root/'model-sample'),'--tokenizer-path',str(root/'model-sample'),
        '--host','127.0.0.1','--port','18400','--device','npu','--tp-size','1',
        '--dtype','bfloat16','--context-length','4096','--max-total-tokens','4096',
        '--chunked-prefill-size','512','--max-prefill-tokens','512',
        '--mem-fraction-static','0.70','--max-running-requests','1','--random-seed','0',
        '--disable-cuda-graph','--disable-custom-all-reduce','--skip-server-warmup',
        '--weight-loader-disable-mmap','--attention-backend','ascend',
        '--sampling-backend','pytorch','--disable-radix-cache',
        '--disable-shared-experts-fusion','--language-only',
        '--kt-weight-path',str(root/'cpu-experts-f32-v1'),'--kt-method','LLAMAFILE',
        '--kt-cpuinfer','64','--kt-threadpool-count','2','--kt-numa-nodes','4','5',
        '--kt-num-gpu-experts','64','--kt-max-deferred-experts-per-token','0',
        '--kt-expert-placement-strategy','frequency',
        '--init-expert-location',str(out/'uniform-mask.pt')]
    hooks=[dict(name='qwen-structure',target_modules=['model.layers.*.mlp'],
                hook_factory='qwen_bringup_hooks:make_structure',
                config=dict(output=str(out),profile=str(root/'q64_profile.json'))),
           dict(name='qwen-routes',target_modules=['model.layers.*.mlp.topk'],
                hook_factory='qwen_bringup_hooks:make_routes',
                config=dict(output=str(out),profile=str(root/'q64_profile.json')))]
    args+=['--forward-hooks',json.dumps(hooks)]
    env={'PYTHONPATH':str(root/'sglang-python')+':'+str(root/'tools')+':/workspace/kt-src',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'ASCEND_RT_VISIBLE_DEVICES':'0','TRITON_CACHE_DIR':str(root/'triton-cache'),
         'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
         'SGLANG_APPLY_CONFIG_BACKUP':'none'}
    hashes={}
    for tree in [root/'sglang-python',Path(kt_kernel.__file__).parent]:
        for p in sorted(tree.rglob('*')):
            if p.is_file() and (p.suffix=='.py' or '.so' in p.name):hashes[str(p)]=sha(p)
    hashes[str(root/'tools/qwen_bringup_hooks.py')]=sha(root/'tools/qwen_bringup_hooks.py')
    result=dict(argv=args,env=env,profile_sha256=sha(root/'q64_profile.json'),
                mask_sha256=sha(out/'uniform-mask.pt'),code_hashes=hashes,
                torch=torch.__version__,torch_npu=torch_npu.__version__,
                python=sys.version,kt_import=kt_kernel.__file__,
                graph=False,deferred=False,dynamic=False,batch=1,tp=1,
                purpose='BRINGUP_NOT_QUALITY',quality_authorized=False)
    (out/'resolved-command.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Q64_MASK_AND_COMMAND_FROZEN_NOT_STARTED')


if __name__=='__main__':main()
