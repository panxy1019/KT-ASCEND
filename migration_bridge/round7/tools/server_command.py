"""Round7-only launch adapter; frozen old launchers remain untouched."""
import json
import os
from pathlib import Path
import sys


def main():
    root, config = map(Path, sys.argv[1:3])
    assert str(root).startswith('/workspace/c64-dev/round7/')
    cfg = json.loads(config.read_text())
    assert cfg['workers'] == 64 and cfg['cpu_set'] == '128-191'
    assert cfg['deferred'] in (0,1)
    for k in list(os.environ):
        if k.startswith(('KT_R6_', 'KT_R61_', 'KT_R62_', 'KT_R63_',
                         'KT_R64_', 'KT_R65_', 'KT_R66_', 'KT_DEBUG_', 'KT_C64_CAPTURE')):
            os.environ.pop(k)
    args = ['-m','sglang.launch_server','--model-path','/workspace/models/DeepSeek-V2-Lite-604d5664',
        '--tokenizer-path','/workspace/results/r64-tokenizer-compat-v1','--host','127.0.0.1',
        '--port','31000','--device','npu','--tp-size','1','--dtype','bfloat16',
        '--context-length','8192','--max-total-tokens','8192','--chunked-prefill-size','4096',
        '--max-prefill-tokens','4096','--mem-fraction-static','0.55','--max-running-requests','1',
        '--random-seed','0','--disable-cuda-graph','--disable-custom-all-reduce','--skip-server-warmup',
        '--weight-loader-disable-mmap','--attention-backend','ascend','--sampling-backend','pytorch',
        '--disable-radix-cache','--kt-weight-path','/workspace/results/r65-assets-v1/all-moe-f32.gguf',
        '--kt-method','LLAMAFILE','--kt-cpuinfer','64','--kt-threadpool-count','2',
        '--kt-numa-nodes','4','5','--kt-gpu-experts-ratio','0.125',
        '--kt-max-deferred-experts-per-token',str(cfg['deferred']),
        '--kt-expert-placement-strategy','frequency','--init-expert-location',cfg['activation_path']]
    if cfg['dynamic']:
        args += ['--kt-enable-dynamic-expert-update','--kt-gpu-prefill-token-threshold',str(cfg['threshold'])]
    if cfg['recorder']:
        # Infinite buffer avoids silently retaining only the last 1000 forwards.
        args += ['--record-kt-gpu-expert-distribution','--expert-distribution-recorder-mode','stat',
                 '--expert-distribution-recorder-buffer-size','-1']
        os.environ['SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR'] = str(root/'recorder')
    if cfg['observer']:
        args += ['--forward-hooks',json.dumps([
            dict(name='round7-structure',target_modules=['model.layers.*.mlp'],
                 hook_factory='round7_hooks:make_structure',config=dict(output=str(root),profile=cfg['profile'])),
            dict(name='round7-route-pressure',target_modules=['model.layers.*.mlp.topk'],
                 hook_factory='round7_hooks:make_routes',config=dict(output=str(root),profile=cfg['profile']))])]
    (root/'resolved-command.json').write_text(json.dumps(dict(argv=args,config=cfg,
        env={k:os.environ.get(k) for k in ('PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES',
            'SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR','HF_HUB_OFFLINE')}),indent=2)+'\n')
    os.execv('/workspace/results/venv-r6/bin/python',['/workspace/results/venv-r6/bin/python',*args])


if __name__ == '__main__': main()
