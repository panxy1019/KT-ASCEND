"""Round7-only launch adapter; frozen old launchers remain untouched."""
import json
import hashlib
import os
from pathlib import Path
import sys


def direct_reference_args(args,cfg):
    """R64 oracle is the historical direct Ascend path, not KTEP with64 slots."""
    if not cfg.get('reference_r64',False):return args
    if cfg['dynamic'] or cfg['deferred']!=0 or cfg['observer'] or cfg['recorder']:
        raise ValueError('DIRECT_REFERENCE_FORBIDS_HYBRID_FEATURES_OR_HOOKS')
    result=list(args[:args.index('--kt-weight-path')])
    for flag,value in (('--context-length','2048'),('--max-total-tokens','4096'),
                       ('--chunked-prefill-size','2048'),('--max-prefill-tokens','2048')):
        result[result.index(flag)+1]=value
    return result


def main():
    root, config = map(Path, sys.argv[1:3])
    assert str(root).startswith('/workspace/c64-dev/round7/')
    cfg = json.loads(config.read_text())
    assert cfg['workers'] == 64 and cfg['cpu_set'] == '128-191'
    assert cfg['deferred'] in (0,1)
    if cfg.get('provenance'):
        provenance=json.loads(Path(cfg['provenance']).read_text())
        from runtime_selection import select_cpu_runtime,select_sglang_runtime
        sglang_runtime=select_sglang_runtime(cfg,provenance)
        if sglang_runtime:
            os.environ['PYTHONPATH']=sglang_runtime+os.pathsep+os.environ.get('PYTHONPATH','')
        cpu_runtime=select_cpu_runtime(cfg,provenance)
        if cpu_runtime:
            os.environ['PYTHONPATH']=cpu_runtime+os.pathsep+os.environ.get('PYTHONPATH','')
        checked={}
        for name,info in provenance['files'].items():
            digest=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            assert digest==info['sha256'],f'IMPORTED_ARTIFACT_DRIFT:{name}'
            checked[name]=digest
        gguf=Path(provenance['gguf']['path'])
        assert gguf.stat().st_size==provenance['gguf']['size'],'GGUF_SIZE_DRIFT'
        (root/'artifact-preflight.json').write_text(json.dumps(dict(checked_sha256=checked,
            frozen_provenance_sha256=hashlib.sha256(Path(cfg['provenance']).read_bytes()).hexdigest(),
            gguf_size=gguf.stat().st_size,gguf_sha256=provenance['gguf']['sha256'],
            gguf_hash_scope='Initial full hash retained; RO artifact is not rehashed on each server restart.'),indent=2)+'\n')
    elif cfg.get('cpu_runtime_root') or cfg.get('sglang_runtime_root'):
        raise ValueError('PRIVATE_RUNTIME_REQUIRES_PROVENANCE')
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
        '--kt-numa-nodes','4','5','--kt-gpu-experts-ratio',('1.0' if cfg.get('reference_r64',False) else '0.125'),
        '--kt-max-deferred-experts-per-token',str(cfg['deferred']),
        '--kt-expert-placement-strategy','frequency','--init-expert-location',cfg['activation_path']]
    args=direct_reference_args(args,cfg)
    if cfg.get('deferred_trace',False):
        assert cfg['observer'],'DEFERRED_TRACE_FORBIDDEN_IN_TIMED_PERFORMANCE'
        os.environ['KT_DEBUG_DEFERRED_TRACE_FILE']=str(root/'deferred-trace.jsonl')
        os.environ['KT_DEBUG_DEFERRED_TRACE_LIMIT']='1000000'
    if cfg['dynamic']:
        args += ['--kt-enable-dynamic-expert-update','--kt-gpu-prefill-token-threshold',str(cfg['threshold'])]
    if cfg['recorder']:
        # Infinite buffer avoids silently retaining only the last 1000 forwards.
        args += ['--record-kt-gpu-expert-distribution','--expert-distribution-recorder-mode','stat',
                 '--expert-distribution-recorder-buffer-size','-1']
        os.environ['SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR'] = str(root/'recorder')
    hooks=[]
    if cfg['observer']:
        hooks += [
            dict(name='round7-structure',target_modules=['model.layers.*.mlp'],
                 hook_factory='round7_hooks:make_structure',config=dict(output=str(root),profile=cfg['profile'],dynamic_audit=cfg['dynamic'],deferred_audit=cfg['deferred']==1)),
            dict(name='round7-route-pressure',target_modules=['model.layers.*.mlp.topk'],
                 hook_factory='round7_hooks:make_routes',config=dict(output=str(root),profile=cfg['profile']))]
    if cfg.get('dynamic_telemetry',False):
        assert cfg['dynamic'] and not cfg['observer'],'LIGHT_TELEMETRY_MUST_BE_SEPARATE_FROM_HEAVY_AUDIT'
        hooks.append(dict(name='round7-dynamic-telemetry',target_modules=['model.layers.*.mlp'],
            hook_factory='dynamic_telemetry:make_hook',config=dict(output=str(root))))
    if hooks:args += ['--forward-hooks',json.dumps(hooks)]
    (root/'resolved-command.json').write_text(json.dumps(dict(argv=args,config=cfg,
        env={k:os.environ.get(k) for k in ('PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES',
            'SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR','HF_HUB_OFFLINE',
            'KT_DEBUG_DEFERRED_TRACE_FILE','KT_DEBUG_DEFERRED_TRACE_LIMIT')}),indent=2)+'\n')
    os.execv('/workspace/results/venv-r6/bin/python',['/workspace/results/venv-r6/bin/python',*args])


if __name__ == '__main__': main()
