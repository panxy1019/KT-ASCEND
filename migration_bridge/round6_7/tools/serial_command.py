"""R6.7 test-only launcher, same frozen production flags as R6.5."""
import json
import os
from pathlib import Path
import sys
from common import sha, validate_profile, write


def main():
    root, profile, observer = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    if not str(root).startswith('/workspace/results/r65-r67-') or observer not in (0, 1):
        raise ValueError('LAUNCH_SCOPE')
    for key in list(os.environ):
        if key.startswith(('KT_R6_', 'KT_R61_', 'KT_R62_', 'KT_R63_', 'KT_R64_',
                           'KT_R65_', 'KT_R66_', 'KT_DEBUG_')):
            os.environ.pop(key)
    data = json.loads(profile.read_text()); validate_profile(data)
    if sha(profile.with_suffix('.pt')) != data['pt_sha256']:
        raise ValueError('PLACEMENT_BYTES_DRIFT')
    args = ['-m', 'sglang.launch_server', '--model-path', '/workspace/models/DeepSeek-V2-Lite-604d5664',
        '--tokenizer-path', '/workspace/results/r64-tokenizer-compat-v1', '--host', '127.0.0.1',
        '--port', '31000', '--device', 'npu', '--tp-size', '1', '--dtype', 'bfloat16',
        '--context-length', '2048', '--max-total-tokens', '4096', '--chunked-prefill-size', '2048',
        '--max-prefill-tokens', '2048', '--mem-fraction-static', '0.55', '--max-running-requests', '1',
        '--random-seed', '0', '--disable-cuda-graph', '--disable-custom-all-reduce', '--skip-server-warmup',
        '--weight-loader-disable-mmap', '--attention-backend', 'ascend', '--sampling-backend', 'pytorch',
        '--disable-radix-cache']
    if data['resident'] != 64:
        args += ['--kt-weight-path', '/workspace/results/r65-assets-v1/all-moe-f32.gguf',
            '--kt-method', 'LLAMAFILE', '--kt-cpuinfer', '32', '--kt-threadpool-count', '1',
            '--kt-numa-nodes', '0', '--kt-gpu-experts-ratio', str(data['gpu_experts_ratio']),
            '--kt-max-deferred-experts-per-token', '0', '--kt-expert-placement-strategy', 'frequency',
            '--init-expert-location', str(profile.with_suffix('.pt'))]
        # "frequency" here loads a frozen binary uniform mask. No frequency
        # fitting or dynamic update is performed; the actual mask is audited.
    if observer:
        tools = str(Path(__file__).parent)
        os.environ['PYTHONPATH'] = tools + ':/workspace/r65-tools/migration_bridge/round6_5/tools:' + os.environ['PYTHONPATH']
        args += ['--forward-hooks', json.dumps([
            dict(name='r67-structure-and-operands', target_modules=['model.layers.*.mlp'],
                 hook_factory='r67_hooks:make_mlp_hook', config=dict(profile=str(profile), output=str(root))),
            dict(name='r67-routes', target_modules=['model.layers.*.mlp.topk'],
                 hook_factory='r67_hooks:make_route_hook', config=dict(output=str(root / 'routing.jsonl')))])]
    write(root / 'resolved-command.json', dict(argv=args, profile_sha256=sha(profile),
          observer=bool(observer), workers=32, cpu_set='0-31', env={k:os.environ.get(k) for k in (
              'ASCEND_RT_VISIBLE_DEVICES', 'PYTHONPATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
              'SGLANG_APPLY_CONFIG_BACKUP', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}))
    python = '/workspace/results/venv-r6/bin/python'
    os.execv(python, [python, *args])


if __name__ == '__main__': main()
