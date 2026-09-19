"""Derive diagnostic Q32 Dynamic candidates without altering model arithmetic."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace/round8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=json.loads((ROOT/'q32-perf-launch-v2/resolved-command.json').read_text())
    for threshold in (256,512,1024):
        name=f'q32-dynamic-{threshold}-v1';output=ROOT/name;output.mkdir(exist_ok=False)
        config=json.loads(json.dumps(source));args=config['argv'][:]
        nodes=(0,1) if threshold in (256,1024) else (2,3)
        numa_index=args.index('--kt-numa-nodes')
        assert args[numa_index+1:numa_index+3]==['4','5']
        args[numa_index+1:numa_index+3]=[str(n) for n in nodes]
        config['env']['TRITON_CACHE_DIR']=f'/workspace/round8/triton-cache-dynamic-{threshold}'
        args.extend(['--kt-gpu-prefill-token-threshold',str(threshold),
                     '--kt-enable-dynamic-expert-update','--forward-hooks',json.dumps([dict(
                         name='qwen-dynamic-structure',target_modules=['model.layers.*.mlp'],
                         hook_factory='qwen_dynamic_audit:make_structure',config=dict(
                             output=f'/workspace/round8/{name}',
                             profile='/workspace/round8/q32_profile.json'))])])
        config.update(argv=args,dynamic=True,dynamic_threshold=threshold,
                      numa_nodes=list(nodes),c64_definition='2x32',
                      purpose='DEV_ONLY_DYNAMIC_SYSTEM_QUALIFICATION',quality_authorized=False,
                      source_command_sha256=sha(ROOT/'q32-perf-launch-v2/resolved-command.json'))
        audit=ROOT/'tools/qwen_dynamic_audit.py'
        config['code_hashes'][str(audit)]=sha(audit)
        (output/'resolved-command.json').write_text(json.dumps(config,indent=2)+'\n')
        print(name,sha(output/'resolved-command.json'))


if __name__=='__main__':main()
