"""Run the candidate SGLang server after separately recorded resource admission.

This is an entry point to SGLang's own server, scheduler and model runner.
It neither grants resource admission nor claims model/quality qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require_capacity_device_gate(capacity, candidate, identity, payload, native_sha256=None):
    import os
    kwargs = dict(storage=candidate.storage, workers=candidate.workers,
                  source_identity=identity['source_identity'],
                  layer0_sha256=identity['layer0_sha256'])
    if capacity == 32:
        from sglang.srt.layers.round9.q32_gate import require_q32_device_gate
        return require_q32_device_gate(os.environ.get('ROUND9_DYNAMIC_DEVICE_GATE'),
            payload['device_gate'], native_sha256=native_sha256, **kwargs)
    if capacity != 64: raise RuntimeError('unsupported resident capacity')
    if candidate.placement == 'request_boundary':
        from sglang.srt.layers.round9.candidate_config import require_dynamic_device_gate
        return require_dynamic_device_gate(os.environ.get('ROUND9_DYNAMIC_DEVICE_GATE'),
            payload['device_gate'], **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--resolved-profile', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--admission', type=Path)
    a = p.parse_args()
    from sglang.srt.server_args import ServerArgs
    values = json.loads(a.profile.read_text())
    args = ServerArgs(**values)
    for key in ('kt_cpuinfer','kt_threadpool_count','kt_num_gpu_experts',
                'max_running_requests','max_total_tokens','max_mamba_cache_size',
                'disable_cuda_graph','disable_radix_cache','disable_overlap_schedule',
                'enable_multimodal','page_size','chunked_prefill_size','kt_method',
                'kt_enable_dynamic_expert_update','kt_max_deferred_experts_per_token'):
        if getattr(args, key) != values[key]:
            raise RuntimeError(f'SGLang changed frozen field {key}')
    receipt = {'requested': values, 'resolved': {key:getattr(args,key) for key in values},
               'profile_sha256':hashlib.sha256(a.profile.read_bytes()).hexdigest()}
    if a.resolved_profile.exists():
        raise FileExistsError(a.resolved_profile)
    a.resolved_profile.write_text(json.dumps(receipt,indent=2)+'\n')
    if a.validate_only:
        print('PASS_SERVER_ARGS_ONLY', flush=True)
        return
    if a.admission is None:
        raise RuntimeError('a separately verified resource admission receipt is required')
    admission = json.loads(a.admission.read_text())
    import time
    if (admission.get('status') != 'CANDIDATE_READY_FOR_MONITORED_FULL_MODEL_RUN'
            or admission.get('scope') != 'CANDIDATE_EXCLUSIVE_FULL_MODEL_PREFLIGHT'
            or admission.get('profile_sha256') != receipt['profile_sha256']
            or not 0 <= time.time()-admission.get('sampled_at',0) <= 30):
        raise RuntimeError('missing, stale or unrelated resource admission')
    import os
    if (admission.get('resource_mode')!='exclusive' or admission.get('training_allowance_mib')!=0 or
            admission.get('allocator_limit_mib')!=43008 or admission.get('owned_limit_mib')!=45056 or
            admission.get('model_hbm_budget_mib')!=46080 or admission.get('reserve_hbm_mib')!=8192):
        raise RuntimeError('candidate exclusive allocator/process budget mismatch')
    os.environ['ROUND9_RESOURCE_MODE']='exclusive'
    from sglang.srt.layers.round9.candidate_config import load_candidate_profile
    from sglang.srt.layers.round9.candidate_bank import bind_verified_banks
    candidate,_,candidate_sha = load_candidate_profile(os.environ.get('ROUND9_CANDIDATE_PROFILE'))
    identity = bind_verified_banks(os.environ['ROUND9_CANDIDATE_PROFILE'],args.kt_weight_path,args.model_path)
    if admission.get('candidate_profile_sha256')!=candidate_sha or admission.get('bank_identity')!=identity:
        raise RuntimeError('candidate identity differs from fresh admission')
    payload_path=a.admission.parent/'payload-sha256.json'
    if hashlib.sha256(payload_path.read_bytes()).hexdigest()!=admission.get('payload_sha256'):
        raise RuntimeError('candidate payload receipt mismatch')
    payload=json.loads(payload_path.read_text())
    framework = None
    framework_path=os.environ.get('ROUND9_FRAMEWORK_PROFILE')
    if framework_path or admission.get('framework_profile_sha256'):
        from sglang.srt.layers.round9.framework_config import load_framework
        framework = load_framework()
        if (not framework_path or framework is None or
                hashlib.sha256(Path(framework_path).read_bytes()).hexdigest()!=admission.get('framework_profile_sha256') or
                admission.get('framework_profile_sha256') != payload.get('framework_profile')):
            raise RuntimeError('framework profile differs from fresh admission')
    capacity = framework['resident_capacity'] if framework is not None else 64
    if args.kt_num_gpu_experts != capacity:
        raise RuntimeError('server capacity differs from admitted framework profile')
    def tree(root):
        return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if tree(a.admission.parent/'source')!=payload['source']:
        raise RuntimeError('frozen candidate source differs from admission')
    import sglang,kt_kernel,kt_kernel_ext
    effective_sglang=Path(sglang.__file__).resolve().parent.parent
    effective_native=Path(kt_kernel.__file__).resolve().parent.parent
    if tree(effective_sglang)!=payload['sglang'] or tree(effective_native)!=payload['runtime']:
        raise RuntimeError('effective import payload differs from admitted code')
    if hashlib.sha256((effective_sglang/'round9-candidate-overlay.json').read_bytes()).hexdigest()!=admission.get('overlay_sha256'):
        raise RuntimeError('effective overlay receipt differs from admission')
    native_sha256=hashlib.sha256(Path(kt_kernel_ext.__file__).resolve().read_bytes()).hexdigest()
    require_capacity_device_gate(capacity,candidate,identity,payload,native_sha256)
    import ctypes
    import torch
    import torch_npu
    torch.npu.set_device(0)
    acl = ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
    physical = ctypes.c_int32(-1)
    if (acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
            or physical.value != 15 or torch.npu.device_count() != 1
            or list(Path('/dev').glob('davinci[0-9]*')) != [Path('/dev/davinci15')]):
        raise RuntimeError('formal server process does not map only to physical15')
    print('VERIFIED_SERVER_LOGICAL0_PHYSICAL15', torch.npu.mem_get_info(), flush=True)
    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree
    import os
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == '__main__':
    main()
