"""Copy an existing qualified SGLang python tree, add opt-in candidate seam.

Never mutates the supplied baseline; target must not exist. Does not launch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def candidate_model_source(source):
    marker='    cfg = kt_ep_wrapper.create_kt_config_from_server_args(args, layer_id)\n'
    if source.count(marker)!=1 or 'ROUND9_CANDIDATE_PROFILE' in source:
        raise ValueError('unexpected or already patched model source')
    addition='''    candidate_path = os.environ.get("ROUND9_CANDIDATE_PROFILE")
    if not candidate_path:
        raise RuntimeError("isolated candidate runtime requires explicit candidate profile")
    if candidate_path:
        import json
        import kt_kernel_ext
        from pathlib import Path
        from sglang.srt.layers.round9.candidate_config import validate_kt_config, bind_model_profile
        from sglang.srt.layers.round9.candidate_bank import bind_verified_banks
        from sglang.srt.layers.round9.dynamic_boundary import installed_scheduler_boundary
        validate_kt_config(cfg, json.loads(Path(candidate_path).read_text()),
            native_bf16_tail=bool(getattr(kt_kernel_ext,
                "llamafile_bf16_f32_tail_supported", lambda: False)()),
            native_block32_tail=bool(getattr(kt_kernel_ext,
                "llamafile_block32_tail_supported", lambda: False)()),
            dynamic_boundary_installed=installed_scheduler_boundary())
        bank_identity = bind_verified_banks(candidate_path, args.kt_weight_path, args.model_path)
        bind_model_profile(candidate_path, layer_id, bank_identity)
        return
'''
    cap='            cap_bytes = 40448 * 1024**2  #39.5GiB; leaves1.5GiB for unmanaged use'
    if source.count(cap)!=1:raise ValueError('unexpected baseline allocator contract')
    source=source.replace(cap,'''            resource_mode = os.environ.get("ROUND9_RESOURCE_MODE")
            allocator_limits_mib = {"shared": 37376, "exclusive": 43008}
            if resource_mode not in allocator_limits_mib:
                raise RuntimeError("candidate allocator requires admitted resource mode")
            cap_bytes = allocator_limits_mib[resource_mode] * 1024**2''')
    source=source.replace('# Physical64GiB minus user-provided training15GiB and reserve8GiB.',
                          '# Candidate allocator follows the separately admitted resource mode.')
    return source.replace(marker,marker+addition)


def candidate_scheduler_source(source):
    if 'scheduler_before(self, batch)' in source:raise ValueError('scheduler already patched')
    start=source.index('    def event_loop_normal(self):')
    end=source.index('    @DynamicGradMode()',start)
    normal=source[start:end]
    before='            if batch:\n                result = self.run_batch(batch)'
    after='            self.last_batch = batch'
    if normal.count(before)!=1 or normal.count(after)!=1:
        raise ValueError('unexpected normal scheduler event loop')
    normal=normal.replace(before,'''            if batch:
                from sglang.srt.layers.round9.dynamic_boundary import scheduler_before
                scheduler_before(self, batch)
                result = self.run_batch(batch)''')
    normal=normal.replace(after,'''            from sglang.srt.layers.round9.dynamic_boundary import scheduler_after
            scheduler_after(self)
            self.last_batch = batch''')
    return source[:start]+normal+source[end:]


def tree_files(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.relative_to(root)!=Path('round9-candidate-overlay.json')}


def tree_digest(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def install(baseline, target):
    baseline=Path(baseline).resolve();target=Path(target).resolve(strict=False)
    if target.exists() or target.is_symlink():raise FileExistsError(target)
    if baseline==target or baseline in target.parents:raise ValueError('target must be outside baseline')
    model=Path('sglang/srt/models/qwen4_exp.py')
    original=(baseline/model).read_bytes()
    scheduler=Path('sglang/srt/managers/scheduler.py')
    scheduler_patched=candidate_scheduler_source((baseline/scheduler).read_text())
    compile(scheduler_patched,str(scheduler),'exec')
    baseline_files=tree_files(baseline)
    patched=candidate_model_source(original.decode())
    compile(patched,str(model),'exec')
    shutil.copytree(baseline,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (target/model).write_text(patched)
    (target/scheduler).write_text(scheduler_patched)
    runtime=Path(__file__).resolve().parents[1]/'runtime'
    dest=target/'sglang/srt/layers/round9'
    copied={}
    for name in ('candidate_config.py','candidate_bank.py','probe_identity.py','dynamic_boundary.py','dynamic.py','dynamic_adapter.py','expert_source.py','lowbit.py'):
        source=runtime/name
        if not source.is_file():raise FileNotFoundError(source)
        shutil.copy2(source,dest/name)
        copied[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    report=dict(hash_scope='PAYLOAD_EXCLUDING_OVERLAY_RECEIPT_AND_PYCACHE',status='INSTALLED_NOT_DEVICE_QUALIFIED',baseline=str(baseline),
                baseline_model_sha256=hashlib.sha256(original).hexdigest(),
                candidate_model_sha256=hashlib.sha256(patched.encode()).hexdigest(),modules=copied)
    report.update(baseline_files=baseline_files,baseline_tree_sha256=tree_digest(baseline_files))
    candidate_files=tree_files(target)
    report.update(candidate_files=candidate_files,candidate_tree_sha256=tree_digest(candidate_files))
    if tree_files(baseline)!=baseline_files:raise RuntimeError('baseline changed during isolated installation')
    (target/'round9-candidate-overlay.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-python',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(install(a.baseline_python,a.output),indent=2))
