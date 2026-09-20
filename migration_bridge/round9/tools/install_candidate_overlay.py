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
    if candidate_path:
        import json
        import kt_kernel_ext
        from pathlib import Path
        from sglang.srt.layers.round9.candidate_config import validate_kt_config
        validate_kt_config(cfg, json.loads(Path(candidate_path).read_text()),
            native_bf16_tail=bool(getattr(kt_kernel_ext,
                "llamafile_bf16_f32_tail_supported", lambda: False)()))
        return
'''
    return source.replace(marker,marker+addition)


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
    baseline_files=tree_files(baseline)
    patched=candidate_model_source(original.decode())
    compile(patched,str(model),'exec')
    shutil.copytree(baseline,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (target/model).write_text(patched)
    runtime=Path(__file__).resolve().parents[1]/'runtime'
    dest=target/'sglang/srt/layers/round9'
    copied={}
    for name in ('candidate_config.py','dynamic.py','dynamic_adapter.py','expert_source.py','lowbit.py'):
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
