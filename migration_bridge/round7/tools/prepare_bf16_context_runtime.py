"""Private SGLang source copy with exact scoped BF16 transport compatibility port."""
import argparse
import json
from pathlib import Path
import shutil
import time
from bf16_context_compat import port_source
from static_qualification import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--llamafile-staging',action='store_true');a=p.parse_args()
    assert a.output.is_relative_to('/workspace/c64-dev/round7')
    a.output.mkdir(parents=True,exist_ok=False)
    source=Path('/workspace/kt-src/third_party/sglang/python')
    original=source/'sglang/srt/layers/moe/kt_ep_wrapper.py'
    assert sha(original)=='8eec41a88d728b4262d7f8407e00fd3b0239e36fc3b3600ea8a85d3db401a1f5'
    runtime=a.output/'sglang-python'
    shutil.copytree(source,runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=runtime/original.relative_to(source)
    target.write_text(port_source(original.read_text(),llamafile_staging=a.llamafile_staging))
    helper=None
    if a.llamafile_staging:
        helper=target.with_name('llamafile_bf16_staging.py')
        shutil.copy2(Path(__file__).with_name(helper.name),helper)
    prior=json.loads((a.base/'provenance.json').read_text())
    provenance={**prior,'time_unix':time.time(),'sglang_runtime_root':str(runtime),
        'reference_runtime_provenance_sha256':sha(a.base/'provenance.json'),
        'qualification':'BF16_CONTEXT_COMPAT_REQUALIFICATION_PENDING',
        'bf16_context_port':dict(original_sha256=sha(original),ported_sha256=sha(target),
            original_path=str(original),ported_path=str(target),
            scope=('BF16_CPUSET_DEVICE_API_AND_TP1_F32_GGUF_STAGING' if a.llamafile_staging
                   else 'BF16_METHOD_CPUSET_PRESERVATION_AND_DEVICE_STREAM_API_ONLY'))}
    provenance['files']={}
    for name,info in prior['files'].items():
        old=Path(name);assert sha(old)==info['sha256'],'REFERENCE_SOURCE_DRIFT'
        prior_root=Path(prior.get('sglang_runtime_root',str(source)))
        if old.is_relative_to(source):new=runtime/old.relative_to(source)
        elif old.is_relative_to(prior_root):new=runtime/old.relative_to(prior_root)
        else:new=old
        provenance['files'][str(new)]=dict(sha256=sha(new),size=new.stat().st_size)
    provenance['files'][str(target)]=dict(sha256=sha(target),size=target.stat().st_size)
    if helper:
        provenance['files'][str(helper)]=dict(sha256=sha(helper),size=helper.stat().st_size)
        provenance['bf16_context_port']['staging_helper_sha256']=sha(helper)
    provenance['tools_sha256']={x.name:sha(x) for x in Path(__file__).parent.iterdir() if x.is_file()}
    write(a.output/'provenance.json',provenance)
    shutil.copytree(a.base/'profiles',a.output/'profiles')
    cfg=json.loads((a.base/'uniform-dev-config.json').read_text())
    cfg.update(sglang_runtime_root=str(runtime),provenance=str(a.output/'provenance.json'),
        profile=str(a.output/'profiles/uniform.json'),activation_path=str(a.output/'profiles/uniform.pt'))
    write(a.output/'uniform-dev-config.json',cfg)
    cfg.update(recorder=False,observer=False,reference_r64=True)
    write(a.output/'fresh-r64-config.json',cfg)
    assert sha(original)==provenance['bf16_context_port']['original_sha256']
    print(json.dumps(provenance['bf16_context_port']))


if __name__=='__main__':main()
