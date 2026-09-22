"""New runtime provenance/configs only; never copy old qualification results."""
import argparse
import json
from pathlib import Path
import shutil
import time
from static_qualification import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    build=json.loads((a.build/'build.json').read_text())
    assert build['status']=='BUILT_NOT_QUALIFIED','BUILD_PREDECESSOR'
    native=Path(build['native_path']);assert sha(native)==build['native_sha256']
    assert sha(Path('/workspace/c64-dev/runtime/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'))==build['original_native_sha256']
    for layer in (1,13,26):
        r=json.loads((Path('/workspace/c64-dev/round7/cpu-backing-fixed-v1')/f'L{layer}'/'result.json').read_text())
        assert r['status']=='PASS' and len(r['rows'])==6,'EVICTION_REGRESSION_PREDECESSOR'
    (a.output/'profiles').mkdir()
    for name in ('uniform','frequency'):
        for ext in ('json','pt'):
            shutil.copy2(a.base/'profiles'/f'{name}.{ext}',a.output/'profiles'/f'{name}.{ext}')
    original=json.loads((a.base/'provenance.json').read_text())
    provenance={**original,'time_unix':time.time(),'reference_runtime_provenance_sha256':sha(a.base/'provenance.json'),
        'cpu_runtime_root':str(a.build/'runtime'),'cpu_native_sha256':sha(native),
        'native_build':build,'qualification':'POST_FIX_REQUALIFICATION_PENDING'}
    provenance['files']={}
    for path,info in original['files'].items():
        old=Path(path)
        assert sha(old)==info['sha256'],'ORIGINAL_IMPORTED_ARTIFACT_DRIFT'
        if old.is_relative_to('/workspace/c64-dev/runtime'):
            new=a.build/'runtime'/old.relative_to('/workspace/c64-dev/runtime')
        else:new=old
        provenance['files'][str(new)]=dict(sha256=sha(new),size=new.stat().st_size)
    for path in sorted((a.build/'runtime/kt_kernel').rglob('*')):
        if path.is_file() and (path.suffix in ('.py','.so')):
            provenance['files'][str(path)]=dict(sha256=sha(path),size=path.stat().st_size)
    provenance['tools_sha256']={x.name:sha(x) for x in Path(__file__).parent.iterdir() if x.is_file()}
    write(a.output/'provenance.json',provenance)
    cfg=json.loads((a.base/'uniform-dev-config.json').read_text())
    cfg.update(cpu_runtime_root=str(a.build/'runtime'),provenance=str(a.output/'provenance.json'),
        profile=str(a.output/'profiles/uniform.json'),activation_path=str(a.output/'profiles/uniform.pt'))
    write(a.output/'uniform-dev-config.json',cfg)
    cfg.update(recorder=False,observer=False,reference_r64=True)
    write(a.output/'fresh-r64-config.json',cfg)
    print(json.dumps(dict(status='PREPARED_NOT_QUALIFIED',native_sha256=sha(native))))


if __name__=='__main__':main()
