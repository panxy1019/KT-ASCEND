"""Preserve red/green subprocess evidence on an idle owned diagnostic lane."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/admin/kt-round8-qwen35');CONTAINER='kt-round8-qwen35-prep'
out=ROOT/'dynamic-nz-transfer-regression-v1';out.mkdir(exist_ok=False)
state=json.loads(subprocess.check_output(['docker','inspect',CONTAINER]))[0]['State']
assert not state['Running'],'preserve active owned work'
subprocess.run(['docker','start',CONTAINER],check=True)
try:
    results={}
    for mode,folder in [('old','sglang-python'),('fixed','sglang-python-dynamic-v2')]:
        source=f'{folder}/sglang/srt/layers/moe/kt_ep_wrapper.py'
        r=subprocess.run(['docker','exec',CONTAINER,'/workspace/round8/venv/bin/python',
            '/workspace/round8/tools/test_dynamic_nz_transfer.py','--source',f'/workspace/round8/{source}'],capture_output=True,text=True,timeout=180)
        results[mode]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
            source_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest())
        (out/f'{mode}.json').write_text(json.dumps(results[mode],indent=2))
    assert results['old']['returncode']!=0 and 'UNSAFE_NZ_SLICE_COPY_WARNING' in results['old']['stderr']
    assert results['fixed']['returncode']==0 and 'Check input storage_offset' not in results['fixed']['stderr']
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',negative_control='EXPECTED_WARNING_FAILURE',
        fixed='NO_WARNING_EXACT_WEIGHT_ORACLE',updates=3),indent=2))
finally:
    subprocess.run(['docker','stop','-t','20',CONTAINER],capture_output=True,timeout=90)
