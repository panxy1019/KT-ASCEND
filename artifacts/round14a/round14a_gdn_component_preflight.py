#!/usr/bin/env python3
"""No-device same-image gate for a frozen Round14a GDN component snapshot."""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROLES = {290: "A", 291: "B"}

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''): digest.update(part)
    return digest.hexdigest()

def tree(root):
    root=Path(root)
    return {str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name not in {'source-sha256.json','host-controller.json'}
            and '__pycache__' not in p.parts and p.suffix != '.pyc'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve(strict=True)
    devices=sorted(str(p) for p in Path('/dev').glob('davinci[0-9]*'))
    if devices: raise RuntimeError(f'component preflight exposed NPU devices: {devices}')
    manifest_path=root/'source-sha256.json'
    if tree(root) != json.loads(manifest_path.read_text()):
        raise RuntimeError('Round14a component manifest mismatch')
    contract=json.loads((root/'ROUND14A_A_GDN_COMPONENT_CONTRACT.json').read_text())
    version,block=contract.get('version'),contract.get('block')
    if (ROLES.get(version) != block
        or contract.get('stage') != 'ROUND14A_A_GDN_PRODUCTION_EQUIVALENT_COMPONENT'
        or contract.get('candidate_default') != 'OFF'
        or contract.get('production_execution_mode') != 'torch.inference_mode'
        or contract.get('empty_cache_allowed') is not False
        or contract.get('same_shape_cycles') != 20
        or contract.get('legal_global_cache_limit_bytes') != 65536
        or contract.get('automatic_retries') != 0):
        raise RuntimeError('Round14a component contract mismatch')
    qwen_path=root/'overlay-python/sglang/srt/models/qwen4_exp.py'; qwen=qwen_path.read_text()
    if (qwen.count('ROUND14_GDN_BACKEND_ENABLE') != 1
        or qwen.count('sigmoid_gated_rms_norm(') != 1
        or 'os.environ.get("ROUND14_GDN_BACKEND_ENABLE", "0") == "1"' not in qwen):
        raise RuntimeError('GDN candidate source gate mismatch')
    probe=(root/'round14a_gdn_component_probe.py').read_text()
    if probe.count('@torch.inference_mode()') < 2 or 'torch.npu.empty_cache()' in probe:
        raise RuntimeError('production-equivalent execution/lifecycle contract missing')
    for name in ('round14a_gdn_component_probe.py','round14a_gdn_component_controller.py',
                 'round14a_gdn_component_preflight.py','monitor_owned_server.py','numa_policy.py'):
        path=root/name; compile(path.read_text(),str(path),'exec')
    tools=root/'source/tools'; sys.path.insert(0,str(tools))
    try:
        from install_candidate_overlay import tree_digest,tree_files
        overlay=root/'overlay-python'; receipt=json.loads((overlay/'round9-candidate-overlay.json').read_text())
        if receipt.get('candidate_tree_sha256') != tree_digest(tree_files(overlay)):
            raise RuntimeError('component overlay receipt mismatch')
    finally: sys.path.remove(str(tools))
    result={'schema':1,'status':'PASS_ROUND14A_A_GDN_COMPONENT_OFFLINE_PREFLIGHT',
            'version':version,'block':block,'physical_devices_exposed':devices,
            'manifest_sha256':sha256(manifest_path),'qwen_sha256':sha256(qwen_path),
            'candidate_default':'OFF','production_execution_mode':'torch.inference_mode',
            'empty_cache_calls':0}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],version,block,flush=True)

if __name__=='__main__': main()
