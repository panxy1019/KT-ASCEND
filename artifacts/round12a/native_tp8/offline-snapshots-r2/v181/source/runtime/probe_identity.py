"""Bind bounded probes to an existing trusted full-verification receipt.

Checks all file sizes and hashes the small config/index files. This is not a
fresh full checkpoint rehash; the verified source must remain read-only.
"""
import hashlib
import json
import pathlib


def verify_probe_identity(*, model, receipt, expected_model,
                          expected_receipt_sha256, revision, manifest_sha256):
    root=pathlib.Path(model).resolve(strict=True)
    if root != pathlib.Path(expected_model).resolve(strict=True):
        raise ValueError('unapproved model path')
    raw=pathlib.Path(receipt).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest != expected_receipt_sha256:
        raise ValueError('identity receipt hash mismatch')
    data=json.loads(raw)
    if (data['status']!='PASS' or data['revision']!=revision
            or data['manifest_sha256']!=manifest_sha256):
        raise ValueError('identity receipt does not match frozen source')
    seen=set()
    for record in data['files']:
        name=record['path']; path=(root/name).resolve(strict=True)
        if name in seen or not path.is_relative_to(root) or record['status']!='PASS':
            raise ValueError('invalid or duplicate receipt file')
        seen.add(name)
        if path.stat().st_size!=record['size']:
            raise ValueError('source file size changed: '+name)
        if name in ('config.json','model.safetensors.index.json'):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('source metadata changed: '+name)
    if not {'config.json','model.safetensors.index.json'} <= seen:
        raise ValueError('missing required metadata identity')
    return {'revision':revision,'manifest_sha256':manifest_sha256,
            'receipt_sha256':digest,'model_path':str(root),'file_sizes_checked':len(seen),
            'verification_scope':'TRUSTED_FULL_RECEIPT_PLUS_LIVE_SIZES_AND_METADATA_HASHES'}
