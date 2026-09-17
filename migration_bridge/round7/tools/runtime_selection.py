"""Bind private CPU import precedence to explicit frozen provenance."""
from pathlib import Path


def select_sglang_runtime(cfg,provenance):
    root=cfg.get('sglang_runtime_root')
    if root is None:
        if provenance.get('sglang_runtime_root'):raise ValueError('PRIVATE_SGLANG_NOT_SELECTED')
        return None
    root=Path(root)
    if not root.is_absolute() or '..' in root.parts or not root.is_relative_to('/workspace/c64-dev/round7'):
        raise ValueError('PRIVATE_SGLANG_OUTSIDE_OWNED_SCOPE')
    if str(root)!=provenance.get('sglang_runtime_root'):
        raise ValueError('PRIVATE_SGLANG_PROVENANCE_MISMATCH')
    wrapper=str(root/'sglang/srt/layers/moe/kt_ep_wrapper.py')
    if wrapper not in provenance['files']:raise ValueError('PRIVATE_SGLANG_WRAPPER_NOT_FROZEN')
    if provenance['files'][wrapper]['sha256']!=provenance['bf16_context_port']['ported_sha256']:
        raise ValueError('PRIVATE_SGLANG_HASH_BINDING')
    helper_hash=provenance['bf16_context_port'].get('staging_helper_sha256')
    if helper_hash:
        helper=str(root/'sglang/srt/layers/moe/llamafile_bf16_staging.py')
        if provenance['files'].get(helper,{}).get('sha256')!=helper_hash:
            raise ValueError('PRIVATE_STAGING_HELPER_HASH_BINDING')
    resident_hash=provenance['bf16_context_port'].get('resident_layout_helper_sha256')
    if resident_hash:
        helper=str(root/'sglang/srt/layers/moe/bf16_resident_layout.py')
        if provenance['files'].get(helper,{}).get('sha256')!=resident_hash:
            raise ValueError('PRIVATE_RESIDENT_LAYOUT_HELPER_HASH_BINDING')
    return str(root)


def select_cpu_runtime(cfg, provenance):
    root = cfg.get('cpu_runtime_root')
    if root is None:
        if provenance.get('cpu_runtime_root'):
            raise ValueError('PRIVATE_RUNTIME_NOT_SELECTED')
        return None
    root = Path(root)
    if not root.is_absolute() or '..' in root.parts or not root.is_relative_to('/workspace/c64-dev/round7'):
        raise ValueError('PRIVATE_RUNTIME_OUTSIDE_OWNED_SCOPE')
    if str(root) != provenance.get('cpu_runtime_root'):
        raise ValueError('PRIVATE_RUNTIME_PROVENANCE_MISMATCH')
    native = str(root/'kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so')
    if native not in provenance['files']:
        raise ValueError('PRIVATE_NATIVE_NOT_FROZEN')
    if provenance['files'][native]['sha256'] != provenance['cpu_native_sha256']:
        raise ValueError('PRIVATE_NATIVE_HASH_BINDING')
    return str(root)
