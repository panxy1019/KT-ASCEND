"""Source-bound admission of the exact two-slot original-BF16 PLE transport."""
import hashlib
import json
from pathlib import Path
import re

SOURCE_FILES = frozenset(('ple_async.py','ple_rows.py','host_budget.py','framework_config.py'))
CHECKS = frozenset(('original_rows_exact','duplicates_exact','cross_shard_exact',
    'consumer_stream_order','cancel_late_completion','slot_reuse','request_switch',
    'zero_credits_after_close'))


def require_ple_device_gate(path, expected_sha, *, source_identity, source_sha256,
                            torch_version, torch_npu_version):
    if not path or not isinstance(expected_sha,str) or re.fullmatch('[0-9a-f]{64}',expected_sha) is None:
        raise RuntimeError('PLE requires frozen component receipt')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise RuntimeError('PLE receipt SHA changed')
    gate=json.loads(raw)
    exact=dict(schema=1,status='PASS_PLE_ASYNC_ORIGINAL_ROWS',
        scope='COMPONENT_ONLY_NOT_MODEL_OR_OVERLAP_PERFORMANCE',physical_device=15,
        slots=2,max_rows=512,row_width=160,dtype='torch.bfloat16',
        pinned_bytes=327680,device_buffer_bytes=327680,
        torch_version=torch_version,torch_npu_version=torch_npu_version,
        source_identity=source_identity,source_sha256=source_sha256)
    if (not isinstance(gate,dict) or any(gate.get(key)!=value for key,value in exact.items()) or
            not isinstance(torch_version,str) or not torch_version or
            not isinstance(torch_npu_version,str) or not torch_npu_version or
            not isinstance(source_identity,dict) or not source_identity or
            set(source_sha256)!=SOURCE_FILES or
            any(not isinstance(v,str) or re.fullmatch('[0-9a-f]{64}',v) is None for v in source_sha256.values())):
        raise RuntimeError('PLE receipt/source/geometry/runtime identity mismatch')
    checks=gate.get('checks')
    if (not isinstance(checks,dict) or set(checks)!=CHECKS or
            any(value is not True for value in checks.values()) or
            type(gate.get('peak_extra_bytes')) is not int or
            not 327680 <= gate['peak_extra_bytes'] <= 64*1024**2):
        raise RuntimeError('PLE numerical/lifecycle/resource component gate incomplete')
    return gate
