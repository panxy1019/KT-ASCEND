"""Explicitly reuse unchanged CPU arithmetic evidence, never fabricate a rerun."""
import hashlib
import json
import math
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_operator_evidence(target, source=None):
    target=Path(target);source=Path(source) if source else target
    summary=source/'operator-representatives/summary.json'
    report=json.loads(summary.read_text())
    if report['status']!='PASS' or report['gate']!=.01:
        raise ValueError('OPERATOR_SAFETY_PREDECESSOR')
    if not report['runs'] or any(not math.isfinite(r['max_rel_l2']) or r['max_rel_l2']>.01 for r in report['runs']):
        raise ValueError('OPERATOR_SAFETY_OVERFLOW')
    old=json.loads((source/'provenance.json').read_text())
    new=json.loads((target/'provenance.json').read_text())
    for key in ('cpu_native_sha256','cpu_runtime_root','model_revision','gguf','torch','torch_npu'):
        if old[key]!=new[key]:raise ValueError('OPERATOR_PROVENANCE_DRIFT:'+key)
    root=Path(old['cpu_runtime_root'])
    cpu_files={k:v for k,v in old['files'].items() if Path(k).is_relative_to(root)}
    if not cpu_files:raise ValueError('CPU_FILES_NOT_FROZEN')
    for path,info in cpu_files.items():
        if new['files'].get(path)!=info or sha(Path(path))!=info['sha256']:
            raise ValueError('CPU_ARTIFACT_DRIFT:'+path)
    return dict(status='PASS',scope='REUSED_UNCHANGED_CPU_ARITHMETIC' if source!=target else 'CURRENT_BASE_OPERATOR_RUN',
        source_summary=str(summary),source_summary_sha256=sha(summary),
        source_provenance_sha256=sha(source/'provenance.json'),
        target_provenance_sha256=sha(target/'provenance.json'),
        cpu_native_sha256=new['cpu_native_sha256'],
        limitation='Does not establish new wrapper/cache/migration safety; those require independent runtime system gates.')
