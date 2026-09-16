"""Validate the blocked handoff against archived artifacts; never promote PASS."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];evidence=root/'evidence'
    def read(path):return json.loads(path.read_text())
    manifest=read(evidence/'r65-failure-evidence-manifest.json')
    for row in manifest['files']:
        path=evidence/row['path']
        if not path.resolve().is_relative_to(evidence.resolve()):raise ValueError('ARCHIVE_PATH')
        if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('ARTIFACT_HASH:'+row['path'])
    gate=read(evidence/'r65-scale-matrix-v2/four-R32/pairwise.json')
    if gate['status']!='STATIC_NUMERICAL_SCALE_BLOCKER' or gate['B_pair']!=2.1875 or gate['K']!=32:
        raise ValueError('FROZEN_GATE')
    if gate['max_pairwise']!=9.59375 or gate['overflow_count']!=27:raise ValueError('SUMMARY_DRIFT')
    if read(evidence/'r65-four-R32-localization-v1/full-repeat-comparison.json')['status']!='MATCHED_HISTORY_LOGITS_REPEAT_EXACT':
        raise ValueError('REPEAT')
    if read(evidence/'r65-failure-experts-v1/report.json')['status']!='IMPLICATED_EXPERT_SAFETY_PASS':raise ValueError('EXPERT')
    source=read(evidence/'r65-preflight-v1/runtime-source-manifest-v2.json')
    provenance=read(root/'provenance.json')
    ext=next(r for r in source['files'] if r['path'].endswith('kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'))
    if ext['sha256']!=provenance['cpu_extension_sha256']:raise ValueError('EXTENSION_DRIFT')
    if read(root/'round7_baseline.json')['status']!='NOT_FROZEN':raise ValueError('ILLEGAL_PROMOTION')
    if read(root/'deferred_probe.json')['abba_blocks_completed']!=0:raise ValueError('ILLEGAL_DEFERRED')
    required=['00_PROVENANCE.md','01_CAPACITY_PREFLIGHT.md','02_STATIC_MASK_CONTRACT.md',
        '03_SINGLE_LAYER_RATIO_PROBE.md','04_FOUR_LAYER_RATIO_PROBE.md','05_ALL_LAYER_RATIO_SWEEP.md',
        '06_HBM_HOST_PERFORMANCE_PARETO.md','07_STATIC_FREQUENCY_PLACEMENT.md','08_ROUND7_BASELINE_SELECTION.md',
        '09_DEFERRED_HEAVY_PROFILE_PROBE.md','10_FINAL_ADR.md','ROUND6_5_FINAL_SUMMARY.md']
    if any(not (root/name).is_file() for name in required):raise ValueError('MISSING_DOCUMENT')
    print(json.dumps(dict(status='BLOCKED_FREEZE_VALIDATED_NOT_QUALIFICATION_PASS',artifacts=len(manifest['files']))))


if __name__=='__main__':main()
