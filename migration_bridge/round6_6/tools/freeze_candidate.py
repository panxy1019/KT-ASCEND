"""Freeze reviewed QD interpretation after exact controls, before any HD output."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    base=Path('/workspace/results')
    frozen=base/'r65-r66-study-freeze-v1'
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if (base/'r65-r66-hd6-v1').exists():raise ValueError('HELDOUT_ALREADY_STARTED')
    review=json.loads(a.review.read_text())
    assert review['qd_distribution_review']=='ACCEPT_FOR_INDEPENDENT_HELDOUT'
    assert review['new_threshold_fitting'] is False
    contract=frozen/'CONTRACT_V1.json'
    assert sha(contract)=='4a374b76690ccbafc4fb26006463eb416e80b06a3b834992595e67365393c79b'
    audits=base/'r65-r66-qd6-audits-v1'
    controls=base/'r65-r66-qd6-controls-v1'
    assert json.loads((audits/'progress.json').read_text())['status']=='QD6_AUDITS_COMPLETE_AWAIT_CANDIDATE_REVIEW'
    assert json.loads((controls/'progress.json').read_text())['status']=='SAME_PATH_10_REPEAT_PREFIX_AND_RESTART_PASS'
    bindings={}
    for path,status in [
        (audits/'transfers.json','STUDY_TRANSFER_MERGE_COVERAGE_AUDIT_PASS'),
        (audits/'logits.json','MATCHED_HISTORY_LOGIT_DIAGNOSTICS_COMPLETE'),
        (audits/'routers.json','ROUTER_DIAGNOSTICS_COMPLETE'),
        (controls/'repeat/generation.json','GENERATION_PASS'),
        (controls/'restart/generation.json','GENERATION_PASS'),
    ]:
        obj=json.loads(path.read_text());assert obj['status']==status
        bindings[str(path)]=sha(path)
    assert review['qd_logits_sha256']==sha(audits/'logits.json')
    assert review['qd_routers_sha256']==sha(audits/'routers.json')
    for stage in ('r65-r66-qd6-sequence-guard-v1','r65-r66-qd6-audit-guard-v1','r65-r66-qd6-controls-guard-v1'):
        assert json.loads((base/stage/'result.json').read_text())['reason']=='EXIT_ZERO'
        assert json.loads((base/stage/'reaping.json').read_text())['complete']
        bindings[str(base/stage/'result.json')]=sha(base/stage/'result.json')
    runtime=base/'r65-r66-runtime-source-final-b0.json'
    for item in json.loads(runtime.read_text())['files']:
        assert sha(Path(item['path']))==item['sha256'],'ACTUAL_RUNTIME_DRIFT'
    b0=base/'r65-r66-b0-decision-v1.json'
    assert sha(b0)==json.loads(contract.read_text())['B0_sha256']
    for name,digest in json.loads((frozen/'FREEZE_MANIFEST.json').read_text())['files'].items():
        assert sha(frozen/name)==digest
    candidate=dict(version='r66-heavy-static-candidate-v1',status='FROZEN_BEFORE_HD6',
        contract_sha256=sha(contract),review_sha256=sha(a.review),qd_evidence=bindings,
        qd6_sha256=sha(frozen/'qd6.json'),hd6_sha256=sha(frozen/'hd6.json'),
        runtime_manifest_sha256=sha(runtime),model_distribution='DIAGNOSTIC_ONLY_NO_REFITTED_GLOBAL_MAX_BOUND',
        heldout_execution_allowed=True,quality_execution_allowed=False,final_qualification=False,
        historical_round6_5='BLOCKED_UNCHANGED')
    with a.output.open('x') as f:json.dump(candidate,f,sort_keys=True,indent=2);f.write('\n')
    print(sha(a.output))


if __name__=='__main__':main()
