"""Local stdlib-only evidence integrity check; does not rerun NPU tests."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]/'evidence'
    manifest=json.loads((root/'r65-5a-artifact-manifest-v1.json').read_text())
    verified=0;remote=0
    for row in manifest['files']:
        if not row['in_compact_archive']:
            remote+=1;continue
        path=root/row['path']
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==row['bytes'],path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
        verified+=1
    decision=json.loads((root/'r65-5a-causal-v1/root_cause_decision.json').read_text())
    assert decision['classification']=='INHERENT_CROSS_BACKEND_NUMERICAL_AMPLIFICATION'
    assert all(decision['checks'].values())
    assert decision['ROUND6_5_HEAVY_OFFLOAD_ALIGNMENT']=='BLOCKED_HEAVY_OFFLOAD_RUNTIME'
    assert not decision['promotion_allowed'] and decision['ROUND7']=='NOT_STARTED'
    before=json.loads((root/'r65-5a-runtime-source-manifest-v1.json').read_text())
    after=json.loads((root/'r65-5a-causal-v1/runtime-source-final.json').read_text())
    assert before['files']==after['files'] and before['repositories']==after['repositories']
    assert (root/'r65-5a-causal-v1/gguf-rehash.sha256').read_text().split()[0]=='20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2'
    print(json.dumps(dict(status='EVIDENCE_FREEZE_VALIDATED_ROOT_CAUSE_ONLY',
        local_artifacts_verified=verified,remote_raw_artifacts_hashed_not_copied=remote,
        source_files_unchanged=len(before['files']),promotion_allowed=False)))


if __name__=='__main__':main()
