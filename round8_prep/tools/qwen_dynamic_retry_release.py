"""Explicit v3 evidence routing; never rewrite failed v2 artifacts."""
import argparse
import json
from pathlib import Path
import sys
from run_qwen_dynamic_heldout_retry import retry_config, sha


def validate_release_identity(release, retry_sha):
    assert release['heldout']=='PASS' and release['cmmlu']=='PASS'
    assert release['heldout_attempt']=='qwen-dynamic-heldout-v3'
    assert release['retry_sha256']==retry_sha


def verify_retry(root):
    frozen=root/'qwen-dynamic-candidate-v2';retry=root/'qwen-dynamic-heldout-retry-v3'
    meta=json.loads((retry/'retry.json').read_text());ack=json.loads((retry/'retry-ack.json').read_text())
    assert len(ack['local_commit'])==40 and ack['retry_sha256']==sha(retry/'retry.json')
    assert sha(frozen/'candidate.json')==meta['candidate_sha256']
    assert sha(frozen/'heldout.json')==meta['original_config_sha256']
    assert sha(retry/'heldout.json')==meta['config_sha256']
    assert sha(root/'tools/run_qwen_dynamic_heldout_retry.py')==meta['runner_sha256']
    assert json.loads((retry/'heldout.json').read_text())==retry_config(json.loads((frozen/'heldout.json').read_text()))
    assert sha(root/'qwen-dynamic-heldout-v3/resolved-command.json')==meta['config_sha256']
    assert json.loads((root/'qwen-dynamic-heldout-v2/state.json').read_text())['status']=='FAILED'
    return sha(retry/'retry.json')


def frozen_source(root,name):
    contract=json.loads((root/'qwen-dynamic-candidate-v2/candidate.json').read_text())
    path=root/'tools'/name
    assert sha(path)==contract['tools_sha256'][name]
    return path.read_text()


def replace_once(source,old,new):
    assert source.count(old)==1,'unexpected frozen source'
    return source.replace(old,new)


def audit(root):
    retry_sha=verify_retry(root)
    source=frozen_source(root,'audit_qwen_dynamic_results.py')
    source=replace_once(source,"out=root/f'qwen-dynamic-{mode}-v2'",
        "out=root/('qwen-dynamic-heldout-v3' if mode=='heldout' else 'qwen-dynamic-cmmlu-v2')")
    namespace={'__name__':'frozen_audit_with_explicit_retry_path'}
    exec(compile(source,'frozen_audit[heldout-v3]','exec'),namespace)
    result={mode:namespace['audit'](root,mode) for mode in ('heldout','cmmlu')}
    release={mode:result[mode]['status'] for mode in result}
    for mode in ('heldout','cmmlu'):
        release[mode+'_summary_sha256']=result[mode]['summary_sha256']
    release.update(heldout_attempt='qwen-dynamic-heldout-v3',retry_sha256=retry_sha)
    result['release_proposal']=release
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/admin/kt-round8-qwen35'))
    p.add_argument('--performance',action='store_true');args=p.parse_args()
    if not args.performance:
        print(json.dumps(audit(args.root),indent=2));return
    assert args.root==Path('/home/admin/kt-round8-qwen35')
    # Fail closed by independently re-auditing before any performance process starts.
    expected=audit(args.root)['release_proposal']
    release=json.loads((args.root/'qwen-dynamic-candidate-v2/qualification-release.json').read_text())
    validate_release_identity(release,expected['retry_sha256'])
    assert all(release[k]==v for k,v in expected.items())
    source=frozen_source(args.root,'run_qwen_dynamic_perf.py')
    source=replace_once(source,"sha(ROOT/f'qwen-dynamic-{mode}-v2/summary.json')",
        "sha(ROOT/('qwen-dynamic-heldout-v3' if mode=='heldout' else 'qwen-dynamic-cmmlu-v2')/'summary.json')")
    # Timed configs, serving entrypoint, isolation and all other guards unchanged.
    sys.argv=[sys.argv[0]]
    exec(compile(source,'frozen_performance[heldout-v3-release]','exec'),{'__name__':'__main__'})


if __name__=='__main__':main()
