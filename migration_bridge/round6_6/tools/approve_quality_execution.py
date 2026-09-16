"""Bind completed heldout/tail review to an unchanged, bounded quality execution.

This is permission to collect evidence, NOT final numerical/quality promotion.
"""
import hashlib
import json
from pathlib import Path


def main():
    base = Path('/workspace/results')
    tools = Path(__file__).parent
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text())
    hd = base / 'r65-r66-heldout-sequence-v1'
    expected = {
        'hd6_logits.json': '36af0eeef0ab9117f17b2d4b8fdaea0648fc253a2b1c40dce49366033738c903',
        'hd6_routers.json': '902ec37ac56287b47aba9dd1602dbe1e5c376d3e93d6b844e9f38e1252567832',
        'hd6_transfers.json': '2dc605e7797e28b8da20f2ba10e1496e2cb5bed480b410653f5023b9e49a6691',
    }
    for name, digest in expected.items():
        assert sha(hd / name) == digest, name
    assert read(hd / 'progress.json')['status'] == 'HD6_ONE_SHOT_AUDITS_COMPLETE_AWAIT_REVIEW'
    assert read(hd / 'hd6_transfers.json')['status'] == 'STUDY_TRANSFER_MERGE_COVERAGE_AUDIT_PASS'
    rows = read(hd / 'hd6_logits.json')['rows']
    assert len(rows) == 768
    assert sum(r['historical_B_pair_overflow_pairs'] > 0 for r in rows) == 3
    assert all(r['historical_stable_exact'] and r['historical_ambiguity_membership'] for r in rows)
    tailpath = base / 'r65-r66-hd6-outlier-operators-v1/three_way.json'
    tail = read(tailpath)
    assert tail['status'] == 'NATURALLY_ROUTED_OUTLIER_OPERATOR_REVIEW_PASS'
    assert len(tail['rows']) == 52
    for r in tail['rows']:
        assert r['deterministic'] and r['local_safety_pass'] and r['finite']
        assert all(r['naturally_routed_rows'])
        for key in ('production_vs_canonical_fp32', 'production_vs_npu'):
            assert r['comparisons'][key]['relative_l2'] <= .01
    for name in ('r65-r66-heldout-sequence-guard-v1', 'r65-r66-hd6-outlier-guard-v1'):
        assert read(base / name / 'result.json')['reason'] == 'EXIT_ZERO'
        assert read(base / name / 'reaping.json')['complete']
    runtime = base / 'r65-r66-runtime-source-final-b0.json'
    assert sha(runtime) == '68d1a5814a70e123eb55c840ec9a995fda1fe4802120374f61f32e2d1775570a'
    for item in read(runtime)['files']:
        assert sha(Path(item['path'])) == item['sha256'], item['path']
    assets = {
        '/workspace/models/DeepSeek-V2-Lite-604d5664/config.json': 'f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d',
        '/workspace/results/r64-tokenizer-compat-v1/config.json': 'f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d',
        '/workspace/results/r64-tokenizer-compat-v1/tokenizer_config.json': 'd5f57d20f87e97e94dac06c5f91871bc819fe3fff41b2b0c96c04c2622dc95bb',
        '/workspace/results/r64-tokenizer-compat-v1/tokenizer.json': '41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669',
    }
    for name, digest in assets.items():
        assert sha(Path(name)) == digest, name
    parsers = {
        'quality_answers.py': '4cae28e44dba632d4a4bf2e4d2c5cd6138abc77e6b5401f5e682bff451d882ba',
        'quality_answers_v2.py': '7d394487111e227f1f5b709d8199564fe5688b91363628c2b4fa4ef4ab743fbc',
    }
    for name, digest in parsers.items():
        assert sha(tools / name) == digest, name
    candidate = base / 'r65-r66-candidate-v1.json'
    assert sha(candidate) == '43a0bd582a4b126feb7ed851b85e1e94c1de98131700547eead4615d3e21967a'
    scripts = ['run_final_screen_sequence.py', 'run_quality_screen.py',
               'compare_quality_screen.py', 'analyze_free_generation.py', 'run_study_generation.py',
               *parsers]
    approval = dict(
        version='r66-quality-execution-review-v1', heldout_system_review='PASS',
        quality_execution_allowed=True, final_qualification=False,
        historical_bound_pass=False, historical_overflow_positions=3,
        new_threshold_fitting=False, heldout_rerun=False,
        model_distribution_review='TAILS_REVIEWED_UNDER_PRE_FROZEN_DIAGNOSTIC_ONLY_CONTRACT',
        interpretation='Local tail safety passed; larger model drift remains and requires independent task quality. No claim of global-logit equivalence.',
        quality_scope='PRE_FROZEN_32_GSM8K_256_CMMLU_ENGINEERING_SCREEN_NOT_POPULATION_NONINFERIORITY',
        candidate_sha256=sha(candidate), heldout_logits_sha256=expected['hd6_logits.json'],
        heldout_routers_sha256=expected['hd6_routers.json'], heldout_transfers_sha256=expected['hd6_transfers.json'],
        tail_operator_sha256=sha(tailpath), tail_operator_count=52,
        runtime_manifest_sha256=sha(runtime), asset_sha256=assets, parser_sha256=parsers,
        execution_tool_sha256={name: sha(tools / name) for name in scripts},
        allowed_profile_sha256=['dc9754048575609eb832049778e6c97e6c9b527a2bbebf8a8e96331549bc8285',
                               '5b5c685956943f575a5cfa8d4c754a2774c0c4e53577fe9340aab35f8b03469b'])
    output = base / 'r65-r66-quality-execution-approval-v1.json'
    with output.open('x') as f:
        json.dump(approval, f, sort_keys=True, indent=2)
        f.write('\n')
    print(sha(output))


if __name__ == '__main__':
    main()
