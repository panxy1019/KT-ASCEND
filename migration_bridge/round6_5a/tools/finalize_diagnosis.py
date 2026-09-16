"""Validate causal evidence before recording a bounded root-cause decision."""
import hashlib
import json
from pathlib import Path


def main():
    base=Path('/workspace/results');causal=base/'r65-5a-causal-v1'
    def read(path):return json.loads(path.read_text())
    subset=read(base/'r65-5a-subset-v1/probe/npu_subset_equivalence.json')
    canonical=[read(base/f'{name}/probe/canonical_mixed_oracle.json') for name in
               ('r65-5a-canonical-v1','r65-5a-prefill-v1')]
    analysis=read(causal/'counterfactual_replay_v2.json')['profiles']
    router=read(causal/'router_boundary.json')
    original=analysis['C4'];routes=analysis['baseline_routes'];oracle=analysis['npu_for_cpu']
    guards=[base/'r65-5a-subset-v1/guard',base/'r65-5a-canonical-v1/guard',
        base/'r65-5a-prefill-v1/guard',base/'r65-5a-bisection-v1/C1/guard',
        base/'r65-5a-bisection-v2/C2/guard',base/'r65-5a-bisection-v2/C3/guard',
        causal/'router-guard',causal/'baseline_routes/guard',causal/'npu_for_cpu/guard']
    guard_rows=[]
    for guard in guards:
        result=read(guard/'result.json');reap=read(guard/'reaping.json')
        assert result['reason'] in ('EXIT_ZERO','CANCELLED') and reap['complete']
        guard_rows.append(dict(path=str(guard),reason=result['reason'],reaped=True))
    actual_profile=read(causal/'npu_for_cpu/resolved-command.json')['profile']
    frozen_profile=read(base/'r65-profiles-v1/four-R32.json')
    assert actual_profile==frozen_profile
    structure=[json.loads(s) for s in (causal/'npu_for_cpu/physical-structure.jsonl').read_text().splitlines()]
    assert len(structure)==26
    for row in structure:
        expected=next(r for r in frozen_profile['rows'] if r['layer']==row['layer'])
        assert row['npu_ids']==expected['npu_ids'] and row['cpu_ids']==expected['cpu_ids']
    intervention=[json.loads(s) for s in (causal/'npu_for_cpu-interventions/interventions.jsonl').read_text().splitlines()]
    assert len(intervention)==260 and {(r['layer'],r['call']) for r in intervention}=={
        (layer,call) for layer in (1,9,17,26) for call in range(1,66)}
    localmax=max(e['relative_l2'] for d in canonical for row in d['rows'] for e in row['expert_rows'] if e['cpu_owned'])
    checks=dict(npu_subset_exact=len(subset['rows'])==32 and all(r['exact'] and r['deterministic'] and r['finite'] for r in subset['rows']),
        canonical_final_exact=all(r['canonical_final_vs_production']['exact'] for d in canonical for r in d['rows']),
        cpu_capture_replay_exact=all(r['cpu_replay_vs_capture']['exact'] for d in canonical for r in d['rows']),
        local_expert_safety=localmax<=0.01,
        router_replay_exact=router['status']=='ROUTER_REPLAY_EXACT',
        route_counterfactual_reduces_l17_jump=routes['rows'][16]['mlp_output']['relative_l2']<original['rows'][16]['mlp_output']['relative_l2'],
        route_counterfactual_reduces_full_logit_l2=routes['full_logits']['failure_token54']['relative_l2']<original['full_logits']['failure_token54']['relative_l2'],
        npu_for_cpu_full_logits_exact=oracle['full_logits']['exact'] and oracle['full_logits']['exact_positions']==64,
        original_r32_ownership_retained=True,intervention_coverage_exact=True,lifecycles_reaped=True)
    assert all(checks.values()),checks
    result=dict(classification='INHERENT_CROSS_BACKEND_NUMERICAL_AMPLIFICATION',
        evidence_state='A3_VERIFIED',scope='FROZEN_H2_EN_01_FAILURE_TRAJECTORY_AND_BOUNDED_OPERATOR_PROBES',
        production_fix='NONE_NO_SEMANTIC_INVARIANT_DEFECT_DEMONSTRATED',checks=checks,
        local_expert_rel_l2_max=localmax,
        aggregate_cpu_partial_vs_gmm_max_rel_l2=max(r['relative_l2'] for r in intervention),
        layer_prefix_minimum='C1_LAYER1_ONLY',guards=guard_rows,
        counterfactual_caveat='Fixed baseline routes reduce L17/full-logit L2, but worsen maximum pairwise distortion to 16.65625; not a fix.',
        method_reduction_tree_proven=False,kv_cache_bytes_directly_compared=False,
        exclusions='No claim that every input/kernel shape is defect-free; no universal numerical tolerance derived.',
        ROUND6_5_HEAVY_OFFLOAD_ALIGNMENT='BLOCKED_HEAVY_OFFLOAD_RUNTIME',
        BLOCKER_SUBCLASS='STATIC_NUMERICAL_SCALE_BLOCKER',ROUND7='NOT_STARTED',
        DEFAULT_DEFERRED='OFF',promotion_allowed=False,
        next_study='Independent heavy-static numerical qualification with new QD/HD; no old threshold refit',
        recovered_tooling_failures=['forced baseline rejected by qualification evaluator; diagnostic exact-history join used instead',
            'offline layer summary imported serving dependency; CPU-only metrics separated and v2 summary completed'])
    with (causal/'root_cause_decision.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(classification=result['classification'],checks=checks)))


if __name__=='__main__':main()
