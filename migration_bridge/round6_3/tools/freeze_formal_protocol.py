"""Freeze numerical method and bounded QD3 execution budget before inference."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'round6_2/tools'))
from tail_statistics import tolerance_rank


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from evaluate_stateful_contract import canonical
    plan = json.loads((HERE/'evidence/pilot-retry1/size-plan.json').read_text())
    assert plan['selected_prompts'] == 128
    assert json.loads((HERE/'evidence/formal-adapter-unit/result.json').read_text())['success']
    corpora = {}
    for name in ('qd3', 'hd3', 'fd3'):
        path = HERE/'corpora/frozen'/f'{name}.json'
        value = json.loads(path.read_text())
        assert value['sha256'] == canonical({k:v for k,v in value.items() if k != 'sha256'})
        audit = json.loads((HERE/'corpora'/f'{name}_token_audit.json').read_text())
        assert audit['status'] == 'PASS' and audit['pilot_sha256'] == sha(path)
        corpora[name.upper()] = value['sha256']
    latencies = [r['seconds'] for r in json.loads((HERE/'evidence/pilot-retry1/deferred-capture.json').read_text())['rows']]
    p95 = sorted(latencies)[math.ceil(.95*len(latencies))-1]
    protocol = dict(version='round6.3-v2-preregistration-v1', prompts_per_corpus=128,
        token_budget=64, candidate_topk=32, exposure_minimum=500,
        coverage=.95, alpha_per_family=.05/3, prompt_tail_risk_budget=.05,
        tolerance_rank_1based=tolerance_rank(128, .95, .05/3),
        corpus_sha256=corpora, source_review_sha256=sha(HERE/'corpora/formal_source_review.json'),
        pilot_size_plan_sha256=sha(HERE/'evidence/pilot-retry1/size-plan.json'),
        metrics_module_sha256=sha(HERE.parent/'round6_2/tools/evaluate_contract_v2.py'),
        statistics_module_sha256=sha(HERE.parent/'round6_2/tools/tail_statistics.py'),
        stateful_evaluator_sha256=sha(HERE/'tools/evaluate_stateful_contract.py'),
        model_revision='604d5664dddd88a0433dbae533b7fe9472482de0',
        tokenizer_sha256='41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669',
        placement_sha256='23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c',
        gguf_sha256='4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f',
        quality_manifest_sha256='c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce',
        runtime=dict(tp=1, batch=1, npu=0, cpu_affinity='0-15', graph=False,
                     production_overlap=True, cpu_threads=16, deferred_budget=1),
        formal_history='natural defer0 -> immutable history -> forced defer0 and forced defer1; exactly64 targets plus1 excluded lookahead per request',
        statistical_unit='complete prompt block, conditional on independent exchangeable blocks',
        hard_gates=['stable_top1_exact','ambiguity_membership','outside_U_zero',
                    'stable_candidate_pair_order','no_exposure_byte_exact','all_finite'],
        insufficient_exposure='STOP; no appending/filtering/reweighting',
        hd3_entry='candidate committed and candidate-bound D1 runtime PASS; no refit',
        observer_gate_sha256=sha(HERE/'evidence/observer-correction-result.json'))
    assert protocol['tolerance_rank_1based'] == 127
    protocol['sha256'] = canonical(protocol)
    budget = dict(profile='QD3 batch1', prompts=128, targets_per_mode=8192,
        physical_forwards_per_mode=8320, model_modes=3,
        pilot_request_seconds_p50=statistics.median(latencies), pilot_request_seconds_p95=p95,
        estimated_model_request_seconds=3*128*p95,
        estimated_with_startup_and_audit_seconds=3*128*p95+900,
        client_timeout_per_mode_seconds=3000, server_timeout_per_mode_seconds=3600,
        analytical_step_timeout_seconds=1800, minimum_free_disk_bytes=40*1024**3,
        disk_budget_note='three compact capture sets plus full-logit request copies, exports and reports; reserve40GiB',
        no_numerical_outputs_read=True, preregistration_sha256=protocol['sha256'])
    for name, value in [('PREREGISTRATION.json', protocol), ('QD3_EXECUTION_BUDGET.json', budget)]:
        with (HERE/name).open('x') as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write('\n')
    print(json.dumps(dict(preregistration_sha256=protocol['sha256'], budget=budget)))


if __name__ == '__main__':
    main()
