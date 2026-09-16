"""Freeze approved 2pp primary protocol only after dev review, before primary outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

EVALUATORS = ("run_primary_quality.py", "quality_answers_v2.py", "quality_answers.py",
              "paired_quality_interval.py", "summarize_primary_quality.py", "owned_server.py",
              "capacity_python_adapter.py", "run_owned_process.py", "run_compact_controls.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def validate_dev(report, review):
    if report.get("status") != "DEV_EXECUTION_COMPLETE_PENDING_PROTOCOL_REVIEW" or not report.get("protocol_v2"):
        raise ValueError("DEV_NOT_COMPLETE")
    if report.get("primary_outputs_observed") is not False or report.get("profile") != "D1_NODEFER":
        raise ValueError("DEV_SCOPE_DRIFT")
    if len(report["rows"]) != 64 or any(sum(r["benchmark"]==b for r in report["rows"]) != 32 for b in ("gsm8k","cmmlu")):
        raise ValueError("DEV_REVIEW_REQUIRES_32_PLUS_32")
    if review.get("status") != "APPROVED_FOR_PRE_PRIMARY_FREEZE" or review.get("primary_outputs_observed") is not False:
        raise ValueError("DEV_REVIEW_NOT_APPROVED")
    for key in ("stopping_review", "parser_review", "truncation_review", "baseline_validity_review", "statistical_assumptions"):
        if not isinstance(review.get(key),str) or not review[key].strip():
            raise ValueError("MISSING_DEV_REVIEW:"+key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","dev-report","review","overlap","source-provenance","runtime-report","tokenizer-probe","output"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    dev, review=load(args.dev_report), load(args.review)
    validate_dev(dev,review)
    assert review["dev_report_sha256"] == sha(args.dev_report)
    manifest=load(args.inputs/"MANIFEST.json")
    assert manifest["version"] == "round6.4-quality-draft-v2" and manifest["primary_execution_allowed"] is False
    for name,digest in manifest["files"].items():
        assert sha(args.inputs/name) == digest
    assert dev["dev_sha256"] == manifest["files"]["dev.json"]
    assert load(args.overlap)["quality_sha256"] == manifest["files"]["primary.json"]
    assert load(args.runtime_report)["status"] == "RUNTIME_PASS"
    assert load(args.tokenizer_probe)["status"] == "TOKENIZER_COMPAT_SERVING_PASS"
    provenance=load(args.source_provenance)
    protocol=dict(version="round6.4-quality-v2", status="FROZEN_BEFORE_PRIMARY_OUTPUT",
        primary_execution_allowed=True, primary_outputs_observed=False, frozen_unix=time.time(),
        margin_absolute=.02, primary_batch=1, endpoint_counts={"gsm8k":1319,"cmmlu":2048},
        primary_input_sha256=manifest["files"]["primary.json"], dev_input_sha256=manifest["files"]["dev.json"],
        manifest_sha256=sha(args.inputs/"MANIFEST.json"), dev_report_sha256=sha(args.dev_report),
        review_sha256=sha(args.review), overlap_audit_sha256=sha(args.overlap),
        source_provenance_sha256=sha(args.source_provenance),
        # Versioned test-tool paths differ between the dev and primary bundle;
        # their bytes are bound by evaluator_sha256 below, not by old path keys.
        actual_source_sha256={k:v for k,v in provenance["file_sha256"].items()
                              if not k.startswith("/workspace/results/r64-next-tools-")},
        evaluator_sha256={n:sha(Path(__file__).with_name(n)) for n in EVALUATORS},
        runtime_report_sha256=sha(args.runtime_report), tokenizer_serving_probe_sha256=sha(args.tokenizer_probe),
        tokenizer_sha256=manifest["tokenizer_sha256"],
        tokenizer_config_sha256=manifest["tokenizer_config_sha256"],
        tokenizer_overlay_config_sha256="d5f57d20f87e97e94dac06c5f91871bc819fe3fff41b2b0c96c04c2622dc95bb",
        input_bos_policy="explicit_model_config_add_bos_token_true", bos_id=100000, eos_id=100001,
        interval_method="paired_discordance_exact_marginals_bonferroni", endpoint_one_sided_alpha=.025,
        family_alpha=.05, paired_bootstrap_replicates=20000, seed=6404,
        statistical_target_and_assumptions=review["statistical_assumptions"],
        no_midrun_fitting=True, no_accuracy_based_early_stop=True,
        no_samples_removed=True, invalid_and_truncated_are_wrong=True,
        same_path_and_system_gates_remain_exact=True, cross_path_logits_are_not_acceptance_gate=True,
        adr_safety_review_required=True, quality_is_not_speedup_qualification=True)
    with args.output.open("x") as f:
        json.dump(protocol,f,ensure_ascii=False,indent=2)
        f.write("\n")
    print(json.dumps(dict(status=protocol["status"],sha256=sha(args.output))))


if __name__ == "__main__":
    main()
