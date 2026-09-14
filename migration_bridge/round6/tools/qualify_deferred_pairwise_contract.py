#!/usr/bin/env python3
"""Derive on QD, then validate without refitting on HD or larger placements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BOUND_SOURCES = ("p99", "p99_5", "p99_9", "max")
FACTORS = (1.0, 1.25, 1.5)


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verified_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    declared = payload.pop("sha256")
    computed = canonical_sha256(payload)
    if declared != computed:
        raise SystemExit(f"canonical payload SHA mismatch: {path}")
    payload["sha256"] = declared
    return payload


def classify(rows: list[dict], bound: float) -> dict:
    stable = ambiguous = stable_exact = membership = overflow = 0
    failures = []
    ambiguity_sizes = []
    for row in rows:
        anchor = int(row["baseline_top1_id"])
        ambiguity = [anchor] + [
            int(token) for token, margin in zip(row["candidate_ids"], row["baseline_pairwise_margins"])
            if margin <= bound + 1e-12
        ]
        ambiguity = list(dict.fromkeys(ambiguity))
        chosen = int(row["deferred_top1_id"])
        if row["pairwise_abs_max"] > bound + 1e-12:
            overflow += 1
            failures.append({"type": "PAIRWISE_OVERFLOW", "prompt_id": row["prompt_id"], "token_index": row["token_index"], "observed": row["pairwise_abs_max"]})
        if len(ambiguity) == 1:
            stable += 1
            if chosen == anchor:
                stable_exact += 1
            else:
                failures.append({"type": "STABLE_FLIP", "prompt_id": row["prompt_id"], "token_index": row["token_index"], "chosen": chosen, "anchor": anchor})
        else:
            ambiguous += 1
            ambiguity_sizes.append(len(ambiguity))
            if chosen in ambiguity:
                membership += 1
            else:
                failures.append({"type": "AMBIGUITY_ESCAPE", "prompt_id": row["prompt_id"], "token_index": row["token_index"], "chosen": chosen, "ambiguity": ambiguity})
    return {
        "bound": bound, "stable": stable, "stable_exact": stable_exact,
        "ambiguous": ambiguous, "ambiguity_membership": membership,
        "stable_ratio": stable / len(rows) if rows else 0.0,
        "ambiguity_ratio": ambiguous / len(rows) if rows else 0.0,
        "ambiguity_size_max_diagnostic": max(ambiguity_sizes, default=1),
        "pairwise_overflow": overflow, "failures": failures,
        "pass": bool(rows) and not failures,
    }


def derive(args: argparse.Namespace) -> None:
    metrics = verified_json(args.metrics)
    if metrics["corpus_sha256"] != verified_json(args.corpus)["sha256"]:
        raise SystemExit("QD corpus SHA mismatch")
    if not metrics["all_finite"] or not metrics["rows"]:
        raise SystemExit("QD non-finite or empty")
    if metrics["route_telemetry_state"] != "POSITION_ALIGNED":
        raise SystemExit("QD route telemetry is not position-aligned; derive only after fixing instrumentation")
    sensitivity = []
    for source in BOUND_SOURCES:
        for factor in FACTORS:
            row = classify(metrics["rows"], metrics["pairwise_abs_distortion"][source] * factor)
            row.update({"source": source, "factor": factor})
            sensitivity.append(row)
    # Rule declared before HD: QD maximum with 25% reserve, also bounding
    # full-logit max and relative-L2 by their QD maxima with the same reserve.
    selected = next(row for row in sensitivity if row["source"] == "max" and row["factor"] == 1.25)
    if not selected["pass"] or selected["stable"] == 0:
        raise SystemExit("QD candidate does not preserve a non-empty exact stable region")
    contract = {
        "version": "round6-deferred-pairwise-v1-candidate",
        "state": "QD_DERIVED_PENDING_HD",
        "mechanism": "defer0_vs_defer1_matched_history_pairwise_order",
        "pairwise_margin_bound": selected["bound"],
        "pairwise_bound_source": "QD max pairwise absolute distortion * 1.25",
        "full_logit_max_abs_bound": metrics["full_logit_max_abs"]["max"] * 1.25,
        "full_logit_rel_l2_bound": metrics["full_logit_rel_l2"]["max"] * 1.25,
        "candidate_topk": metrics["candidate_topk"],
        "expert_rel_l2_max": 0.01,
        "same_path_determinism_required": True,
        "stable_top1_exact_required": True,
        "ambiguity_membership_required": True,
        "ambiguity_cardinality_gate": False,
        "all_finite_required": True,
        "quality_gate": "no_statistically_meaningful_regression",
        "qd_sha256": metrics["corpus_sha256"],
        "qd_metrics_sha256": metrics["sha256"],
    }
    contract["sha256"] = canonical_sha256(contract)
    report = {
        "schema_version": 1,
        "derivation_rule": "QD max * 1.25, predeclared before HD",
        "sensitivity": sensitivity,
        "selected": selected,
        "contract_sha256": contract["sha256"],
        "route_telemetry_state": metrics["route_telemetry_state"],
    }
    report["sha256"] = canonical_sha256(report)
    args.output_contract.parent.mkdir(parents=True, exist_ok=True)
    args.output_contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "contract_sha256": contract["sha256"]}, indent=2, sort_keys=True))


def validate(args: argparse.Namespace) -> None:
    metrics = verified_json(args.metrics)
    contract = verified_json(args.contract)
    corpus = verified_json(args.corpus)
    if metrics["corpus_sha256"] != corpus["sha256"]:
        raise SystemExit("validation corpus SHA mismatch")
    if args.role == "HD" and corpus["sha256"] == contract["qd_sha256"]:
        raise SystemExit("HD must be independent of QD")
    if metrics["candidate_topk"] != contract["candidate_topk"]:
        raise SystemExit("candidate top-k differs from frozen contract")
    classification = classify(metrics["rows"], float(contract["pairwise_margin_bound"]))
    full_abs_overflow = [
        (row["prompt_id"], row["token_index"]) for row in metrics["rows"]
        if row["full_logit_max_abs"] > contract["full_logit_max_abs_bound"] + 1e-12
    ]
    rel_l2_overflow = [
        (row["prompt_id"], row["token_index"]) for row in metrics["rows"]
        if row["full_logit_rel_l2"] > contract["full_logit_rel_l2_bound"] + 1e-12
    ]
    passed = bool(
        metrics["all_finite"] and classification["pass"]
        and not full_abs_overflow and not rel_l2_overflow
        and metrics["route_telemetry_state"] == "POSITION_ALIGNED"
    )
    report = {
        "schema_version": 1, "role": args.role, "contract_sha256": contract["sha256"],
        "corpus_sha256": corpus["sha256"], "metrics_sha256": metrics["sha256"],
        "classification": classification,
        "full_logit_max_abs_overflow": full_abs_overflow,
        "full_logit_rel_l2_overflow": rel_l2_overflow,
        "all_finite": metrics["all_finite"],
        "route_telemetry_state": metrics["route_telemetry_state"],
        "status": f"{args.role}_VERIFIED" if passed else f"{args.role}_REJECTED",
    }
    report["sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("role", "classification", "full_logit_max_abs_overflow", "full_logit_rel_l2_overflow", "status")}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive_parser = commands.add_parser("derive")
    derive_parser.add_argument("--metrics", type=Path, required=True)
    derive_parser.add_argument("--corpus", type=Path, required=True)
    derive_parser.add_argument("--output-contract", type=Path, required=True)
    derive_parser.add_argument("--output-report", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--metrics", type=Path, required=True)
    validate_parser.add_argument("--contract", type=Path, required=True)
    validate_parser.add_argument("--corpus", type=Path, required=True)
    validate_parser.add_argument("--role", choices=("QD", "HD", "D2", "D3"), required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (derive if args.command == "derive" else validate)(args)


if __name__ == "__main__":
    main()
