"""Preregistered QD2 derivation or one-shot HD2 validation; no refitting in HD2."""
import argparse
import json
from pathlib import Path
import torch
from assemble_capture import canonical_sha, byte_sha
from tail_statistics import derive_threshold, summarize_tail, prompt_block_maxima


METRICS = ("contraction", "full_logit_max_abs", "full_logit_rel_l2")


def metrics(base, deferred, baseline_token, deferred_token):
    if base.shape != (102400,) or deferred.shape != base.shape:
        raise ValueError("LOGIT_SHAPE_MISMATCH")
    if not torch.isfinite(base).all() or not torch.isfinite(deferred).all():
        raise ValueError("NONFINITE_LOGITS")
    b, d = base.double(), deferred.double()
    a = baseline_token
    if b[a] != b.max() or d[deferred_token] != d.max():
        raise ValueError("SERVING_ANCHOR_NOT_MAXIMUM")
    # Stable sorting on the original token-ID order makes ties deterministic.
    universe = torch.argsort(b, descending=True, stable=True)[:32].tolist()
    universe = sorted(set(universe + [a]))
    m0 = b[a] - b[universe]
    md = d[a] - d[universe]
    contraction = (m0 - md).clamp_min(0)
    difference = d - b
    return dict(contraction=float(contraction.max()),
                beneficial_expansion=float((md - m0).clamp_min(0).max()),
                signed_distortions=(md - m0).tolist(),
                full_logit_max_abs=float(difference.abs().max()),
                full_logit_rel_l2=float(difference.norm() / b.norm().clamp_min(1e-12)),
                full_logit_mean_abs=float(difference.abs().mean()),
                full_logit_cosine=float(torch.dot(b, d) / (b.norm() * d.norm())) if b.norm() > 0 and d.norm() > 0 else None,
                cosine_status="FINITE" if b.norm() > 0 and d.norm() > 0 else "ZERO_NORM_UNDEFINED",
                baseline_token=a, deferred_token=deferred_token,
                candidate_ids=universe, baseline_margins=m0.tolist(), deferred_margins=md.tolist(),
                baseline_candidate_logits=b[universe].tolist(),
                deferred_candidate_logits=d[universe].tolist(),
                outside_universe=deferred_token not in universe,
                logits_byte_exact=byte_sha(base) == byte_sha(deferred))


def classify(row, reserve):
    margins = {i: m for i, m in zip(row["candidate_ids"], row["baseline_margins"])}
    stable = all(m > reserve for i, m in margins.items() if i != row["baseline_token"])
    ambiguity = [i for i, m in margins.items() if m <= reserve]
    base = row["baseline_candidate_logits"]
    deferred = row["deferred_candidate_logits"]
    flips = sum(base[i] - base[j] > reserve and deferred[i] - deferred[j] <= 0
                for i in range(len(base)) for j in range(len(base)) if i != j)
    reasons = []
    if row["outside_universe"]:
        reasons.append("OUTSIDE_CANDIDATE_UNIVERSE")
    if stable and row["baseline_token"] != row["deferred_token"]:
        reasons.append("STABLE_TOP1_MISMATCH")
    if not stable and row["deferred_token"] not in ambiguity:
        reasons.append("AMBIGUITY_ESCAPE")
    if flips:
        reasons.append("STABLE_CANDIDATE_PAIR_FLIP")
    if row["deferred_exposure"] == "NO_EXPOSURE" and not row["logits_byte_exact"]:
        reasons.append("NO_EXPOSURE_LOGIT_MISMATCH")
    return dict(stable=stable, ambiguity_set=ambiguity, stable_pair_flips=flips, failures=reasons)


def read_frozen(path):
    value = json.loads(path.read_text())
    if canonical_sha({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]:
        raise ValueError("MANIFEST_HASH_MISMATCH")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["derive", "heldout"])
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--deferred", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse numerical result overwrite")
    corpus = read_frozen(args.corpus)
    if corpus["corpus"] != ("QD2" if args.action == "derive" else "HD2"):
        raise ValueError("CORPUS_PURPOSE_MISMATCH")
    manifests = [read_frozen(root / "manifest.json") for root in (args.baseline, args.deferred)]
    if [m["mode"] for m in manifests] != ["defer0", "defer1"]:
        raise ValueError("MODE_MISMATCH")
    if any(m["corpus_sha256"] != corpus["sha256"] or m["row_count"] != 8192 for m in manifests):
        raise ValueError("CORPUS_COVERAGE_MISMATCH")
    if manifests[0]["baseline_history_sha256"] != manifests[1]["baseline_history_sha256"]:
        raise ValueError("HISTORY_MISMATCH")
    if any(m.get("complete_forward_count") != 16384 for m in manifests):
        raise ValueError("COMPLETE_FORWARD_COUNT_MISMATCH")
    indexed = [{(r["prompt_id"], r["token_index"]): r for r in m["rows"]} for m in manifests]
    expected = {(p["id"], t) for p in corpus["prompts"] for t in range(64)}
    if any(set(index) != expected or len(index) != 8192 for index in indexed):
        raise ValueError("POSITION_COVERAGE_MISMATCH")
    # Route-only gate MUST precede reading any logit tensor or numerical metric.
    exposure = sum(r["deferred_event_count_current_row"] + r["deferred_event_count_prefix"] > 0
                   for r in manifests[1]["rows"])
    if exposure < 500:
        args.output.mkdir(parents=True)
        report = dict(status="INSUFFICIENT_EXPOSURE", exposure_positions=exposure,
                      numerical_metrics_read=False, required_minimum=500)
        (args.output / "exposure_gate.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        raise SystemExit(1)
    for root in (args.baseline, args.deferred):
        audit = json.loads((root / "transition_audit.json").read_text())
        if audit.get("status") != "PASS" or audit.get("count") != 8192 or audit.get("failures") != []:
            raise ValueError("DEFERRED_TRANSITION_AUDIT_NOT_PASSED")
    rows = []
    for prompt in corpus["prompts"]:
        logits = [torch.load(root / "logits" / f'{prompt["id"]}.pt', map_location="cpu", weights_only=True)
                  for root in (args.baseline, args.deferred)]
        if any(t.shape != (64, 102400) for t in logits):
            raise ValueError("PROMPT_LOGIT_SHAPE_MISMATCH")
        for t in range(64):
            b, d = [index[(prompt["id"], t)] for index in indexed]
            if any(b[k] != d[k] for k in ("request_input_hash", "history_hash", "baseline_history_sha256")):
                raise ValueError("POSITION_HISTORY_MISMATCH")
            if b["logits_sha256"] != byte_sha(logits[0][t]) or d["logits_sha256"] != byte_sha(logits[1][t]):
                raise ValueError("POSITION_LOGIT_HASH_MISMATCH")
            row = metrics(logits[0][t], logits[1][t], b["serving_token"], d["serving_token"])
            row.update(prompt_id=prompt["id"], token_index=t, category=prompt["category"],
                       deferred_exposure=d["deferred_exposure"],
                       deferred_event_count_current_row=d["deferred_event_count_current_row"],
                       deferred_event_count_prefix=d["deferred_event_count_prefix"],
                       deferred_route_records=d["deferred_route_records"],
                       baseline_capture=b["capture_file"], deferred_capture=d["capture_file"],
                       baseline_forward_id=b["forward_id"], deferred_forward_id=d["forward_id"],
                       request_input_hash=d["request_input_hash"])
            rows.append(row)
    expected_counts = {p["id"]: 64 for p in corpus["prompts"]}
    blocks = {m: prompt_block_maxima(rows, m, expected_counts) for m in METRICS}
    if args.action == "derive":
        reserves = {m: derive_threshold(blocks[m], .95, .05 / 3) for m in METRICS}
        candidate = dict(version="round6.2-v2-candidate", qd2_sha256=corpus["sha256"],
                         candidate_topk=32, reserves=reserves, alpha_per_family=.05 / 3,
                         prompt_tail_risk_budget=.05, statistical_unit="prompt_block_maximum",
                         statistical_scope="conditional on independent exchangeable authored prompt blocks",
                         stable_top1_exact_required=True, ambiguity_membership_required=True,
                         stable_candidate_pair_ordering_required=True,
                         no_exposure_logits_byte_exact_required=True,
                         outside_baseline_top32_plus_anchor_allowed=False,
                         runtime_regression_required=True, quality_required=True,
                         free_generation_required=True, final_adr_required=True)
    else:
        candidate = read_frozen(args.candidate)
        reserves = candidate["reserves"]
    for row in rows:
        row.update(classify(row, reserves["contraction"]["threshold"]))
    exposure = sum(row["deferred_exposure"] != "NO_EXPOSURE" for row in rows)
    failures = [dict(prompt_id=r["prompt_id"], token_index=r["token_index"], failures=r["failures"])
                for r in rows if r["failures"]]
    tails = {m: summarize_tail(blocks[m], reserves[m]["threshold"], .05 / 3, .05) for m in METRICS}
    passed = exposure >= 500 and not failures and (args.action == "derive" or all(t["pass"] for t in tails.values()))
    report = dict(action=args.action, status="PASS" if passed else "REJECTED", exposure_positions=exposure,
                  reserves=reserves, tails=tails, first_failure=failures[0] if failures else None,
                  failure_count=len(failures), rows=rows)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Round 6.2 numerical result", "", f"Status: {report['status']}", "",
             f"Action: {args.action}; positions: {len(rows)}; exposure: {exposure}.", "",
             "Confidence is conditional on independent, exchangeable prompt blocks;",
             "this is not an unconditional guarantee for arbitrary production prompts.", "",
             "| Metric | Reserve | Exceeding prompt blocks | CP upper bound |",
             "| --- | ---: | ---: | ---: |"]
    for metric in METRICS:
        lines.append(f"| {metric} | {reserves[metric]['threshold']} | {tails[metric]['exceedances']} | {tails[metric]['upper_confidence_bound']} |")
    lines.extend(["", "QD2 tail counts are calibration diagnostics, not held-out evidence.", "",
                  f"Hard failures: {len(failures)}; first: {report['first_failure']}", "",
                  "No runtime/quality/free-generation promotion is established by this report."])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    if args.action == "derive" and passed:
        candidate["sha256"] = canonical_sha(candidate)
        (args.output / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
