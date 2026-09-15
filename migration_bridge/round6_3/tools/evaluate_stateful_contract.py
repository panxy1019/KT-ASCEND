"""QD3/HD3 adapter retaining V2 numerical definitions and block-tail rules.

No invocation is authorized by this file's existence: frozen preregistration,
corpus, replay exports, and (for HD3) candidate-bound runtime evidence required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round6_2/tools"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != canonical({k:v for k,v in value.items() if k != "sha256"}):
        raise ValueError(f"FROZEN_HASH_MISMATCH:{path}")
    return value


def integrity_and_exposure(corpus, prereg, baseline, deferred):
    from validate_position import validate
    n = prereg["prompts_per_corpus"]
    if n not in (128, 256, 512) or len(corpus["requests"]) != n or corpus["token_budget"] != 64:
        raise ValueError("UNREGISTERED_SAMPLE_SIZE")
    expected = {(r["rid"], t) for r in corpus["requests"] for t in range(64)}
    if len(expected) != n*64:
        raise ValueError("CORPUS_DUPLICATE_RID")
    indices = []
    for mode, manifest in (("defer0", baseline), ("defer1", deferred)):
        if manifest["mode"] != mode or manifest["corpus_sha256"] != canonical(corpus):
            raise ValueError("EXPORTED_MODE_OR_CORPUS")
        index = {(r["prompt_id"], r["token_index"]): r for r in manifest["rows"]}
        if manifest["row_count"] != n*64 or len(manifest["rows"]) != len(index) or set(index) != expected:
            raise ValueError("POSITION_COVERAGE")
        for row in manifest["rows"]:
            validate(row)
        # Independently reconstruct the causal history; per-row validity alone
        # would not detect accidentally resetting exposure between decode steps.
        for request in corpus["requests"]:
            rid, prompt = request["rid"], request["input_ids"]
            tokens, previous = [], None
            for t in range(64):
                row = index[(rid, t)]
                expected_ids = prompt if t == 0 else [tokens[-1]]
                position = len(prompt) + t - 1
                expected_positions = list(range(len(prompt))) if t == 0 else [position]
                if (row["prompt_hash"] != canonical(prompt) or row["history_hash"] != canonical(tokens) or
                    row["request_input_hash"] != canonical(prompt + tokens) or
                    row["current_input_position"] != position or row["actual_input_ids"] != expected_ids or
                    row["actual_positions"] != expected_positions):
                    raise ValueError("CAUSAL_TOKEN_HISTORY_INVALID")
                if previous is not None:
                    prior = [dict(r, scope="prefix") for r in previous["deferred_route_records"]]
                    prefix = [r for r in row["deferred_route_records"] if r["scope"] == "prefix"]
                    if prefix != prior or row["cpu_route_count_prefix"] != previous["cpu_route_count_prefix"] + previous["cpu_route_count_current_row"]:
                        raise ValueError("CAUSAL_EXPOSURE_RESET_OR_INJECTION")
                tokens.append(row["forced_token"])
                previous = row
        indices.append(index)
    if baseline["baseline_history_sha256"] != deferred["baseline_history_sha256"]:
        raise ValueError("REPLAY_HISTORY_DRIFT")
    for key in expected:
        b, d = (index[key] for index in indices)
        for name in ("request_input_hash", "history_hash", "baseline_history_sha256", "forced_token", "actual_input_ids", "actual_positions"):
            if b[name] != d[name]:
                raise ValueError("POSITION_HISTORY_DRIFT")
    exposure = sum(r["deferred_exposure"] != "NO_EXPOSURE" for r in deferred["rows"])
    return indices, exposure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["derive", "heldout"])
    for name in ("baseline", "deferred", "corpus", "preregistration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--runtime-report", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("REFUSE_RESULT_OVERWRITE")
    corpus, prereg = frozen(args.corpus), frozen(args.preregistration)
    if corpus["corpus"] != ("QD3" if args.action == "derive" else "HD3"):
        raise ValueError("CORPUS_ROLE")
    if (prereg["candidate_topk"] != 32 or prereg["exposure_minimum"] != 500 or
        prereg["coverage"] != .95 or prereg["alpha_per_family"] != .05/3 or
        prereg["prompt_tail_risk_budget"] != .05):
        raise ValueError("FROZEN_METHOD_DRIFT")
    if corpus["sha256"] != prereg["corpus_sha256"][corpus["corpus"]]:
        raise ValueError("CORPUS_NOT_PREREGISTERED")
    manifests = [frozen(path) for path in (args.baseline, args.deferred)]
    indices, exposure = integrity_and_exposure(corpus, prereg, *manifests)
    if exposure < 500:
        args.output.mkdir(parents=True)
        report = dict(status="INSUFFICIENT_EXPOSURE", exposure_positions=exposure,
            required_minimum=500, numerical_metrics_read=False)
        (args.output / "exposure_gate.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        raise SystemExit(1)
    candidate = None
    if args.action == "heldout":
        if args.candidate is None or args.runtime_report is None:
            raise ValueError("HELDOUT_REQUIRES_CANDIDATE_BOUND_RUNTIME")
        candidate, runtime = frozen(args.candidate), frozen(args.runtime_report)
        if (candidate["preregistration_sha256"] != prereg["sha256"] or
            runtime["status"] != "PASS" or runtime["candidate_sha256"] != candidate["sha256"]):
            raise ValueError("RUNTIME_CANDIDATE_MISMATCH")
    # No tensor loads or cross-mode numerical metrics before the gates above.
    import torch
    import evaluate_contract_v2 as v2
    import tail_statistics as stats
    from assemble_capture import byte_sha
    for module, key in ((v2, "metrics_module_sha256"), (stats, "statistics_module_sha256")):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != prereg[key]:
            raise ValueError("NUMERICAL_IMPLEMENTATION_DRIFT")
    torch.set_num_threads(1)
    rows = []
    for request in corpus["requests"]:
        rid = request["rid"]
        for t in range(64):
            b, d = (index[(rid, t)] for index in indices)
            tensors = []
            for row in (b, d):
                data = torch.load(row["capture_file"], map_location="cpu", weights_only=False, mmap=True)
                logits = data["capture.next_token_logits"][0]
                if byte_sha(logits) != row["logits_sha256"]:
                    raise ValueError("POSITION_LOGITS_DRIFT")
                tensors.append(logits)
            metric = v2.metrics(*tensors, b["serving_token"], d["serving_token"])
            metric.update(prompt_id=rid, token_index=t, category=request["category"],
                deferred_exposure=d["deferred_exposure"],
                baseline_capture=b["capture_file"], deferred_capture=d["capture_file"])
            rows.append(metric)
    expected = {r["rid"]:64 for r in corpus["requests"]}
    blocks = {m:stats.prompt_block_maxima(rows, m, expected) for m in v2.METRICS}
    reserves = ({m:stats.derive_threshold(values, .95, .05/3) for m,values in blocks.items()}
                if candidate is None else candidate["reserves"])
    for row in rows:
        row.update(v2.classify(row, reserves["contraction"]["threshold"]))
    failures = [dict(prompt_id=r["prompt_id"], token_index=r["token_index"], failures=r["failures"])
                for r in rows if r["failures"]]
    tails = {m:stats.summarize_tail(values, reserves[m]["threshold"], .05/3, .05) for m,values in blocks.items()}
    passed = not failures and (args.action == "derive" or all(t["pass"] for t in tails.values()))
    report = dict(status="PASS" if passed else "REJECTED", action=args.action,
        exposure_positions=exposure, reserves=reserves, tails=tails, failures=failures, rows=rows,
        preregistration_sha256=prereg["sha256"], corpus_sha256=corpus["sha256"])
    report["sha256"] = canonical(report)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if passed and args.action == "derive":
        candidate = dict(version="round6.3-v2-candidate", reserves=reserves,
            preregistration_sha256=prereg["sha256"], qd3_sha256=corpus["sha256"],
            report_sha256=report["sha256"], candidate_topk=32,
            alpha_per_family=.05/3, prompt_tail_risk_budget=.05,
            statistical_scope="conditional on independent exchangeable prompt blocks",
            heldout_runtime_quality_and_free_generation_required=True)
        candidate["sha256"] = canonical(candidate)
        (args.output / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows", "failures")}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
