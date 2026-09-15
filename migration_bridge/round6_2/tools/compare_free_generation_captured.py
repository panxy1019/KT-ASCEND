"""Judge actual first-divergence forward logits under the frozen V2 reserve."""
import argparse
import json
from pathlib import Path
import torch
from assemble_capture import byte_sha
from evaluate_contract_v2 import metrics, classify
from verify_reserved_inputs import checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--deferred", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse FD decision overwrite")
    baseline, deferred = [checked(root / "manifest.json") for root in (args.baseline, args.deferred)]
    candidate = checked(args.candidate)
    if [baseline["mode"], deferred["mode"]] != ["defer0", "defer1"]:
        raise ValueError("FD_MODE_MISMATCH")
    if baseline["corpus_sha256"] != deferred["corpus_sha256"] or any(m["candidate_sha256"] != candidate["sha256"] for m in (baseline, deferred)):
        raise ValueError("FD_FROZEN_INPUT_MISMATCH")
    indexes = [{row["prompt_id"]: row for row in m["rows"]} for m in (baseline, deferred)]
    if set(indexes[0]) != set(indexes[1]) or len(indexes[0]) != 8:
        raise ValueError("FD_PROMPT_COVERAGE_MISMATCH")
    rows = []
    reserve = candidate["reserves"]["contraction"]["threshold"]
    for prompt_id in indexes[0]:
        b, d = [index[prompt_id] for index in indexes]
        first = next((i for i in range(64) if b["output_ids"][i] != d["output_ids"][i]), None)
        if first is None:
            rows.append(dict(prompt_id=prompt_id, first_divergence=None, classification="EXACT", failures=[]))
            continue
        if b["steps"][first]["full_history_sha256"] != d["steps"][first]["full_history_sha256"]:
            raise ValueError("FD_FIRST_DIVERGENCE_HISTORY_MISMATCH")
        vectors = [torch.load(root / f"{prompt_id}.logits.pt", map_location="cpu", weights_only=True)[first]
                   for root in (args.baseline, args.deferred)]
        if any(byte_sha(v) != r["steps"][first]["logits_sha256"] for v, r in zip(vectors, (b, d))):
            raise ValueError("FD_LOGIT_HASH_MISMATCH")
        row = metrics(*vectors, b["output_ids"][first], d["output_ids"][first])
        current = d["steps"][first]
        position = current["actual_positions"][-1]
        current_count = sum(route["position"] == position for route in current["deferred_routes"])
        prefix_count = sum(len(step["deferred_routes"]) for step in d["steps"][:first]) + len(current["deferred_routes"]) - current_count
        exposure = ("CURRENT_AND_PREFIX_EXPOSURE" if current_count and prefix_count else
                    "CURRENT_EXPOSURE" if current_count else "PREFIX_ONLY_EXPOSURE" if prefix_count else "NO_EXPOSURE")
        row.update(prompt_id=prompt_id, first_divergence=first, deferred_exposure=exposure,
                   current_deferred_routes=current_count, prefix_deferred_routes=prefix_count,
                   reserve=reserve, baseline_capture=b["steps"][first]["capture_file"],
                   deferred_capture=d["steps"][first]["capture_file"],
                   interpretation="Actual generating forwards at identical pre-divergence history; no prefill substitution.")
        row.update(classify(row, reserve))
        row["classification"] = "STABLE" if row["stable"] else "AMBIGUOUS"
        rows.append(row)
    report = dict(status="PASS" if all(not r["failures"] for r in rows) else "FAIL",
                  requests=len(rows), divergences=sum(r["first_divergence"] is not None for r in rows),
                  candidate_sha256=candidate["sha256"], rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text("# Round 6.2 FD first-divergence qualification\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
