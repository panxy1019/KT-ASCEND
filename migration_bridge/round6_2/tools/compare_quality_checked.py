"""Preregistered paired bootstrap: 20000 replicates, seed6202, percentile CI.

Failure to detect a significant decline is not an equivalence/noninferiority
proof. No arbitrary percentage margin or post-output threshold is introduced.
"""
import argparse
import json
from pathlib import Path
import random
from verify_reserved_inputs import checked


def percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired_summary(baseline, deferred, rng, repeats=20000):
    if len(baseline) != len(deferred) or not baseline:
        raise ValueError("QUALITY_PAIR_COUNT_MISMATCH")
    if any(type(value) is not bool for value in baseline + deferred):
        raise ValueError("QUALITY_CORRECTNESS_NOT_BOOLEAN")
    differences = [int(d) - int(b) for b, d in zip(baseline, deferred)]
    n = len(differences)
    # Identically zero samples have a degenerate bootstrap distribution.
    samples = ([0.] * repeats if not any(differences) else
               [sum(rng.choices(differences, k=n)) / n for _ in range(repeats)])
    ci = [percentile(samples, .025), percentile(samples, .975)]
    return dict(count=n, baseline_correct=sum(baseline), deferred_correct=sum(deferred),
                delta=sum(differences) / n, paired_bootstrap_ci95=ci,
                baseline_only_correct=differences.count(-1), deferred_only_correct=differences.count(1),
                discordant_correctness_pairs=sum(value != 0 for value in differences),
                status="QUALITY_REGRESSION" if ci[1] < 0 else "NO_DETECTED_SIGNIFICANT_REGRESSION")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--deferred", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse quality comparison overwrite")
    baseline, deferred, manifest = [checked(path) for path in (args.baseline, args.deferred, args.manifest)]
    if [baseline["mode"], deferred["mode"]] != ["defer0", "defer1"]:
        raise ValueError("QUALITY_MODE_MISMATCH")
    if any(data["quality_manifest_sha256"] != manifest["sha256"] for data in (baseline, deferred)):
        raise ValueError("QUALITY_MANIFEST_MISMATCH")
    indexed = []
    for data in (baseline, deferred):
        index = {(row["benchmark"], row["sample_id"]): row for row in data["rows"]}
        if len(index) != len(data["rows"]):
            raise ValueError("QUALITY_DUPLICATE_SAMPLE")
        if any(row["invalid"] for row in index.values()) or not data["all_output_tokens_valid"]:
            raise ValueError("QUALITY_INVALID_OUTPUT")
        indexed.append(index)
    expected = {(benchmark, sample["id"]) for benchmark, spec in manifest["benchmarks"].items()
                for sample in spec["samples"]}
    references = {(benchmark, sample["id"]): sample["reference_answer"]
                  for benchmark, spec in manifest["benchmarks"].items() for sample in spec["samples"]}
    if any(set(index) != expected for index in indexed):
        raise ValueError("QUALITY_MISSING_OR_EXTRA_SAMPLE")
    rng = random.Random(6202)
    summaries = {}
    for benchmark, spec in sorted(manifest["benchmarks"].items()):
        keys = [(benchmark, sample["id"]) for sample in spec["samples"]]
        for key in keys:
            for index in indexed:
                row = index[key]
                if row["reference_answer"] != references[key]:
                    raise ValueError("QUALITY_REFERENCE_MISMATCH")
                if row["parsed_answer"] not in "ABCD" or len(row["parsed_answer"]) != 1:
                    raise ValueError("QUALITY_INVALID_PARSED_ANSWER")
                if row["correct"] != (row["parsed_answer"] == references[key]):
                    raise ValueError("QUALITY_CORRECTNESS_LABEL_MISMATCH")
        summaries[benchmark] = paired_summary(*[[index[key]["correct"] for key in keys] for index in indexed], rng)
    report = dict(status="PASS" if all(s["status"] != "QUALITY_REGRESSION" for s in summaries.values()) else "FAIL",
                  quality_manifest_sha256=manifest["sha256"], seed=6202, bootstrap_replicates=20000,
                  invalid_output_count=0, summaries=summaries,
                  interpretation="No detected significant regression is not proof of equivalence or noninferiority.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text("# Round 6.2 quality A/B\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
