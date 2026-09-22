"""Preregistered exposure-only planning rule. Does not read model logits."""
import argparse
import json
import math
from pathlib import Path


def choose(exposures):
    if len(exposures) != 32 or any(type(x) is not int or not 0 <= x <= 64 for x in exposures):
        raise ValueError("PILOT_REQUIRES_32_COMPLETE_64_POSITION_BLOCKS")
    mean = sum(exposures) / 32
    lower = max(0., mean - 64 * math.sqrt(math.log(60) / 64))
    candidates = [dict(prompts=n, conservative_total=n*lower - 64*math.sqrt(n*math.log(60)/2))
                  for n in (128, 256, 512)]
    selected = next((c["prompts"] for c in candidates if c["conservative_total"] >= 500), None)
    return dict(mean_exposed_positions=mean, lower_mean=lower, candidates=candidates,
                selected_prompts=selected, exposed_prompts=sum(x > 0 for x in exposures),
                exposure_minimum=500, numerical_metrics_read=False,
                planning_alpha_per_event=.05/3,
                assumption="independent exchangeable bounded prompt blocks; conditional planning bound")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = json.loads(args.pilot.read_text())
    if result["status"] != "PREQUALIFICATION_CAPTURE_PASS":
        raise ValueError("PILOT_CAPTURE_INCOMPLETE")
    exposures = []
    for row in result["rows"]:
        target = [s for s in row["steps"] if s["target"]]
        if len(target) != 64:
            raise ValueError("PILOT_POSITION_COUNT")
        exposures.append(sum(s["cumulative_exposure"] for s in target))
    report = choose(exposures)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
