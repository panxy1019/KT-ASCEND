"""Paired ABBA block inference; resource review is separate and mandatory."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def block_interval(ratios, expected_blocks=5):
    if expected_blocks not in (3,5) or len(ratios) != expected_blocks or any(not np.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("FIVE_POSITIVE_BLOCK_RATIOS_REQUIRED")
    values = np.log(ratios)
    average = float(values.mean())
    sd = float(values.std(ddof=1))
    width = float(t.ppf(.975,expected_blocks-1))*sd/np.sqrt(expected_blocks)
    return dict(block_ratios=list(ratios), geometric_mean=float(np.exp(average)),
        lower=float(np.exp(average-width)), upper=float(np.exp(average+width)),
        variance_resolved=sd > 0, method=f"student_t_log_ratio_df{expected_blocks-1}_95", independent_unit="ABBA_block_not_token")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--resource-review", type=Path, required=True)
    args = p.parse_args()
    campaign = json.loads((args.root/"campaign.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    nblocks=manifest['paired_blocks']
    assert nblocks in (3,5)
    assert campaign["status"] == "PERFORMANCE_CAPTURE_COMPLETE_PENDING_ANALYSIS" and len(campaign["completed"]) == 4*nblocks
    resource = json.loads(args.resource_review.read_text())
    assert resource["status"] in ("ATTRIBUTABLE_WITH_RECORDED_LIMITATIONS", "RESOURCE_CONFOUNDED")
    assert resource["campaign_root"] == str(args.root)
    blocks = []
    for block in range(nblocks):
        phases = []
        for phase,profile in enumerate(manifest["phase_order"]):
            name = f"block{block}-phase{phase}-{profile}"
            assert name in campaign["completed"]
            report = json.loads((args.root/name/"performance-report.json").read_text())
            assert report["status"] == "PERFORMANCE_PHASE_COMPLETE" and report["profile"] == profile
            assert not report["observer"] and report["cache"] == "OFF" and not report["route_recorder"]
            phases.append({r["shape"]:r for r in report["rows"]})
        blocks.append(phases)
    shapes = {}
    for shape in manifest["shapes"]:
        name = shape["id"]
        ratios = []
        for phases in blocks:
            seconds = [p[name]["seconds"] for p in phases]
            ratios.append(float(np.exp((np.log(seconds[0])+np.log(seconds[3])-np.log(seconds[1])-np.log(seconds[2]))/2)))
        result = block_interval(ratios,nblocks)
        for mode,indices in (("off",(0,3)),("on",(1,2))):
            values = [phases[i][name] for phases in blocks for i in indices]
            result[mode] = dict(requests=len(values), latency_p50_p95=np.quantile([r["seconds"] for r in values],[.5,.95]).tolist(),
                ttft_p50_p95=np.quantile([r["ttft_seconds"] for r in values],[.5,.95]).tolist(),
                e2e_tokens_per_second_mean=float(np.mean([r["e2e_tokens_per_second"] for r in values])))
        shapes[name] = result
    primary = shapes[manifest["primary_shape"]]
    status = ("RESOURCE_CONFOUNDED" if resource["status"] == "RESOURCE_CONFOUNDED" else
              "SPEEDUP_VERIFIED" if primary["variance_resolved"] and primary["lower"] > 1 else "NO_MEASURABLE_SPEEDUP")
    result = dict(status=status, primary_shape=manifest["primary_shape"], primary_batch=1,
        shapes=shapes, limitations=manifest["limitations"], resource_review=resource,
        other_shapes_not_used_to_select_primary=True)
    with (args.root/"performance-summary.json").open("x") as out:
        json.dump(result,out,indent=2)
    print(json.dumps(dict(status=status,primary=primary)))


if __name__ == "__main__":
    main()
