"""Describe two finite stability campaigns; never infer zero leak from one delta."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(report, samples):
    if report.get("status") != "STABILITY_RUNTIME_PASS_PENDING_HBM_REVIEW":
        raise ValueError("STABILITY_RUNTIME_NOT_PASS")
    assert report["requests"] == 18 and report["generated_tokens"] == 1152 and report["pending_drained"]
    rss=report["memory_samples"]
    assert len(rss) == 19
    start,end=rss[0]["time_unix"],rss[-1]["time_unix"]
    during=[s for s in samples if start <= s["time_unix"] <= end]
    if any(s["status"] == "MONITOR_ERROR" for s in during):
        raise ValueError("MEMORY_MONITOR_ERROR_DURING_CAMPAIGN")
    valid=[s for s in during if s["status"] == "SAMPLED" and s.get("owned_hbm_mb_by_pid")]
    if len(valid) < 3:
        raise ValueError("INSUFFICIENT_OWNED_HBM_SAMPLES")
    values=[s["summed_owned_rss_bytes"] for s in rss]
    cycle_rss=[rss[n]["summed_owned_rss_bytes"] for n in (6,12,18)]
    cycle_hbm=[]
    for n in (6,12,18):
        at=rss[n]["time_unix"]
        nearest=min(valid,key=lambda s:abs(s["time_unix"]-at))
        if abs(nearest["time_unix"]-at) > 30:
            raise ValueError("HBM_CYCLE_BOUNDARY_NOT_OBSERVED")
        cycle_hbm.append(sum(nearest["owned_hbm_mb_by_pid"].values()))
    hbm=[sum(s["owned_hbm_mb_by_pid"].values()) for s in valid]
    rss_deltas=[b-a for a,b in zip(cycle_rss,cycle_rss[1:])]
    hbm_deltas=[b-a for a,b in zip(cycle_hbm,cycle_hbm[1:])]
    # This is a review flag, not an invented byte threshold or a leak proof.
    growth=all(d>0 for d in rss_deltas) or all(d>0 for d in hbm_deltas)
    return dict(rss_start=values[0],rss_peak=max(values),rss_end=values[-1],
        rss_cycle_ends=cycle_rss,rss_cycle_end_deltas=rss_deltas,
        hbm_start_mb=hbm[0],hbm_peak_mb=max(hbm),hbm_end_mb=hbm[-1],
        hbm_cycle_ends_mb=cycle_hbm,hbm_cycle_end_deltas_mb=hbm_deltas,
        owned_hbm_samples=len(valid),
        other_npu0_hbm_mb_range=[min(s["other_process_hbm_mb"] for s in valid),max(s["other_process_hbm_mb"] for s in valid)],
        all_cycle_ends_grow_review_required=growth,
        rss_semantics="sum of owned process RSS, includes shared-page double counting",
        observation_limit="Finite campaign only; not a proof of zero leak or asymptotic boundedness")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-a","campaign-b","memory-a","memory-b","output"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    a,b=[json.loads(path.read_text()) for path in (args.campaign_a,args.campaign_b)]
    assert b["reference_sha256"] == sha(args.campaign_a)
    assert a["references"] == b["references"] and a["corpus_sha256"] == b["corpus_sha256"]
    profiles={}
    for name,report,path in (("A",a,args.memory_a),("B",b,args.memory_b)):
        profiles[name]=summarize(report,[json.loads(s) for s in path.read_text().splitlines()])
    result=dict(status="MEMORY_OBSERVATION_COMPLETE_ADR_REVIEW_REQUIRED",generated_tokens=2304,
                same_path_exact=True, campaigns=profiles,
                artifact_sha256={name:sha(path) for name,path in vars(args).items() if name != "output"})
    with args.output.open("x") as out:
        json.dump(result,out,indent=2)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
