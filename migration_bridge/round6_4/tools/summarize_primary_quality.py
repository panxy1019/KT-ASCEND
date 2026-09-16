"""Evaluate complete paired primary endpoints using a previously frozen method."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_quality_interval import evaluate


def stratified_bootstrap(pairs, subjects):
    rng = np.random.default_rng(6404)
    samples = np.zeros(20000)
    for subject in sorted(set(subjects)):
        values = np.array([v for v,s in zip(pairs,subjects) if s == subject])
        counts = np.array([(values == x).sum() for x in (-1,0,1)])
        simulated = rng.multinomial(len(values), counts/len(values), size=20000)
        samples += simulated[:,2]-simulated[:,0]
    return np.quantile(samples/len(pairs), [.025,.975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--off", type=Path, required=True)
    p.add_argument("--on", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--overlap-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b = [json.loads((r/"quality-report.json").read_text()) for r in (args.off,args.on)]
    protocol = json.loads(args.protocol.read_text())
    assert hashlib.sha256(args.overlap_audit.read_bytes()).hexdigest() == protocol["overlap_audit_sha256"]
    overlap = {r["id"]:r for r in json.loads(args.overlap_audit.read_text())["rows"]}
    assert protocol["interval_method"] == "paired_discordance_exact_marginals_bonferroni"
    assert protocol["margin_absolute"] == .02 and protocol["primary_batch"] == 1
    assert a["status"] == b["status"] == "PRIMARY_EXECUTION_COMPLETE"
    assert (a["profile"], b["profile"]) == ("D1_NODEFER", "D1")
    assert a["binding"] == b["binding"]
    assert a["binding"]["protocol_sha256"] == hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    assert [r["id"] for r in a["rows"]] == [r["id"] for r in b["rows"]]
    endpoints = {}
    for benchmark, count in (("gsm8k",1319), ("cmmlu",2048)):
        off = [r for r in a["rows"] if r["benchmark"] == benchmark]
        on = [r for r in b["rows"] if r["benchmark"] == benchmark]
        assert len(off) == len(on) == count
        result = evaluate([r["correct"] for r in off], [r["correct"] for r in on])
        result.update(accuracy_off=result["correct_off"]/count, accuracy_on=result["correct_on"]/count,
                      invalid_off=sum(r["invalid"] for r in off), invalid_on=sum(r["invalid"] for r in on),
                      truncated_off=sum(r["truncated"] for r in off), truncated_on=sum(r["truncated"] for r in on),
                      seconds_off=sum(r["seconds"] for r in off), seconds_on=sum(r["seconds"] for r in on),
                      quality_run_is_formal_performance=False)
        result["relative_accuracy_change"] = result["delta"]/result["accuracy_off"] if result["accuracy_off"] else None
        result["overlap_subsets_descriptive_only"] = {}
        for label in ("historical_literal_overlap", "dev_literal_overlap", "no_known_literal_overlap"):
            def belongs(r):
                x = overlap[r["id"]]
                if label == "historical_literal_overlap":
                    return bool(x["historical_sources"])
                if label == "dev_literal_overlap":
                    return x["dev_text_overlap"]
                return not x["historical_sources"] and not x["dev_text_overlap"]
            chosen = [(x,y) for x,y in zip(off,on) if belongs(x)]
            result["overlap_subsets_descriptive_only"][label] = dict(n=len(chosen),
                correct_off=sum(x["correct"] for x,y in chosen), correct_on=sum(y["correct"] for x,y in chosen))
        subjects = sorted({r.get("subject") or benchmark for r in off})
        result["subjects"] = {s:dict(n=sum((r.get("subject") or benchmark)==s for r in off),
            correct_off=sum(r["correct"] for r in off if (r.get("subject") or benchmark)==s),
            correct_on=sum(r["correct"] for r in on if (r.get("subject") or benchmark)==s)) for s in subjects}
        if benchmark == "cmmlu":
            result["stratified_bootstrap_supplemental_95"] = stratified_bootstrap(
                [int(y["correct"])-int(x["correct"]) for x,y in zip(off,on)], [r["subject"] for r in off])
        endpoints[benchmark] = result
    states = [r["status"] for r in endpoints.values()]
    status = ("QUALITY_ACCEPTED_WITHIN_MARGIN" if all(s=="NONINFERIOR" for s in states) else
              "QUALITY_REGRESSION" if "REGRESSION_BEYOND_MARGIN" in states else "QUALITY_INCONCLUSIVE")
    result = dict(status=status, endpoints=endpoints, primary_profile="batch1_only",
                  protocol_sha256=a["binding"]["protocol_sha256"],
                  generalization_scope=protocol["statistical_target_and_assumptions"],
                  adr_safety_review_required=True, speedup_qualified=False)
    with args.output.open("x") as out:
        json.dump(result,out,indent=2)
    print(json.dumps(dict(status=status, endpoint_states=dict(zip(endpoints,states)))))


if __name__ == "__main__":
    main()
