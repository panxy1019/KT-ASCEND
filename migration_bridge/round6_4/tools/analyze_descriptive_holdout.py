"""Descriptive matched-history distributions, never a numerical acceptance gate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    root = args.root
    progress = json.loads((root/"progress.json").read_text())
    assert progress["status"] == "DESCRIPTIVE_CAPTURE_COMPLETE_PENDING_ANALYSIS"
    captures = {}
    for mode in ("off_native", "on_forced", "on_native"):
        integrity = json.loads((root/mode/"final-integrity.json").read_text())
        assert integrity["status"] == "PASS" and integrity["physical_captures"] == 136*65
        result = json.loads((root/mode/"capture/result.json").read_text())
        captures[mode] = {r["rid"]:r for r in result["rows"]}
    assert set(captures["off_native"]) == set(captures["on_forced"]) == set(captures["on_native"])
    rows = []
    for rid, off_row in captures["off_native"].items():
        on_row = captures["on_forced"][rid]
        native_row = captures["on_native"][rid]
        assert off_row["input_ids"] == on_row["input_ids"] == native_row["input_ids"]
        assert off_row["baseline_output_ids"] == on_row["baseline_output_ids"]
        a = torch.load(root/"off_native/capture"/f"{rid}.logits.pt", weights_only=True).double()
        b = torch.load(root/"on_forced/capture"/f"{rid}.logits.pt", weights_only=True).double()
        assert a.shape == b.shape == (64,102400) and torch.isfinite(a).all() and torch.isfinite(b).all()
        diff = b-a
        loga, logb = a.log_softmax(-1), b.log_softmax(-1)
        logm = torch.logaddexp(loga, logb)-torch.log(torch.tensor(2., dtype=torch.float64))
        jsd = .5*((loga.exp()*(loga-logm)).sum(-1)+(logb.exp()*(logb-logm)).sum(-1))
        kl = (loga.exp()*(loga-logb)).sum(-1)
        topa, topb = a.topk(32,dim=-1).indices, b.topk(32,dim=-1).indices
        baseline_margin = a.topk(2,dim=-1).values.diff(dim=-1).neg().flatten()
        values = dict(max_abs=diff.abs().amax(-1), mean_abs=diff.abs().mean(-1),
            relative_l2=diff.norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-12),
            kl_off_to_on=kl, jsd=jsd, baseline_margin=baseline_margin)
        assert all(torch.isfinite(v).all() for v in values.values())
        first = next((i for i,(x,y) in enumerate(zip(off_row["baseline_output_ids"], native_row["baseline_output_ids"])) if x != y), None)
        item = dict(rid=rid, corpus="HD3" if rid.startswith("hd3-") else "FD3", positions=64,
            first_free_divergence=first, cross_mode_token_exact=first is None,
            top1_agreement=int((a.argmax(-1)==b.argmax(-1)).sum()),
            top32_overlap_mean=sum(len(set(x.tolist())&set(y.tolist())) for x,y in zip(topa,topb))/(64*32),
            metrics={k:dict(mean=float(v.mean()), max=float(v.max())) for k,v in values.items()},
            steps=[dict(step=i, current_exposure=on_row["steps"][i]["current_deferred"] > 0,
                        current_deferred_routes=on_row["steps"][i]["current_deferred"],
                        cumulative_exposure=on_row["steps"][i]["cumulative_exposure"],
                        **{k:float(v[i]) for k,v in values.items()}) for i in range(64)])
        if first is not None:
            item["first_free_divergence_pair"] = dict(off=off_row["baseline_output_ids"][first],
                on=native_row["baseline_output_ids"][first], baseline_margin=float(baseline_margin[first]),
                classification="APPROXIMATE_GRAPH_DIVERGENCE_DIAGNOSTIC")
        rows.append(item)
        print(json.dumps(dict(rid=rid, first_free_divergence=first)), flush=True)
    # Independent reporting unit is prompt. Do not treat 64 correlated positions
    # as 64 IID statistical observations or derive an acceptance threshold.
    summaries = {}
    for corpus in ("HD3", "FD3"):
        selected = [r for r in rows if r["corpus"] == corpus]
        summaries[corpus] = dict(prompts=len(selected), positions=64*len(selected),
            free_divergences=sum(r["first_free_divergence"] is not None for r in selected),
            prompt_max_quantiles={k:torch.tensor([r["metrics"][k]["max"] for r in selected],dtype=torch.float64).quantile(
                torch.tensor([.5,.95,.99,1.],dtype=torch.float64)).tolist() for k in values})
    report = dict(status="DESCRIPTIVE_ANALYSIS_COMPLETE", numerical_acceptance_gate=False,
        quality_qualified=False, quantiles=[.5,.95,.99,1.], summaries=summaries, rows=rows,
        capture_progress_sha256=hashlib.sha256((root/"progress.json").read_bytes()).hexdigest())
    with (root/"descriptive-report.json").open("x") as out:
        json.dump(report,out,indent=2)


if __name__ == "__main__":
    main()
