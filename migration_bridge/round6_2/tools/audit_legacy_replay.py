"""Read-only audit of retained R6.1 raw files; no inference and no NPU access."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def audit(root):
    request = json.loads((root / "request.json").read_text())
    expected_ids = [int(x[1]) for x in request["response"]["meta_info"]["input_token_logprobs"]]
    if sha(expected_ids) != request["request_input_ids_sha256"]:
        raise ValueError("request input logprob IDs do not reproduce input hash")
    passes = []
    selected_logits = None
    for path in sorted((root / "tensors").rglob("Pass*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ids = payload["model.forward_batch_info.input_ids"].reshape(-1).tolist()
        positions = payload["model.forward_batch_info.positions"].reshape(-1).tolist()
        logits = payload.get("logits_processor")
        passes.append(dict(path=str(path), input_ids=ids, positions=positions,
                           input_ids_sha256=sha(ids),
                           logits_sha256=tensor_sha(logits) if logits is not None else None))
        if positions[-1] == len(expected_ids) and logits is not None:
            if selected_logits is not None:
                raise ValueError("ambiguous selected forward")
            selected_logits = logits[-1].float().contiguous()
            selected_index = len(passes) - 1
    if selected_logits is None or selected_index == 0:
        raise ValueError("no compensated pass and predecessor")
    previous = passes[selected_index - 1]
    current = passes[selected_index]
    association = (previous["input_ids"] == expected_ids
                   and previous["positions"] == list(range(len(expected_ids)))
                   and current["input_ids"] == request["response_output_ids"]
                   and current["positions"] == [len(expected_ids)])
    # Independent API evidence: relative logprob differences cancel logsumexp.
    top = request["response"]["meta_info"]["output_top_logprobs"][0]
    anchor = request["response_output_ids"][0]
    anchor_lp = next(float(x[0]) for x in top if int(x[1]) == anchor)
    gaps = [abs(float(selected_logits[anchor] - selected_logits[int(x[1])])
                - (anchor_lp - float(x[0]))) for x in top]
    return dict(root=str(root), association=association,
                input_hash_verified=True, passes=passes,
                selected_pass=current["path"], predecessor_pass=previous["path"],
                final_logits_sha256=tensor_sha(selected_logits),
                top1=int(selected_logits.argmax()), api_anchor=anchor,
                api_top_candidate_gap_max_error=max(gaps),
                classification="OFFSET_COMPENSATED" if association and max(gaps) < 1e-5 else "UNSAFE_FOR_NUMERICAL_REUSE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit(root) for root in args.root]
    if len(args.root) == 2:
        left = {p.name:p for p in (args.root[0]/"stages").glob("*.pt")}
        right = {p.name:p for p in (args.root[1]/"stages").glob("*.pt")}
        checks = []
        for name in sorted(left.keys() & right.keys()):
            a = torch.load(left[name], map_location="cpu", weights_only=False)["tensor"]
            b = torch.load(right[name], map_location="cpu", weights_only=False)["tensor"]
            checks.append(dict(name=name, shape=list(a.shape),
                               left_sha=tensor_sha(a), right_sha=tensor_sha(b),
                               exact=torch.equal(a,b)))
    else:
        checks = []
    result = dict(schema="r62-legacy-replay-audit-v1", roots=results,
                  independent_stage_checks=checks,
                  stage_count=len(checks), stage_exact=sum(x["exact"] for x in checks),
                  inference_executed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result,out,indent=2,ensure_ascii=False)
        out.write("\n")
    print(json.dumps({"roots":[{k:v for k,v in x.items() if k not in ("passes",)} for x in results],
                      "stage_count":len(checks), "stage_exact":sum(x["exact"] for x in checks)}))


if __name__ == "__main__":
    main()
