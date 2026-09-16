"""Read-only first-divergence comparison from already captured failed batch."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    audit = json.loads((args.root/"batch-partial-audit.json").read_text())
    assert audit["status"] == "PARTIAL_FAILURE_INTEGRITY_AUDITED_NOT_QUALIFIED"
    failure = audit["first_token_divergences"][0]
    rid = failure["rid"]
    reference_rid = rid.replace("-r1-","-r0-")
    assert reference_rid != rid
    entries = [json.loads(s) for s in (args.root/"index.jsonl").read_text().splitlines()]
    left = [e for e in entries if reference_rid in e["batch"]["request_ids"]]
    right = [e for e in entries if rid in e["batch"]["request_ids"]]
    assert len(left) == len(right) == 65
    report = dict(status="READ_ONLY_FAILURE_COMPARISON",failure=failure,positions=[],first_difference_by_tensor={},
                  audit_sha256=hashlib.sha256((args.root/"batch-partial-audit.json").read_bytes()).hexdigest())
    for step,(a,b) in enumerate(zip(left,right)):
        if step > failure["token_index"]:
            break
        x,y = [torch.load(e["file"],map_location="cpu",weights_only=False,mmap=True) for e in (a,b)]
        row = dict(token_index=step,left_sequence=a["sequence"],right_sequence=b["sequence"],different_tensors=[])
        for key in sorted(set(x)&set(y)):
            if not isinstance(x[key],torch.Tensor) or not isinstance(y[key],torch.Tensor):
                continue
            equal = (x[key].shape == y[key].shape and x[key].dtype == y[key].dtype and
                     torch.equal(x[key].contiguous().view(torch.uint8),y[key].contiguous().view(torch.uint8)))
            if not equal:
                row["different_tensors"].append(key)
                report["first_difference_by_tensor"].setdefault(key,step)
        if step == failure["token_index"]:
            index_a = a["batch"]["request_ids"].index(reference_rid)
            index_b = b["batch"]["request_ids"].index(rid)
            ids = [failure["reference_token"],failure["actual_token"]]
            logits_a,logits_b = x["capture.next_token_logits"][index_a],y["capture.next_token_logits"][index_b]
            row["candidate_ids"] = ids
            row["reference_logits"] = logits_a[ids].float().tolist()
            row["actual_logits"] = logits_b[ids].float().tolist()
        report["positions"].append(row)
    with args.output.open("x") as out:
        json.dump(report,out,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "positions"}))
    print(json.dumps(report["positions"][-1]))


if __name__ == "__main__":
    main()
