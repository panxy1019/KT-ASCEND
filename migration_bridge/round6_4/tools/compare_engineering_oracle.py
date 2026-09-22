"""Fail-closed byte comparison of same-graph production/private oracle captures."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    root = args.root
    report = json.loads((root/"report.json").read_text())
    assert report["status"] == "CAPTURE_COMPLETE_PENDING_EXACT_COMPARISON"
    runs = []
    for mode in ("production", "explicit"):
        assert json.loads((root/mode/"final-integrity.json").read_text())["status"] == "PASS"
        rows = json.loads((root/mode/"capture/result.json").read_text())["rows"]
        assert len(rows) == 4 and len({r["rid"] for r in rows}) == 4
        runs.append({r["rid"]:r for r in rows})
    assert set(runs[0]) == set(runs[1]) == {"r64-oracle-"+pid for pid in report["selected_s6"]}
    oracle = json.loads((root/"explicit/capture/oracle-audit.json").read_text())
    assert oracle["status"] == "DIAGNOSTIC_ORACLE_TRACE_PASS"
    required = {"capture.next_token_logits"}
    for layer in (7, 8):
        required.update(f"r62.ktep.layer{layer}.{key}" for key in
                        ("moe_input", "topk_ids", "routing_weights", "cpu_partial", "npu_owned_partial", "routed_merge"))
    result = dict(status="RUNNING", common_tensor_checks=0, byte_exact=0, full_logit_exact=0,
                  physical_forwards_per_mode=260, private_forwards=oracle["private_forwards"],
                  source_report_sha256=hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),
                  mismatches=[], diagnostic_only_extra_keys=[])
    try:
        extras = set()
        for rid in runs[0]:
            a, b = (r[rid] for r in runs)
            assert a["input_ids"] == b["input_ids"] and a["baseline_output_ids"] == b["baseline_output_ids"]
            assert len(a["steps"]) == len(b["steps"]) == 65
            for left, right in zip(a["steps"], b["steps"]):
                assert left["step"] == right["step"]
                data = [torch.load(s["file"], map_location="cpu", weights_only=False, mmap=True) for s in (left, right)]
                keys = [{k for k,v in d.items() if isinstance(v, torch.Tensor)} for d in data]
                assert required <= keys[0] and required <= keys[1], "MISSING_ORACLE_CORE_TENSOR"
                extras.update(keys[0]^keys[1])
                for key in sorted(keys[0]&keys[1]):
                    x, y = (d[key] for d in data)
                    same = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
                    result["common_tensor_checks"] += 1
                    result["byte_exact"] += int(same)
                    if key == "capture.next_token_logits":
                        result["full_logit_exact"] += int(same)
                    if not same:
                        result["mismatches"].append(dict(rid=rid, step=left["step"], key=key, left=left["file"], right=right["file"]))
        result["diagnostic_only_extra_keys"] = sorted(extras)
        assert not result["mismatches"] and result["full_logit_exact"] == 260, "SAME_GRAPH_ORACLE_MISMATCH"
        assert result["private_forwards"] > 0, "ORACLE_NOT_EXERCISED"
        result["status"] = "ENGINEERING_ORACLE_PASS"
    except BaseException as exc:
        result.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        with (root/"comparison.json").open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
