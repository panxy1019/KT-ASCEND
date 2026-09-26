"""Validate the three independent Round10 B-route component receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ERROR_KEYS = (
    "relative_l2", "canonical_conv_state_relative_l2",
    "canonical_recurrent_state_relative_l2", "decode_relative_l2",
    "decode_conv_state_relative_l2", "decode_recurrent_state_relative_l2",
    "reset_relative_l2", "reset_conv_state_relative_l2",
    "reset_recurrent_state_relative_l2",
)


def validate(gdn, qsa, ple):
    if gdn.get("status") != "PASS_GDN_COMPONENT_ONLY" or len(gdn.get("rows", [])) != 18:
        raise RuntimeError("GDN current component result incomplete")
    for row in gdn["rows"]:
        gate = gdn["gates"]["bfloat16" if "bfloat16" in row["dtype"] else "float32"]
        values = [row[key] for key in ERROR_KEYS]
        if not all(math.isfinite(value) and value <= gate for value in values) or len(set(row["repeat_hashes"])) != 1:
            raise RuntimeError("GDN independent reference/state gate failed")
    if gdn.get("after_allocated") != 0 or gdn.get("after_reserved") != 0:
        raise RuntimeError("GDN component allocations not retired")

    if (qsa.get("status") != "PASS_QSA_LOCAL_ONLY" or len(qsa.get("rows", [])) != 30
            or qsa.get("topk_stress") is not True or qsa.get("signed_multihead_ranking") is not True):
        raise RuntimeError("QSA current component result incomplete")
    for row in qsa["rows"]:
        gate = qsa["gates"]["bfloat16" if "bfloat16" in row["dtype"] else "float32"]
        values = [row["relative_l2"], row["index_query_relative_l2"], row["compressed_key_relative_l2"]]
        if not all(math.isfinite(value) and value <= gate for value in values) or len(set(row["repeat_hashes"])) != 1:
            raise RuntimeError("QSA independent reference/index gate failed")
    if qsa.get("after_allocated") != 0 or qsa.get("after_reserved") != 0:
        raise RuntimeError("QSA component allocations not retired")

    checks = {
        "original_rows_exact", "duplicates_exact", "cross_shard_exact",
        "consumer_stream_order", "slot_reuse", "request_switch",
        "cancel_late_completion", "zero_credits_after_close",
    }
    if (ple.get("status") != "PASS_PLE_ASYNC_ORIGINAL_ROWS" or ple.get("physical_device") != 12
            or set(ple.get("checks", {})) != checks or not all(ple["checks"].values())
            or len(ple.get("cases", [])) != 3 or not all(case.get("exact") is True for case in ple["cases"])):
        raise RuntimeError("PLE async ownership/original-row gate failed")
    if gdn["source_identity"] != qsa["source_identity"] or gdn["source_identity"] != ple["source_identity"]:
        raise RuntimeError("B-route probes used different model sources")
    return {
        "schema": 1,
        "status": "PASS_ROUND10_T5_COMPONENTS",
        "source_identity": gdn["source_identity"],
        "gdn": {"rows": 18, "peak_allocated": gdn["peak_allocated"], "gates": gdn["gates"]},
        "qsa": {"rows": 30, "peak_allocated": qsa["peak_allocated"], "gates": qsa["gates"],
                "topk_stress": True, "signed_multihead_ranking": True},
        "ple": {"cases": 3, "checks": ple["checks"], "peak_extra_bytes": ple["peak_extra_bytes"],
                "torch_version": ple["torch_version"], "torch_npu_version": ple["torch_npu_version"]},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("gdn", "qsa", "ple", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=validate(*[json.loads(getattr(args,name).read_text()) for name in ("gdn","qsa","ple")])
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result["status"],flush=True)


if __name__=="__main__":main()
