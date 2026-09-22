"""Old8+8/new32+32 dev intersection exactness; no cross-Deferred equality gate."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root):
    report=json.loads((root/"dev-report.json").read_text())
    assert report["status"] == "DEV_EXECUTION_COMPLETE_PENDING_PROTOCOL_REVIEW"
    assert report["protocol_v2"] and report["profile"] == "D1_NODEFER"
    assert not report["primary_outputs_observed"]
    rows={}
    for line in (root/"requests.jsonl").read_text().splitlines():
        item=json.loads(line)
        rid=item["payload"]["rid"]
        assert rid not in rows
        rows[rid]=item
    return report,rows


def compare(first,second):
    result=[]
    for rid,item in first.items():
        if rid not in second:
            raise ValueError("MISSING_RESTART_ROW:"+rid)
        other=second[rid]
        if item["payload"] != other["payload"]:
            raise ValueError("DEV_RESTART_PROTOCOL_DRIFT:"+rid)
        a,b=item["response"],other["response"]
        for key in ("output_ids","text"):
            if a[key] != b[key]:
                raise ValueError("DEV_RESTART_NONDETERMINISM:"+rid+":"+key)
        if a["meta_info"]["finish_reason"] != b["meta_info"]["finish_reason"]:
            raise ValueError("DEV_RESTART_STOPPING_DRIFT:"+rid)
        result.append(rid)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--dev32",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    a,first=read(args.pilot)
    b,second=read(args.dev32)
    assert len(first)==16 and len(second)==64
    assert a["dev_sha256"]==b["dev_sha256"]
    matches=compare(first,second)
    result=dict(status="DEV32_CLEAN_RESTART_INTERSECTION_EXACT", matched=matches,
                count=len(matches), pilot_raw_sha256=sha(args.pilot/"requests.jsonl"),
                dev32_raw_sha256=sha(args.dev32/"requests.jsonl"),
                scope="Same OFF profile on frozen dev-v2 inputs; not quality acceptance")
    with args.output.open("x") as out:
        json.dump(result,out,indent=2)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
