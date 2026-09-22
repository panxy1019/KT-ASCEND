"""Inventory literal normalized-question overlap, without deleting any sample."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def texts(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("question", "prompt", "text") and isinstance(item, str):
                yield normalized(item)
            elif isinstance(item, (dict, list)):
                yield from texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from texts(item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--quality", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(p for p in args.bridge.glob("*/corpora/**/*.json") if "round6_4" not in p.parts)
    docs = {}
    for path in paths:
        docs[str(path.relative_to(args.bridge))] = set(texts(json.loads(path.read_text())))
    primary = json.loads((args.quality/"primary.json").read_text())["rows"]
    dev = list(texts(json.loads((args.quality/"dev.json").read_text())))
    records = []
    for row in primary:
        question = normalized(row["question"])
        matches = [name for name, strings in docs.items() if any(question == text or question in text for text in strings)]
        dev_matches = sum(question == text or question in text for text in dev)
        records.append(dict(id=row["id"], benchmark=row["benchmark"], historical_sources=matches,
                            dev_text_overlap=dev_matches > 0))
    report = dict(status="AUDIT_COMPLETE_NO_SAMPLES_REMOVED", method="NFKC/casefold/whitespace question equality or substring of historical prompt",
                  limitation="Literal overlap audit, not proof of semantic novelty or absence from model pretraining; historical authoring drafts conservatively included",
                  quality_sha256=hashlib.sha256((args.quality/"primary.json").read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(args.bridge)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  counts={b:dict(total=sum(r["benchmark"]==b for r in records),
                                 historical_overlap=sum(r["benchmark"]==b and bool(r["historical_sources"]) for r in records),
                                 dev_overlap=sum(r["benchmark"]==b and r["dev_text_overlap"] for r in records)) for b in ("gsm8k", "cmmlu")},
                  rows=records)
    with args.output.open("x") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
