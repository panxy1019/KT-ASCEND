"""Exact/normalized historical text dedupe locally; pinned tokenization on A3."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--historical-root", type=Path)
    p.add_argument("--dedupe-report", type=Path)
    p.add_argument("--tokenizer", type=Path)
    args = p.parse_args()
    source = json.loads(args.source.read_text())
    texts = [row["text"] for row in source["prompts"]]
    if len(texts) != 32 or len(set(map(normalized, texts))) != 32:
        raise ValueError("PILOT_SIZE_OR_DUPLICATE")
    if args.historical_root:
        history, files = set(), {}
        for path in sorted(args.historical_root.rglob("*.json")):
            if "round6_3" in path.parts or not ({"corpora", "fixtures"} & set(path.parts)):
                continue
            payload = json.loads(path.read_text())
            history.update(normalized(s) for s in strings(payload))
            files[str(path)] = sha(path)
        if not files:
            raise ValueError("NO_HISTORICAL_CORPUS")
        duplicates = [i for i, text in enumerate(texts) if normalized(text) in history]
        report = dict(status="FAIL" if duplicates else "PASS", source_sha256=sha(args.source),
            scanned_files=files, duplicate_indices=duplicates,
            normalization="NFKC+casefold+whitespace collapse",
            limitation="exact normalized text dedupe; not a proof of semantic independence")
    else:
        from tokenizers import Tokenizer
        checked = json.loads(args.dedupe_report.read_text())
        if checked["status"] != "PASS" or checked["source_sha256"] != sha(args.source):
            raise ValueError("PILOT_DEDUPE_SOURCE_DRIFT")
        token_sha = sha(args.tokenizer)
        if token_sha != "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669":
            raise ValueError("PILOT_TOKENIZER_DRIFT")
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        requests = [dict(rid=f"r63-pilot-{i:02d}", **row,
            input_ids=tokenizer.encode(row["text"], add_special_tokens=False).ids)
            for i, row in enumerate(source["prompts"])]
        if len({tuple(row["input_ids"]) for row in requests}) != 32:
            raise ValueError("PILOT_TOKEN_DUPLICATE")
        if any(not r["input_ids"] or len(r["input_ids"]) + 64 > 512 for r in requests):
            raise ValueError("PILOT_CONTEXT_LIMIT")
        report = dict(version="r63-pilot-input-v1", source_sha256=sha(args.source),
            dedupe_sha256=sha(args.dedupe_report), tokenizer_sha256=token_sha,
            token_budget=64, categories=dict(Counter(r["category"] for r in requests)),
            requests=requests)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(json.dumps(dict(path=str(args.output), sha256=sha(args.output), status=report.get("status", "FROZEN"))))
    if report.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
