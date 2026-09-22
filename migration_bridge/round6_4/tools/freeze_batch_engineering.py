"""Freeze 12 S6 plus four retained engineering prompts; no model outputs used."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from run_compact_controls import checked, S6_SHA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s6", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = checked(args.s6)
    assert source["sha256"] == S6_SHA
    tok_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    assert tok_sha == "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    requests = [dict(id=r["id"], input_ids=r["input_ids"], source="frozen_S6") for r in source["prompts"]]
    texts = ["Describe why a shaded metal railing can feel colder than nearby wood.",
             "请用两句话解释为什么晴天夜晚草叶上可能出现露水。",
             "A store has 17 blue notebooks and receives 26 more. How many blue notebooks does it have?",
             'Return JSON containing the keys "instrument", "material", and "strings" for a violin.']
    requests.extend(dict(id=f"retained-engineering-{i}", input_ids=tokenizer.encode(text, add_special_tokens=False).ids,
                         text=text, source="Round6.3 prepare_stateful_smoke engineering") for i,text in enumerate(texts))
    assert len(requests) == len({r["id"] for r in requests}) == 16
    cohorts = {str(b): [[r["id"] for r in requests[i:i+b]] for i in range(0, 16, b)] for b in (2, 4)}
    for b in (2, 4):
        assert max(sum(len(r["input_ids"]) for r in requests[i:i+b]) for i in range(0,16,b)) <= 1024
    result = dict(version="r64-batch-engineering-v1", source_s6_sha256=S6_SHA, tokenizer_sha256=tok_sha,
                  token_budget=64, primary_quality=False, cohorts=cohorts, requests=requests,
                  same_process_repeats=10, clean_restart_repeats=1,
                  schedule_rule="One atomic multi-input API call per fixed cohort; await completion before next cohort; audit actual model batches")
    raw = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(result["sha256"])


if __name__ == "__main__":
    main()
