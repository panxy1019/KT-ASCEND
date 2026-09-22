"""Freeze independent engineering smoke inputs, or a baseline forcing table."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--baseline-result", type=Path)
    p.add_argument("--arm", action="store_true")
    p.add_argument("--synthetic-control", action="store_true")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--token-budget", type=int, default=8)
    args = p.parse_args()
    if args.arm:
        raw = args.baseline_result.read_bytes()
        with args.output.open("x") as out:
            out.write(hashlib.sha256(raw).hexdigest() + "\n")
        return
    if args.baseline_result:
        raw = args.baseline_result.read_bytes()
        baseline = json.loads(raw)
        if baseline["status"] != "PREQUALIFICATION_CAPTURE_PASS" or baseline["forced"]:
            raise ValueError("R63_BASELINE_NOT_VERIFIED")
        result = dict(version="r63-forced-history-v1", session_id=args.output.stem,
            baseline_result_sha256=hashlib.sha256(raw).hexdigest(),
            requests=[{k: row[k] for k in ("rid", "input_ids", "baseline_output_ids")}
                      for row in baseline["rows"]])
        if args.synthetic_control:
            result["requests"] += [dict(rid=row["rid"] + "-synthetic",
                input_ids=row["input_ids"], baseline_output_ids=[101] * len(row["baseline_output_ids"]),
                purpose="uniform synthetic forcing control; not model-generated baseline")
                for row in baseline["rows"]]
    else:
        from tokenizers import Tokenizer
        raw = args.tokenizer.read_bytes()
        if hashlib.sha256(raw).hexdigest() != "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669":
            raise ValueError("TOKENIZER_DRIFT")
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        texts = [
            "Describe why a shaded metal railing can feel colder than nearby wood.",
            "请用两句话解释为什么晴天夜晚草叶上可能出现露水。",
            "A store has 17 blue notebooks and receives 26 more. How many blue notebooks does it have?",
            'Return JSON containing the keys "instrument", "material", and "strings" for a violin.',
        ]
        if not 1 <= args.repeats <= 5 or args.token_budget not in (8, 64):
            raise ValueError("INVALID_SMOKE_SIZE")
        texts = texts * args.repeats
        result = dict(version="r63-engineering-smoke-v1", token_budget=args.token_budget,
            purpose="tool validation only; not pilot/QD3/HD3/FD3",
            tokenizer_sha256=hashlib.sha256(raw).hexdigest(),
            requests=[dict(rid=f"r63-smoke-{i:02d}", text=text,
                input_ids=tokenizer.encode(text, add_special_tokens=False).ids)
                for i, text in enumerate(texts)])
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
