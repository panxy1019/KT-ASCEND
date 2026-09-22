"""Prepare versioned offline quality inputs before any primary ON observation.

Draft status deliberately remains until dev-only parser/capacity precheck and
historical-overlap audit close. Never sample using model results.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile
import pyarrow.parquet as pq
from tokenizers import Tokenizer

GSM_REV = "740312add88f781978c0658806c59bc2815b9866"
CMMLU_REV = "efcc940752ea4a1ea94d2727f11f83858d64fc8e"
TOKENIZER_SHA = "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    tokenizer_path = args.sources / "tokenizer.json"
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == TOKENIZER_SHA
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}

    def record(path):
        source_hashes[str(path.relative_to(args.sources))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def row(benchmark, split, subject, index, question, prompt, answer, budget):
        return dict(id=f"{benchmark}-{split}-{subject}-{index:05d}", benchmark=benchmark,
                    split=split, subject=subject, source_row_index=index, question=question,
                    prompt=prompt, input_ids=tokenizer.encode(prompt).ids,
                    answer=answer, max_new_tokens=budget)

    primary, dev = [], []
    for split in ("train", "test"):
        table = pq.read_table(record(args.sources / "gsm8k/main" / f"{split}-00000-of-00001.parquet"))
        rows = table.to_pylist()
        assert len(rows) == (7473 if split == "train" else 1319)
        for i, item in enumerate(rows if split == "test" else rows[:32]):
            prompt = ("Solve the following math problem. Explain your reasoning, then end with "
                      "a line in the form #### <final numeric answer>.\n\nQuestion: " +
                      item["question"] + "\n\nAnswer:")
            entry = row("gsm8k", split, "main", i, item["question"], prompt,
                        item["answer"].rsplit("####", 1)[1].strip(), 512)
            (primary if split == "test" else dev).append(entry)
    archive = zipfile.ZipFile(record(args.sources / "cmmlu/cmmlu_v1_0_1.zip"))
    tests = {}
    for name in sorted(archive.namelist()):
        if name.startswith("test/") and name.endswith(".csv"):
            tests[Path(name).stem] = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
    assert len(tests) == 67
    total = sum(map(len, tests.values()))
    allocations = {s: 2048*len(rows)//total for s, rows in tests.items()}
    order = sorted(tests, key=lambda s: (-(2048*len(tests[s]) % total), s))
    for subject in order[:2048-sum(allocations.values())]:
        allocations[subject] += 1
    rng = random.Random(6404)
    for subject, rows in tests.items():
        for split in ("test", "dev"):
            data = rows if split == "test" else list(csv.DictReader(io.StringIO(
                archive.read(f"dev/{subject}.csv").decode("utf-8-sig"))))
            indices = sorted(rng.sample(range(len(data)), allocations[subject])) if split == "test" else range(len(data))
            for i in indices:
                item = data[i]
                assert item["Answer"] in "ABCD" and len(item["Answer"]) == 1
                prompt = ("请回答下面的单项选择题。只输出正确选项的一个大写字母（A、B、C或D），不要解释。\n\n题目：" +
                          item["Question"] + "\n" + "\n".join(f"{k}. {item[k]}" for k in "ABCD") + "\n答案：")
                entry = row("cmmlu", split, subject, i, item["Question"], prompt, item["Answer"], 32)
                (primary if split == "test" else dev).append(entry)
    assert len(primary) == 1319+2048 and len({r["id"] for r in primary}) == len(primary)
    for name, entries in (("primary", primary), ("dev", dev)):
        payload = dict(version="round6.4-quality-draft-v1", rows=entries)
        (args.output / f"{name}.json").write_bytes(canonical(payload) + b"\n")
    for dataset in ("gsm8k", "cmmlu"):
        record(args.sources / dataset / "README.md")
    manifest = dict(version="round6.4-quality-draft-v1", status="PENDING_DEV_CAPACITY_OVERLAP_AUDIT",
                    primary_execution_allowed=False, seed=6404, source_sha256=source_hashes,
                    gsm8k=dict(repo="openai/gsm8k", revision=GSM_REV, test=1319, license="MIT"),
                    cmmlu=dict(repo="lmlmcat/cmmlu", redirected_from="haonan-li/cmmlu", revision=CMMLU_REV,
                               test=2048, available_test=total, subjects=67, allocation=allocations,
                               selection="Hamilton proportional allocation; sorted subjects; Random(6404) sampling without replacement",
                               license="NonCommercial: card metadata BY-NC-4.0; prose BY-NC-SA-4.0; retain both notices"),
                    tokenizer_sha256=TOKENIZER_SHA,
                    files={name: hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ("primary.json", "dev.json")},
                    max_input_tokens={b:max(len(r["input_ids"]) for r in primary if r["benchmark"] == b) for b in ("gsm8k", "cmmlu")},
                    max_required_context=max(len(r["input_ids"])+r["max_new_tokens"] for r in primary),
                    decoding=dict(temperature=0, ignore_eos=False, batch=1, truncation="forbidden"))
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
