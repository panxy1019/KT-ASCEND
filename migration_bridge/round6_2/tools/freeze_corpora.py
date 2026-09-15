"""Freeze both authored corpora together; offline tokenizer, no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata

TOKENIZER_SHA = "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA:
        raise ValueError("TOKENIZER_DRIFT")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    excluded, exclusion_files = set(), []
    for path in sorted(args.exclude_root.glob("**/corpora/*.json")):
        if "round6_2" in path.parts:
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        for row in data.get("prompts", []):
            if isinstance(row, dict) and "text" in row:
                excluded.add(normalize(row["text"]))
        exclusion_files.append({"path":str(path), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    seen = set(excluded)
    manifests = {}
    for name in ("qd2", "hd2"):
        source = args.source_dir / f"{name}_source.json"
        domains = json.loads(source.read_text())["domains"]
        expected = {"english":24,"chinese":24,"code":32,"json":24,"math":24}
        if {k:len(v) for k,v in domains.items()} != expected:
            raise ValueError("CORPUS_SIZE_OR_DOMAIN_DRIFT")
        prompts = []
        for domain, texts in domains.items():
            for index, text in enumerate(texts, 1):
                normalized = normalize(text)
                if normalized in seen:
                    raise ValueError(f"DUPLICATE_OR_HISTORICAL_PROMPT: {text}")
                seen.add(normalized)
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not ids or len(ids) + 64 > 512:
                    raise ValueError("INPUT_LENGTH_INVALID")
                prompts.append(dict(id=f"{name}_{domain}_{index:03d}", category=domain,
                    text=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    input_ids=ids, input_ids_sha256=canonical_sha(ids)))
        random.Random(6202).shuffle(prompts)
        manifest = dict(schema_version=1, corpus=name.upper(), count=128,
            category_counts=expected, prompts=prompts,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            tokenizer_sha256=TOKENIZER_SHA,
            model_revision="604d5664dddd88a0433dbae533b7fe9472482de0",
            protocol=dict(positions=64,temperature=0,ignore_eos=True,
                          add_special_tokens=False,order_seed=6202,
                          baseline_history="freeze_defer0_generation_before_mode_comparison"),
            exclusion_files=exclusion_files,
            independence_scope="Distinct authored tasks; between-prompt exchangeability is a modeling assumption, not established by shuffling.")
        manifest["sha256"] = canonical_sha(manifest)
        manifests[name] = manifest
    # Do not leave just one corpus frozen on validation failure above.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name, manifest in manifests.items():
        (args.output_dir / f"{name}.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    pair = {name:value["sha256"] for name,value in manifests.items()}
    (args.output_dir/"pair_freeze.json").write_text(json.dumps(pair,indent=2)+"\n")
    print(json.dumps(pair))


if __name__ == "__main__":
    main()
