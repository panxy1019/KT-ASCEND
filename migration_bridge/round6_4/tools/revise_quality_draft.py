"""Pre-primary v2 protocol: configured BOS, zero-shot CoT, task delimiters.

Selection, labels, sampling, budgets and 2pp margin are unchanged. Only train/dev
v1 was observed. This draft is still forbidden for primary execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()+b"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--sources",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    old=json.loads((args.input/"MANIFEST.json").read_text())
    assert old["version"] == "round6.4-quality-draft-v1"
    assert digest(args.sources/"tokenizer.json") == old["tokenizer_sha256"]
    config=json.loads((args.sources/"tokenizer_config.json").read_text())
    assert digest(args.sources/"tokenizer_config.json") == "31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8"
    assert config["add_bos_token"] is True and config["add_eos_token"] is False
    tokenizer=Tokenizer.from_file(str(args.sources/"tokenizer.json"))
    bos=config["bos_token"]
    assert tokenizer.token_to_id(bos["content"] if isinstance(bos,dict) else bos) == 100000
    args.output.mkdir(exist_ok=False)
    for name in ("primary.json","dev.json"):
        assert digest(args.input/name) == old["files"][name]
        payload=json.loads((args.input/name).read_text())
        for row in payload["rows"]:
            if row["benchmark"] == "gsm8k":
                row["prompt"] = "Question: "+row["question"]+"\nAnswer: Let's think step by step."
                row["stop"] = ["\nQuestion:","\n\nQuestion:","\nQ:"]
            else:
                row["stop"] = ["\n"]
            # JSON vocabulary/pipeline is unchanged. Explicitly honor the model
            # config's BOS policy; generic TokenizersBackend omits this flag.
            row["input_ids"] = [100000]+tokenizer.encode(row["prompt"],add_special_tokens=False).ids
            row["input_bos_policy"] = "explicit_model_config_add_bos_token_true"
            assert len(row["input_ids"])+row["max_new_tokens"] <= 1024
        payload["version"]="round6.4-quality-draft-v2"
        (args.output/name).write_bytes(canonical(payload))
    new=dict(old, version="round6.4-quality-draft-v2",status="PENDING_DEV_V2_REVIEW",
             parent_manifest_sha256=digest(args.input/"MANIFEST.json"),primary_execution_allowed=False,
             tokenizer_config_sha256=digest(args.sources/"tokenizer_config.json"),
             input_bos_policy="explicit_model_config_add_bos_token_true",
             tokenizer_loader_overlay="r64-tokenizer-compat-v1", primary_outputs_observed=False,
             answer_extraction="gsm8k_last_numeric_or_explicit_hash_answer; cmmlu_single_letter",
             files={name:digest(args.output/name) for name in ("primary.json","dev.json")})
    rows=json.loads((args.output/"primary.json").read_text())["rows"]
    new["max_input_tokens"]={b:max(len(r["input_ids"]) for r in rows if r["benchmark"]==b) for b in ("gsm8k","cmmlu")}
    new["max_required_context"]=max(len(r["input_ids"])+r["max_new_tokens"] for r in rows)
    (args.output/"MANIFEST.json").write_text(json.dumps(new,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(new,ensure_ascii=False))


if __name__ == "__main__":
    main()
