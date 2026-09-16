"""Offline, model-local loader overlay for Transformers 5.3 ByteLevel regression.

Does not edit the read-only model or replace tokenizer bytes. Explicitly records
the sole tokenizer_config class change and verifies the resulting pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

TOKENIZER_SHA = "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    import transformers
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert sha(args.model / "tokenizer.json") == TOKENIZER_SHA
    config = json.loads((args.model / "tokenizer_config.json").read_text())
    assert config["tokenizer_class"] == "LlamaTokenizerFast"
    raw = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    original = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    fast = PreTrainedTokenizerFast.from_pretrained(str(args.model), local_files_only=True)
    raw_json = json.loads(raw.to_str())
    fast_json = json.loads(fast.backend_tokenizer.to_str())
    for key in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder", "added_tokens"):
        assert raw_json[key] == fast_json[key], f"PIPELINE_MISMATCH:{key}"
    samples = ["hello world", "\n\nAnswer: 72", "中文 空格与换行\n答案：A", "def f(x):\n    return x + 1", "#### -1,234.50"]
    probes = []
    for text in samples:
        ids = raw.encode(text).ids
        assert fast.encode(text) == ids
        decoded = raw.decode(ids)
        assert fast.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) == decoded
        probes.append(dict(text=text, ids=ids, raw_decode=decoded,
                           old_ids=original.encode(text), old_decode=original.decode(ids, skip_special_tokens=True)))
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.model / "tokenizer.json", args.output / "tokenizer.json")
    for name in ("special_tokens_map.json", "config.json"):
        if (args.model / name).exists():
            shutil.copyfile(args.model / name, args.output / name)
    patched = dict(config, tokenizer_class="PreTrainedTokenizerFast")
    (args.output / "tokenizer_config.json").write_text(json.dumps(patched, ensure_ascii=False, indent=2)+"\n")
    loaded = AutoTokenizer.from_pretrained(str(args.output), local_files_only=True)
    for probe in probes:
        assert loaded.encode(probe["text"]) == probe["ids"]
        assert loaded.decode(probe["ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) == probe["raw_decode"]
    assert original.eos_token_id == fast.eos_token_id == loaded.eos_token_id == 100001
    report = dict(status="TOKENIZER_ONLY_VERIFIED_SERVING_NOT_YET_VERIFIED", transformers=transformers.__version__,
                  source_model=str(args.model), original_class=type(original).__name__, new_class=type(loaded).__name__,
                  tokenizer_sha256=TOKENIZER_SHA, source_config_sha256=sha(args.model/"tokenizer_config.json"),
                  overlay_config_sha256=sha(args.output/"tokenizer_config.json"), changed_keys=["tokenizer_class"],
                  eos_token_id=loaded.eos_token_id, probes=probes,
                  source="https://github.com/huggingface/transformers/issues/45488")
    (args.output / "compatibility.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
