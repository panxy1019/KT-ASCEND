"""Build the frozen Round11 T1 semantic fixture with the actual tokenizer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from round11_t1_protocol import CHUNK_TOKENS, MODEL_REVISION, sha256_json, validate_fixture


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def render_and_encode(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if (
        not isinstance(rendered, str)
        or not rendered
        or not isinstance(ids, list)
        or not ids
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise ValueError("actual tokenizer returned an invalid receipt")
    return rendered, ids


def receipt(messages, rendered, ids, *, validator, provenance):
    return {
        "input_tokens": len(ids),
        "input_ids": ids,
        "input_ids_sha256": sha256_json(ids),
        "messages": messages,
        "messages_sha256": sha256_json(messages),
        "rendered_chat": rendered,
        "rendered_chat_sha256": sha256_bytes(rendered.encode()),
        "retokenized_input_ids_sha256": sha256_json(ids),
        "tokenization_method": "ACTUAL_TOKENIZER_APPLY_CHAT_TEMPLATE",
        "direct_token_edit": False,
        "provenance": provenance,
        "validator": copy.deepcopy(validator),
    }


def semantic_receipt(tokenizer, content, *, validator, provenance):
    messages = [{"role": "user", "content": content}]
    rendered, ids = render_and_encode(tokenizer, messages)
    rerendered, reencoded = render_and_encode(tokenizer, messages)
    if rerendered != rendered or reencoded != ids:
        raise ValueError("actual tokenizer did not reproduce the semantic input")
    return receipt(
        messages,
        rendered,
        ids,
        validator=validator,
        provenance=provenance,
    )


def find_semantic_lengths(
    tokenizer,
    targets,
    *,
    prefix,
    neutral_word="detail",
    max_attempts=2048,
    validator,
):
    targets = set(targets)
    found = {}
    attempts = 0
    for count in range(max_attempts):
        content = prefix
        if count:
            content += " " + " ".join([neutral_word] * count)
        messages = [{"role": "user", "content": content}]
        rendered, ids = render_and_encode(tokenizer, messages)
        attempts += 1
        length = len(ids)
        if length in targets and length not in found:
            rerendered, reencoded = render_and_encode(tokenizer, messages)
            if rerendered != rendered or reencoded != ids:
                raise ValueError(f"{length}-token semantic candidate did not reproduce")
            found[length] = receipt(
                messages,
                rendered,
                ids,
                validator=validator,
                provenance="ROUND11_TOKENIZER_DERIVED",
            )
        if set(found) == targets:
            return found, attempts
        if length > max(targets) + 8:
            break
    raise ValueError(
        f"exact semantic lengths unavailable within {attempts} attempts; "
        f"targets={sorted(targets)} found={sorted(found)}"
    )


def inherit_case(tokenizer, source, name):
    item = copy.deepcopy(source["cases"][name])
    rendered, ids = render_and_encode(tokenizer, item["messages"])
    if rendered != item["rendered_chat"] or ids != item["input_ids"]:
        raise ValueError(f"{name} failed exact Round10 actual-tokenizer inheritance")
    item["provenance"] = "ROUND10_HFINAL_EXACT"
    return item


def build_fixture(model, source, tokenizer, *, max_attempts=2048):
    model = Path(model).resolve()
    if model.name != MODEL_REVISION or not model.is_dir():
        raise ValueError("model root must be the frozen revision")
    if source.get("model_revision") != MODEL_REVISION:
        raise ValueError("Round10 source fixture/model revision mismatch")
    tokenizer_hashes = {}
    for name in TOKENIZER_FILES:
        path = model / name
        if not path.is_file():
            raise ValueError(f"required tokenizer file missing: {name}")
        tokenizer_hashes[name] = sha256_bytes(path.read_bytes())
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("actual tokenizer chat template missing")

    cases = {
        name: inherit_case(tokenizer, source, name)
        for name in (
            "canary_marker",
            "canary_math391",
            "canary_json",
            "canary_quote",
            "canary_transform",
            "edge128",
        )
    }
    ack = {
        "kind": "assistant_message_prefix",
        "expected_text": "ACK",
        "expected_token_ids": [3897, 248046],
    }
    edge_prefix = (
        "Reply with exactly ACK. This deterministic local qualification request "
        "contains neutral context only; every following word is inert context."
    )
    targets = {129, 130, 160, 192, 256, 384, 512, 768, 1024}
    longer, attempts = find_semantic_lengths(
        tokenizer,
        targets,
        prefix=edge_prefix,
        neutral_word="detail",
        max_attempts=max_attempts,
        validator=ack,
    )
    for length in sorted(targets):
        cases[f"edge{length}"] = longer[length]
    cases["cancel256"] = copy.deepcopy(cases["edge256"])
    cases["cancel256"]["provenance"] = "ROUND11_CANCEL_FROM_EDGE256"
    cases["cancel256"]["validator"] = {"kind": "abort"}

    result = {
        "schema": 1,
        "status": "FROZEN_ROUND11_T1_ACTUAL_TOKENIZER_FIXTURE",
        "model_revision": MODEL_REVISION,
        "chunk_tokens": CHUNK_TOKENS,
        "source_round10_fixture_sha256": sha256_bytes(
            json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ),
        "tokenizer": {
            "chat_template_sha256": sha256_bytes(template.encode()),
            "tokenizer_files_sha256": tokenizer_hashes,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "add_special_tokens": False,
            "local_files_only": True,
        },
        "construction": {
            "method": "ACTUAL_CHAT_TEMPLATE_BOUNDED_SEMANTIC_SEARCH",
            "attempts": attempts,
            "max_attempts": max_attempts,
            "direct_token_edit": False,
        },
        "cases": cases,
    }
    validate_fixture(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    source = json.loads(args.source.read_text())
    result = build_fixture(
        args.model,
        source,
        tokenizer,
        max_attempts=args.max_attempts,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        "PASS_ROUND11_T1_FIXTURE",
        len(result["cases"]),
        result["construction"]["attempts"],
        sha256_bytes(args.output.read_bytes()),
        flush=True,
    )


if __name__ == "__main__":
    main()
