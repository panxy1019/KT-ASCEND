"""Build the frozen Round10 T1 semantic fixture with the actual tokenizer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from round10_t1_protocol import (
    CHUNK_TOKENS,
    THRESHOLD_TOKENS,
    canonical_bytes,
    sha256_json,
    validate_fixture,
)


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
        "validator": validator,
    }


def semantic_receipt(tokenizer, content, *, validator, provenance="TOKENIZER_DERIVED"):
    messages = [{"role": "user", "content": content}]
    rendered, ids = render_and_encode(tokenizer, messages)
    rerendered, reencoded = render_and_encode(tokenizer, messages)
    if rerendered != rendered or reencoded != ids:
        raise ValueError("actual tokenizer did not reproduce the semantic input")
    return receipt(
        messages,
        rendered,
        ids,
        validator=copy.deepcopy(validator),
        provenance=provenance,
    )


def find_semantic_lengths(
    tokenizer,
    targets,
    *,
    prefix,
    neutral_word="detail",
    max_attempts=512,
    validator=None,
):
    targets = set(targets)
    if not targets or any(type(value) is not int or value < 1 for value in targets):
        raise ValueError("positive exact target lengths required")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("positive bounded attempt limit required")
    if validator is None:
        validator = {
            "kind": "assistant_message_prefix",
            "expected_text": "ACK",
            "expected_token_ids": [3897, 248046],
        }
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
                validator=copy.deepcopy(validator),
                provenance="TOKENIZER_DERIVED",
            )
        if set(found) == targets:
            return found, attempts
        if length > max(targets) + 8:
            break
    raise ValueError(
        f"exact semantic lengths unavailable within {attempts} attempts; "
        f"targets={sorted(targets)} found={sorted(found)}"
    )


def _inherit_case(tokenizer, source, source_name, validator):
    item = copy.deepcopy(source["cases"][source_name])
    messages = item.get("messages")
    rendered, ids = render_and_encode(tokenizer, messages)
    if rendered != item.get("rendered_chat") or ids != item.get("input_ids"):
        raise ValueError(f"{source_name} failed exact actual-tokenizer inheritance")
    item["validator"] = copy.deepcopy(validator)
    item["provenance"] = "V90_EXACT"
    return item


def build_fixture(model, source, tokenizer, *, max_attempts=512):
    model = Path(model).resolve()
    revision = model.name
    if (
        not model.is_dir()
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("model root must end in the frozen 40-hex revision")
    if type(max_attempts) is not int or not 32 <= max_attempts <= 4096:
        raise ValueError("bounded attempt limit must be 32..4096")
    tokenizer_hashes = {}
    for name in TOKENIZER_FILES:
        path = model / name
        if not path.is_file():
            raise ValueError(f"required tokenizer file missing: {name}")
        tokenizer_hashes[name] = sha256_bytes(path.read_bytes())
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("actual tokenizer chat template missing")
    if source.get("model_revision") != revision:
        raise ValueError("v90 fixture/model revision mismatch")

    exact = lambda text: {"kind": "exact_text", "expected": text}
    ack = {
        "kind": "assistant_message_prefix",
        "expected_text": "ACK",
        "expected_token_ids": [3897, 248046],
    }
    canary_prompts = {
        "canary_marker": (
            "Reply with exactly R10-HYBRID-OK and no other text.",
            exact("R10-HYBRID-OK"),
        ),
        "canary_json": (
            'Return exactly this JSON with no spaces or code fence: {"ok":true}',
            exact('{"ok":true}'),
        ),
        "canary_quote": (
            "The following word is quoted: 'cobalt'. Reply with exactly cobalt and no punctuation.",
            exact("cobalt"),
        ),
        "canary_transform": (
            "Convert abcXYZ to uppercase. Reply with only the converted text.",
            exact("ABCXYZ"),
        ),
    }
    cases = {
        name: semantic_receipt(
            tokenizer, content, validator=validator, provenance="ROUND10_CANARY"
        )
        for name, (content, validator) in canary_prompts.items()
    }
    cases["canary_math391"] = _inherit_case(
        tokenizer, source, "canary27", exact("391")
    )
    for name, source_name in (
        ("edge63", "boundary63"),
        ("edge64", "boundary64"),
        ("edge65", "boundary65"),
    ):
        cases[name] = _inherit_case(tokenizer, source, source_name, ack)

    attempts = len(canary_prompts) + 4
    edge_prefix = (
        "Reply with exactly ACK. This is a deterministic local qualification "
        "request whose neutral context carries no instruction changes."
    )
    longer, used = find_semantic_lengths(
        tokenizer,
        {127, 128, 129, 130},
        prefix=edge_prefix,
        neutral_word="detail",
        max_attempts=max_attempts - attempts,
        validator=ack,
    )
    attempts += used
    for length, name in ((127, "edge127"), (128, "edge128"), (129, "policy129"), (130, "policy130")):
        cases[name] = longer[length]

    decode_prefix = (
        "List the first twelve positive even integers in order, separated by "
        "commas, then state that the list is complete."
    )
    decode, used = find_semantic_lengths(
        tokenizer,
        {64},
        prefix=decode_prefix,
        neutral_word="context",
        max_attempts=max_attempts - attempts,
        validator={"kind": "multi_step", "minimum_tokens": 8},
    )
    attempts += used
    cases["decode64"] = decode[64]
    cases["cancel128"] = copy.deepcopy(cases["edge128"])
    cases["cancel128"]["provenance"] = "ROUND10_CANCEL_FROM_EDGE128"
    cases["cancel128"]["validator"] = {"kind": "abort"}

    result = {
        "schema": 1,
        "status": "FROZEN_ROUND10_T1_ACTUAL_TOKENIZER_FIXTURE",
        "model_revision": revision,
        "threshold_tokens": THRESHOLD_TOKENS,
        "chunk_tokens": CHUNK_TOKENS,
        "source_v90_fixture_sha256": sha256_bytes(canonical_bytes(source)),
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
    parser.add_argument("--v90-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.v90_fixture.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model.resolve()), local_files_only=True, trust_remote_code=False
    )
    document = build_fixture(
        args.model, source, tokenizer, max_attempts=args.max_attempts
    )
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print("PASS_ROUND10_T1_FIXTURE", document["construction"]["attempts"], flush=True)


if __name__ == "__main__":
    main()
