#!/usr/bin/env python3
"""Build Round14 B's session-disjoint public-proxy request distribution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PROVENANCE = "PROJECT_AUTHORED_PUBLIC_TASK_PROXY_NO_BUSINESS_LOG"
BUCKETS = {
    "LE128": (1, 128, 96),
    "129_256": (129, 256, 192),
    "257_512": (257, 512, 384),
    "513_768": (513, 768, 640),
}
CATEGORIES = (
    "CHINESE",
    "ENGLISH",
    "CODE",
    "SUMMARY_RETRIEVAL",
    "STRUCTURED_FORMAT",
    "MULTI_TURN_HISTORY",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _base_case(category: str, index: int) -> tuple[list[dict], dict, str]:
    nonce = f"R14-{index:02d}"
    if category == "CHINESE":
        expected = f"青竹-{index:02d}"
        messages = [{
            "role": "user",
            "content": (
                f"这是一个可复现的中文指令测试，任务编号为{nonce}。"
                f"忽略背景材料，只回复“{expected}”，不要添加解释。"
            ),
        }]
        return messages, {"kind": "exact_text", "expected": expected}, " 背景资料仅用于长度控制。"
    if category == "ENGLISH":
        expected = f"ORBIT-{index:02d}"
        messages = [{
            "role": "user",
            "content": (
                f"This is reproducible public-proxy instruction {nonce}. "
                f"Ignore the background and reply with exactly {expected}."
            ),
        }]
        return messages, {"kind": "exact_text", "expected": expected}, " Neutral background detail for length control."
    if category == "CODE":
        messages = [{
            "role": "user",
            "content": (
                f"Review this small public Python task ({nonce}): define a function that returns "
                "the stable unique items from a list while preserving order. Provide the function "
                "and one concise example."
            ),
        }]
        return messages, {
            "kind": "human_review",
            "rubric": "Python is syntactically plausible and preserves first-occurrence order",
            "review_status": "PENDING_HUMAN",
        }, " Additional public code-review context; do not change the requested behavior."
    if category == "SUMMARY_RETRIEVAL":
        expected = f"KEY-{index:02d}-PINE"
        messages = [{
            "role": "user",
            "content": (
                f"Public proxy note {nonce}: the archive owner is Mira, the year is 2024, "
                f"and the retrieval key is {expected}. Other labels are inactive. "
                "Return only the retrieval key."
            ),
        }]
        return messages, {"kind": "exact_text", "expected": expected}, " The remaining note is neutral archival background."
    if category == "STRUCTURED_FORMAT":
        expected = {"id": index, "ok": True}
        messages = [{
            "role": "user",
            "content": (
                f"For public proxy record {nonce}, return one JSON object with integer field "
                f"id equal to {index} and boolean field ok equal to true. Add no other keys."
            ),
        }]
        return messages, {"kind": "json_object", "expected": expected}, " Neutral schema background for length control."
    if category == "MULTI_TURN_HISTORY":
        expected = f"HISTORY-{index:02d}"
        messages = [
            {"role": "user", "content": f"Remember the public test code {expected}."},
            {"role": "assistant", "content": "I will retain that code for this conversation."},
            {"role": "user", "content": f"Conversation {nonce}: return only the code I gave earlier."},
        ]
        return messages, {"kind": "exact_text", "expected": expected}, " Neutral conversation background for length control."
    raise ValueError(category)


def build_specs() -> list[dict]:
    rows = []
    index = 0
    for bucket in BUCKETS:
        for position in range(16):
            category = CATEGORIES[(position + index // 16) % len(CATEGORIES)]
            messages, validator, filler = _base_case(category, index)
            split = "DEV" if position < 4 else "HOLDOUT"
            rows.append({
                "request_id": f"r14-b-{index:02d}",
                "session_id": f"r14-session-{index:02d}",
                "split": split,
                "bucket": bucket,
                "category": category,
                "language": "ZH" if category == "CHINESE" else "EN",
                "messages": messages,
                "validator": validator,
                "filler": filler,
                "provenance": PROVENANCE,
                "normal_max_new_tokens": 64,
                "performance_fixed_output_tokens": 32,
            })
            index += 1
    return rows


def render_and_encode(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if (
        not isinstance(rendered, str) or not rendered
        or not isinstance(ids, list) or not ids
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise ValueError("actual tokenizer returned an invalid receipt")
    return rendered, ids


def fit_messages(tokenizer, spec, *, max_repeats=1024):
    low, high, target = BUCKETS[spec["bucket"]]
    best = None
    for repeats in range(max_repeats + 1):
        messages = copy.deepcopy(spec["messages"])
        messages[-1]["content"] += spec["filler"] * repeats
        rendered, ids = render_and_encode(tokenizer, messages)
        length = len(ids)
        if low <= length <= high:
            score = (abs(length - target), repeats)
            if best is None or score < best[0]:
                best = (score, messages, rendered, ids, repeats)
        if best is not None and length > target + 16:
            break
        if length > high and best is None:
            break
    if best is None:
        raise RuntimeError(f"could not fit {spec['request_id']} into {spec['bucket']}")
    _, messages, rendered, ids, repeats = best
    return messages, rendered, ids, repeats


def _in_bucket(bucket: str, length: int) -> bool:
    low, high, _ = BUCKETS[bucket]
    return low <= length <= high


def validate_fixture(document) -> bool:
    rows = document.get("rows") if isinstance(document, dict) else None
    if document.get("status") != "FROZEN_ROUND14_B_PROXY_FIXTURE" or not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("Round14 B fixture row count/status mismatch")
    sessions = [row.get("session_id") for row in rows]
    if len(set(sessions)) != 64 or any(not value for value in sessions):
        raise ValueError("Round14 B session split leaked or duplicated")
    if sum(row.get("split") == "DEV" for row in rows) != 16 or sum(row.get("split") == "HOLDOUT" for row in rows) != 48:
        raise ValueError("Round14 B DEV/HOLDOUT split mismatch")
    if set(row.get("category") for row in rows) != set(CATEGORIES):
        raise ValueError("Round14 B category coverage mismatch")
    for bucket in BUCKETS:
        selected = [row for row in rows if row.get("bucket") == bucket]
        if len(selected) != 16 or sum(row.get("split") == "DEV" for row in selected) != 4:
            raise ValueError("Round14 B bucket balance mismatch")
    for row in rows:
        ids = row.get("input_ids")
        messages = row.get("messages")
        rendered = row.get("rendered_chat")
        if (
            not isinstance(ids, list) or any(type(token) is not int for token in ids)
            or row.get("input_tokens") != len(ids)
            or not _in_bucket(row.get("bucket"), len(ids))
        ):
            raise ValueError("Round14 B actual token length is outside its bucket")
        if (
            row.get("input_ids_sha256") != sha256_json(ids)
            or row.get("messages_sha256") != sha256_json(messages)
            or row.get("rendered_chat_sha256") != sha256_bytes(rendered.encode())
        ):
            raise ValueError("Round14 B traceability hash mismatch")
        if row.get("provenance") != PROVENANCE:
            raise ValueError("Round14 B provenance mismatch")
    return True


def build(model: Path, tokenizer, *, max_repeats=1024) -> dict:
    model = Path(model)
    rows = []
    for spec in build_specs():
        messages, rendered, ids, repeats = fit_messages(
            tokenizer, spec, max_repeats=max_repeats
        )
        row = {key: value for key, value in spec.items() if key != "filler"}
        row.update({
            "messages": messages,
            "rendered_chat": rendered,
            "input_ids": ids,
            "input_tokens": len(ids),
            "filler_repeats": repeats,
            "input_ids_sha256": sha256_json(ids),
            "messages_sha256": sha256_json(messages),
            "rendered_chat_sha256": sha256_bytes(rendered.encode()),
            "tokenization_method": "ACTUAL_TOKENIZER_APPLY_CHAT_TEMPLATE",
            "direct_token_edit": False,
        })
        rows.append(row)
    document = {
        "schema": 1,
        "status": "FROZEN_ROUND14_B_PROXY_FIXTURE",
        "model_revision": MODEL_REVISION,
        "distribution": {
            "source": PROVENANCE,
            "business_logs_used": False,
            "session_level_split": True,
            "dev": 16,
            "holdout": 48,
            "normal_quality_respects_eos": True,
            "fixed_length_performance_is_separate": True,
        },
        "tokenizer": {
            "tokenizer_json_sha256": sha256(model / "tokenizer.json"),
            "tokenizer_config_sha256": sha256(model / "tokenizer_config.json"),
            "chat_template_sha256": sha256_bytes(tokenizer.chat_template.encode()),
            "local_files_only": True,
            "enable_thinking": False,
            "add_generation_prompt": True,
            "add_special_tokens": False,
        },
        "buckets": {
            name: {"minimum": low, "maximum": high, "construction_target": target}
            for name, (low, high, target) in BUCKETS.items()
        },
        "rows": rows,
    }
    validate_fixture(document)
    return document


def sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-repeats", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    document = build(args.model, tokenizer, max_repeats=args.max_repeats)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(
        "PASS_ROUND14_B_FIXTURE", len(document["rows"]),
        sum(row["split"] == "DEV" for row in document["rows"]),
        sha256(args.output), flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
