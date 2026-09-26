import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_round10_t1_fixtures as builder
import round10_t1_protocol as protocol


class WordTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "CHAT " + messages[0]["content"] + " ASSIST"

    def encode(self, rendered, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(100, 100 + len(rendered.split())))


def inherited_case(length, name):
    content = " ".join([name] * (length - 2))
    messages = [{"role": "user", "content": content}]
    rendered = f"CHAT {content} ASSIST"
    ids = list(range(100, 100 + length))
    digest = lambda value: hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "input_tokens": length,
        "input_ids": ids,
        "input_ids_sha256": digest(ids),
        "messages": messages,
        "messages_sha256": digest(messages),
        "rendered_chat": rendered,
        "rendered_chat_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "retokenized_input_ids_sha256": digest(ids),
        "tokenization_method": "ACTUAL_TOKENIZER_APPLY_CHAT_TEMPLATE",
        "direct_token_edit": False,
        "provenance": "V90_EXACT",
    }


class BuilderTests(unittest.TestCase):
    def test_exact_length_search_is_bounded_and_semantic(self):
        tokenizer = WordTokenizer()
        found, attempts = builder.find_semantic_lengths(
            tokenizer,
            {12, 13},
            prefix="Reply exactly ACK. Context",
            neutral_word="detail",
            max_attempts=20,
        )
        self.assertEqual(set(found), {12, 13})
        self.assertLessEqual(attempts, 20)
        for length, receipt in found.items():
            self.assertEqual(len(receipt["input_ids"]), length)
            self.assertFalse(receipt["direct_token_edit"])
            self.assertIn("Reply exactly ACK", receipt["messages"][0]["content"])

    def test_fixture_inherits_63_64_65_and_builds_all_required_cases(self):
        tokenizer = WordTokenizer()
        source = {
            "model_revision": "d" * 40,
            "tokenizer": {
                "tokenizer_files_sha256": {
                    "tokenizer.json": "a" * 64,
                    "tokenizer_config.json": "b" * 64,
                }
            },
            "cases": {
                "canary27": inherited_case(27, "old27"),
                "boundary63": inherited_case(63, "old63"),
                "boundary64": inherited_case(64, "old64"),
                "boundary65": inherited_case(65, "old65"),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / ("d" * 40)
            model.mkdir()
            (model / "tokenizer.json").write_text("tok")
            (model / "tokenizer_config.json").write_text("cfg")
            result = builder.build_fixture(model, source, tokenizer, max_attempts=400)
        self.assertEqual(result["cases"]["edge63"]["input_ids"], source["cases"]["boundary63"]["input_ids"])
        self.assertEqual(result["cases"]["edge64"]["input_ids"], source["cases"]["boundary64"]["input_ids"])
        self.assertEqual(result["cases"]["edge65"]["input_ids"], source["cases"]["boundary65"]["input_ids"])
        self.assertEqual(len(result["cases"]), len(protocol.CASE_LENGTHS))
        protocol.validate_fixture(result)


if __name__ == "__main__":
    unittest.main()
