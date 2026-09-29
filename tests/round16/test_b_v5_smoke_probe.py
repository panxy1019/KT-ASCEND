from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from round16_integration.b_v5_smoke_probe import (
    extract_finite_selected_logprobs,
    freeze_canary_inputs,
    validate_known_answer,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.encode_calls: list[tuple[str, bool]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.calls.append(
            (
                tuple((item["role"], item["content"]) for item in messages),
                tokenize,
                add_generation_prompt,
                enable_thinking,
            )
        )
        assert tokenize is False
        return f"<chat>{messages[0]['content']}</chat>"

    def encode(self, rendered, *, add_special_tokens):
        self.encode_calls.append((rendered, add_special_tokens))
        prompt = rendered.removeprefix("<chat>").removesuffix("</chat>")
        return [101, len(prompt), 202]


class V5SmokeProbeTests(unittest.TestCase):
    def test_exact_text_and_json_are_strict(self):
        validate_known_answer(
            "391", {"kind": "exact_text", "expected": "391"}
        )
        with self.assertRaisesRegex(RuntimeError, "exact_text"):
            validate_known_answer(
                "391\n", {"kind": "exact_text", "expected": "391"}
            )

        validate_known_answer(
            '{"count": 3, "name": "apple"}',
            {
                "kind": "exact_json",
                "expected": {"name": "apple", "count": 3},
            },
        )
        with self.assertRaisesRegex(RuntimeError, "exact_json"):
            validate_known_answer(
                '{"name": "apple", "count": 4}',
                {
                    "kind": "exact_json",
                    "expected": {"name": "apple", "count": 3},
                },
            )

    def test_selected_logprobs_require_every_token_and_finite_value(self):
        values = extract_finite_selected_logprobs(
            [11, 12],
            [
                {11: SimpleNamespace(logprob=-0.25)},
                {12: SimpleNamespace(logprob=-1.5)},
            ],
        )
        self.assertEqual(values, [-0.25, -1.5])

        with self.assertRaisesRegex(RuntimeError, "count"):
            extract_finite_selected_logprobs(
                [11, 12], [{11: SimpleNamespace(logprob=-0.25)}]
            )
        with self.assertRaisesRegex(RuntimeError, "selected token"):
            extract_finite_selected_logprobs(
                [11], [{99: SimpleNamespace(logprob=-0.25)}]
            )
        with self.assertRaisesRegex(RuntimeError, "finite"):
            extract_finite_selected_logprobs(
                [11], [{11: SimpleNamespace(logprob=math.inf)}]
            )

    def test_freeze_canaries_uses_runtime_template_and_encodes_all_inputs(self):
        tokenizer = _Tokenizer()
        contract = {
            "canaries": [
                {
                    "id": "one",
                    "prompt": "first",
                    "validator": {"kind": "exact_text", "expected": "1"},
                },
                {
                    "id": "two",
                    "prompt": "second",
                    "validator": {"kind": "exact_text", "expected": "2"},
                },
            ]
        }

        frozen = freeze_canary_inputs(tokenizer, contract)

        self.assertEqual([item["id"] for item in frozen], ["one", "two"])
        self.assertEqual(frozen[0]["input_ids"], [101, 5, 202])
        self.assertEqual(frozen[1]["input_ids"], [101, 6, 202])
        self.assertEqual(
            tokenizer.calls,
            [
                ((('user', 'first'),), False, True, False),
                ((('user', 'second'),), False, True, False),
            ],
        )
        self.assertEqual(
            tokenizer.encode_calls,
            [("<chat>first</chat>", False), ("<chat>second</chat>", False)],
        )


if __name__ == "__main__":
    unittest.main()
