"""Isolated contracts for the Round9 Qwen4 model overlay.

The recovered f02 package requires accelerator-only imports at module import time.
These tests compile the small pure adapter functions from the model source, so the
same contracts remain executable in the CPU-only development environment.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Set, Tuple, Union

import torch


MODEL = (
    Path(__file__).parents[1]
    / "sglang_overlay/python/sglang/srt/models/qwen4_exp.py"
)
FUNCTIONS = {
    "_owned_cpu_staging",
    "_speculation_is_active",
    "_mean_in_storage_dtype",
    "_flatten_token_kv",
    "_qsa_rope_settings",
    "_normalize_checkpoint_name",
    "_ple_shard_index",
    "_validate_packed_parts",
}
CONSTANTS = {"_PLE_BUFFER_RENAMES", "_PLE_BUFFER_SUFFIXES"}


def _load_contracts():
    tree = ast.parse(MODEL.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & CONSTANTS:
                selected.append(node)
    namespace = {
        "torch": torch,
        "re": re,
        "Dict": Dict,
        "Optional": Optional,
        "Set": Set,
        "Tuple": Tuple,
        "Union": Union,
    }
    exec(compile(ast.Module(selected, type_ignores=[]), str(MODEL), "exec"), namespace)
    missing = FUNCTIONS - namespace.keys()
    if missing:
        raise AssertionError(f"missing model contract functions: {sorted(missing)}")
    return namespace


C = _load_contracts()


class _SpecAlgorithm:
    def __init__(self, none: bool):
        self.none = none

    def is_none(self):
        return self.none


class Qwen4OverlayContractTests(unittest.TestCase):
    def test_cpu_staging_owns_storage_preserves_bits_and_packs_views(self):
        source = torch.tensor([[0., -0., 2.], [3., 4., 5.]], dtype=torch.bfloat16)
        view = source.T
        staged = C["_owned_cpu_staging"](view)
        self.assertTrue(staged.is_contiguous())
        self.assertNotEqual(staged.data_ptr(), source.data_ptr())
        self.assertTrue(torch.equal(staged.view(torch.uint8), view.contiguous().view(torch.uint8)))
        staged[0, 0] = 9
        self.assertEqual(float(source[0, 0]), 0.)
        meta = torch.empty(2, device="meta")
        self.assertIs(C["_owned_cpu_staging"](meta), meta)

    def test_speculation_none_enum_is_allowed(self):
        active = C["_speculation_is_active"]
        self.assertFalse(active(None))
        self.assertFalse(active(_SpecAlgorithm(True)))
        self.assertTrue(active(_SpecAlgorithm(False)))
        self.assertTrue(active(object()), "an unknown non-null enum must fail closed")

    def test_qsa_mean_restores_storage_dtype_before_norm(self):
        values = torch.tensor(
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]],
            dtype=torch.bfloat16,
        )
        result = C["_mean_in_storage_dtype"](values)
        expected = values.float().mean(0).to(torch.bfloat16)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result, expected))
        with self.assertRaises(TypeError):
            C["_mean_in_storage_dtype"](torch.ones(4, 2, dtype=torch.long))

    def test_npu_page_kv_is_flattened_to_token_slots(self):
        paged = torch.arange(3 * 4 * 2 * 5).reshape(3, 4, 2, 5)
        flattened = C["_flatten_token_kv"](paged, 2, 5)
        self.assertEqual(flattened.shape, (12, 2, 5))
        self.assertTrue(torch.equal(flattened[7], paged[1, 3]))

        already_flat = paged.reshape(12, 2, 5)
        self.assertTrue(
            torch.equal(C["_flatten_token_kv"](already_flat, 2, 5), already_flat)
        )
        with self.assertRaises(RuntimeError):
            C["_flatten_token_kv"](paged, 1, 5)

    def test_qsa_rope_uses_nested_checkpoint_parameters(self):
        config = SimpleNamespace(
            head_dim=256,
            hidden_size=2560,
            num_attention_heads=10,
            rope_parameters={
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
            }
        )
        self.assertEqual(C["_qsa_rope_settings"](config, 128), (64, 10_000_000.0))

    def test_real_checkpoint_names_map_to_model_destinations(self):
        normalize = C["_normalize_checkpoint_name"]
        self.assertEqual(
            normalize(
                "model.language_model.layers.24.self_attn.indexer.index_qk_proj.weight"
            ),
            "model.layers.24.indexer.index_qk_proj.weight",
        )
        for checkpoint_suffix, runtime_suffix in (
            ("layer_multipliers", "layer_multipliers"),
            ("ngram_heads_offsets", "ngram_heads_offsets"),
            ("ngram_heads_vocab_sizes", "ngram_heads_vocab_sizes"),
        ):
            self.assertEqual(
                normalize(
                    "model.language_model.layers.1.ple.ple_embedding."
                    + checkpoint_suffix
                ),
                "model.layers.1.ple." + runtime_suffix,
            )
        for component in (
            "hc_norm",
            "input_mix_weight_down",
            "input_mix_weight_up",
            "block_inject_weight",
        ):
            self.assertEqual(
                normalize(
                    "model.language_model.layers.46.attn_hyper_connection."
                    f"{component}.weight"
                ),
                f"model.layers.46.attn_hyper_connection.{component}.weight",
            )
        for component in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
            self.assertEqual(
                normalize(
                    "model.language_model.hyper_connection_mixer."
                    f"{component}.weight"
                ),
                f"model.hyper_connection_mixer.{component}.weight",
            )

    def test_ple_shard_parser_rejects_malformed_layout(self):
        parse = C["_ple_shard_index"]
        key = (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_127.weight"
        )
        self.assertEqual(parse(key, 1), 127)
        self.assertIsNone(parse("model.language_model.layers.0.mlp.gate.weight"))
        with self.assertRaises(KeyError):
            parse(key, 2)
        with self.assertRaises(KeyError):
            parse(key.removesuffix(".weight"))

    def test_partial_packed_load_cannot_qualify(self):
        validate = C["_validate_packed_parts"]
        validate(
            {
                "model.layers.0.qkv_proj.weight": {"q", "k", "v"},
                "model.layers.0.mlp.shared_expert.gate_up_proj.weight": {0, 1},
            }
        )
        with self.assertRaisesRegex(RuntimeError, "qkv_proj"):
            validate({"model.layers.0.qkv_proj.weight": {"q", "v"}})
        with self.assertRaisesRegex(RuntimeError, "gate_up_proj"):
            validate({"model.layers.0.mlp.shared_expert.gate_up_proj.weight": {0}})

    def test_runtime_paths_call_the_contract_helpers(self):
        source = MODEL.read_text()
        self.assertIn("_speculation_is_active(", source)
        self.assertIn("pooled = _mean_in_storage_dtype(", source)
        self.assertIn("pool.get_key_buffer(self.attn.layer_id)", source)
        self.assertIn("pool.get_value_buffer(self.attn.layer_id)", source)
        self.assertIn("pool.reset_qsa_request(self.layer_id, int(req_slot))", source)
        self.assertIn("pool.get_qsa_rope_position_buffer(member_locs)", source)
        self.assertIn("_validate_packed_parts(audit.packed_parts)", source)
        self.assertIn("audit.ple_shard_rows != expected_ple_rows", source)


if __name__ == "__main__":
    unittest.main()
