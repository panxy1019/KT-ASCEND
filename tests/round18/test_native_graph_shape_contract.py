from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

from round18.native_graph_shape_contract import (
    GDN_RUNTIME_TOKEN_SHAPE_API,
    GDN_STATIC_BA_SPLIT_API,
    ROTARY_STATIC_COS_SIN_SPLIT_API,
    install_static_gdn_ba_split,
    install_gdn_runtime_token_shape_contract,
    install_static_rotary_cos_sin_split,
)


class _SliceTensor:
    def __init__(self, values):
        self.values = tuple(values)

    def __getitem__(self, key):
        self.last_key = key
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in (Ellipsis, slice(None))
        ):
            raise AssertionError(f"unexpected tensor key: {key!r}")
        return _SliceTensor(self.values[key[1]])

    @property
    def shape(self):
        return (1, len(self.values))

    def contiguous(self):
        return self

    def reshape(self, shape):
        self.reshape_target = shape
        return self


class _RotaryCache:
    def __getitem__(self, _positions):
        return _SliceTensor(range(64))


class _FrozenGdn:
    num_v_heads = 48
    tp_size = 1
    tp_rank = 0
    disable_tp_for_ba_proj = False

    def split_ba(self, ba):
        b, a = ba.chunk(2, dim=-1)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a


class _FrozenRotary:
    _ASCEND_TRITON_GRID_LIMIT = 65535
    rotary_dim = 64
    head_size = 128
    mrope_section = [16, 24, 24]
    mrope_interleaved = True
    cos_sin_cache = _RotaryCache()

    def _match_cos_sin_cache_dtype(self, query):
        self.matched_query = query

    def forward_triton(self, positions, query, key=None, offsets=None):
        assert positions.ndim == 2
        assert key is not None
        self._match_cos_sin_cache_dtype(query)
        self.cos = None
        self.sin = None
        if self.cos is None and self.sin is None:
            cos_sin = self.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            self.cos = cos.contiguous()
            self.sin = sin.contiguous()
        query_shape = query.shape
        key_shape = key.shape
        assert self.mrope_section
        q, k = triton_mrope(
            query,
            key,
            self.cos,
            self.sin,
            self.mrope_section,
            self.head_size,
            self.rotary_dim,
            self.mrope_interleaved,
        )
        return q.reshape(query_shape), k.reshape(key_shape)


class _FrozenAscendGdn:
    def forward(self, hidden_states, output=None):
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        num_tokens = mixed_qkvz.size(0)
        return num_tokens


class _TargetGdn:
    forward = _FrozenAscendGdn.forward


def _source_sha256(cls: type) -> str:
    source = Path(inspect.getsourcefile(cls) or "").read_bytes()
    return hashlib.sha256(source).hexdigest()


class NativeGraphShapeContractTests(unittest.TestCase):
    def test_gdn_runtime_token_contract_rebinds_exact_ascend_forward(self):
        module = SimpleNamespace(
            torch=SimpleNamespace(),
            triton=SimpleNamespace(),
            fused_qkvzba_split_reshape_cat=lambda *args: args,
            rearrange=lambda value, _pattern: value,
        )
        receipt = install_gdn_runtime_token_shape_contract(
            gdn_cls=_TargetGdn,
            ascend_gdn_cls=_FrozenAscendGdn,
            ascend_gdn_module=module,
            expected_source_sha256=_source_sha256(_FrozenAscendGdn),
        )
        self.assertEqual(receipt["api"], GDN_RUNTIME_TOKEN_SHAPE_API)
        self.assertFalse(receipt["math_changed"])
        self.assertFalse(receipt["kv_contract_claimed"])
        patched_source = inspect.getsource(_TargetGdn.forward)
        self.assertIn("num_tokens = z.size(0)", patched_source)
        self.assertNotIn("num_tokens = mixed_qkvz.size(0)", patched_source)
        self.assertEqual(
            receipt,
            install_gdn_runtime_token_shape_contract(
                gdn_cls=_TargetGdn,
                ascend_gdn_cls=_FrozenAscendGdn,
                ascend_gdn_module=module,
                expected_source_sha256=_source_sha256(_FrozenAscendGdn),
            ),
        )

    def test_static_split_preserves_tp1_halves_without_chunk(self):
        receipt = install_static_gdn_ba_split(
            gdn_cls=_FrozenGdn,
            expected_source_sha256=_source_sha256(_FrozenGdn),
        )
        layer = _FrozenGdn()
        b, a = layer.split_ba(_SliceTensor(range(96)))
        self.assertEqual(b.values, tuple(range(48)))
        self.assertEqual(a.values, tuple(range(48, 96)))
        self.assertEqual(receipt["status"], "INSTALLED_STATIC_GDN_BA_SPLIT")
        self.assertEqual(receipt["api"], GDN_STATIC_BA_SPLIT_API)
        self.assertEqual(receipt["projected_half_width"], "num_v_heads/tp_size")

    def test_preserves_replicated_projection_rank_slice(self):
        class Replicated(_FrozenGdn):
            num_v_heads = 48
            tp_size = 4
            tp_rank = 2
            disable_tp_for_ba_proj = True

            def split_ba(self, ba):
                b, a = ba.chunk(2, dim=-1)
                if self.disable_tp_for_ba_proj and self.tp_size > 1:
                    ba_chunk = self.num_v_heads // self.tp_size
                    ba_start = self.tp_rank * ba_chunk
                    b = b[:, ba_start : ba_start + ba_chunk]
                    a = a[:, ba_start : ba_start + ba_chunk]
                return b, a

        install_static_gdn_ba_split(
            gdn_cls=Replicated,
            expected_source_sha256=_source_sha256(Replicated),
        )
        b, a = Replicated().split_ba(_SliceTensor(range(96)))
        self.assertEqual(b.values, tuple(range(24, 36)))
        self.assertEqual(a.values, tuple(range(72, 84)))

    def test_is_idempotent_and_rejects_source_or_patch_conflicts(self):
        digest = _source_sha256(_FrozenGdn)
        first = install_static_gdn_ba_split(
            gdn_cls=_FrozenGdn,
            expected_source_sha256=digest,
        )
        second = install_static_gdn_ba_split(
            gdn_cls=_FrozenGdn,
            expected_source_sha256=digest,
        )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            install_static_gdn_ba_split(
                gdn_cls=type("Other", (), {"split_ba": lambda self, value: value}),
                expected_source_sha256="0" * 64,
            )

        class Foreign(_FrozenGdn):
            pass

        Foreign.round18_static_ba_split_api = "foreign"
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            install_static_gdn_ba_split(
                gdn_cls=Foreign,
                expected_source_sha256=_source_sha256(Foreign),
            )

    def test_rotary_split_uses_static_rotary_width_and_preserves_call(self):
        calls = []

        def triton_mrope(*args):
            calls.append(args)
            return args[0], args[1]

        module = SimpleNamespace(triton_mrope=triton_mrope, os=SimpleNamespace(environ={}))
        receipt = install_static_rotary_cos_sin_split(
            rotary_cls=_FrozenRotary,
            rotary_module=module,
            expected_source_sha256=_source_sha256(_FrozenRotary),
        )
        layer = _FrozenRotary()
        positions = SimpleNamespace(ndim=2)
        query = _SliceTensor(range(8))
        key = _SliceTensor(range(6))
        q, k = layer.forward_triton(positions, query, key)

        self.assertIs(q, query)
        self.assertIs(k, key)
        self.assertEqual(layer.cos.values, tuple(range(32)))
        self.assertEqual(layer.sin.values, tuple(range(32, 64)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(receipt["api"], ROTARY_STATIC_COS_SIN_SPLIT_API)
        self.assertFalse(receipt["kv_contract_claimed"])


if __name__ == "__main__":
    unittest.main()
