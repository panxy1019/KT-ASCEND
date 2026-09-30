from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

from round18.native_gdn_graph_boundary import (
    GDN_FULL_FORWARD_GRAPH_API,
    GDN_FULL_FORWARD_SPLITTING_OP,
    ensure_gdn_full_forward_splitting_op,
    install_gdn_full_forward_graph_boundary,
)


class _Tensor:
    def __init__(self, value=None):
        self.value = value


class _FrozenAscendGdn:
    def forward(self, hidden_states, output=None):
        # Frozen-source contract markers exercised by the installer:
        # torch.ops.vllm.qwen_gdn_attention_core
        # core_attn_out = self.norm(core_attn_out, z)
        # out, _ = self.out_proj(core_attn_out)
        # output[:num_tokens] = out
        self.original_calls.append((hidden_states, output))
        produced = _Tensor(("native-gdn", hidden_states.value))
        if output is not None:
            output.value = produced.value
        return produced


def _target_types():
    target = type("TargetGdn", (), {"forward": _FrozenAscendGdn.forward})

    class Layer(target):
        prefix = "language_model.model.layers.0.linear_attn"

        def __init__(self):
            self.original_calls = []

    return target, Layer


class _Torch:
    Tensor = _Tensor

    def __init__(self):
        self.ops = SimpleNamespace(vllm=SimpleNamespace())

    @staticmethod
    def empty_like(_value):
        return _Tensor()


def _source_sha256(cls: type) -> str:
    source = Path(inspect.getsourcefile(cls) or "").read_bytes()
    return hashlib.sha256(source).hexdigest()


class NativeGdnGraphBoundaryTests(unittest.TestCase):
    def test_live_worker_installs_full_boundary_before_model_load(self):
        repository_root = Path(__file__).resolve().parents[2]
        source = (
            repository_root / "round18_integration/native_hybrid_live_worker.py"
        ).read_text()
        install_at = source.index("install_gdn_full_forward_graph_boundary(")
        split_at = source.index("ensure_gdn_full_forward_splitting_op(")
        load_at = source.index("result = super().load_model()")
        self.assertLess(install_at, load_at)
        self.assertLess(split_at, load_at)
        self.assertNotIn("install_gdn_runtime_token_shape_contract", source)

    def test_full_forward_is_opaque_and_executes_frozen_native_once(self):
        torch_module = _Torch()
        target_cls, layer_cls = _target_types()
        layer = layer_cls()
        context = SimpleNamespace(
            no_compile_layers={layer.prefix: layer},
        )
        registrations = []

        def register(**kwargs):
            registrations.append(kwargs)
            setattr(torch_module.ops.vllm, kwargs["op_name"], kwargs["op_func"])

        receipt = install_gdn_full_forward_graph_boundary(
            torch_module=torch_module,
            gdn_cls=target_cls,
            ascend_gdn_cls=_FrozenAscendGdn,
            direct_register_custom_op=register,
            get_forward_context=lambda: context,
            expected_source_sha256=_source_sha256(_FrozenAscendGdn),
        )

        hidden = _Tensor("token-batch")
        result = layer.forward(hidden)

        self.assertEqual(result.value, ("native-gdn", "token-batch"))
        self.assertEqual(layer.original_calls, [(hidden, result)])
        self.assertEqual(len(registrations), 1)
        self.assertEqual(registrations[0]["mutates_args"], ["output"])
        self.assertEqual(receipt["api"], GDN_FULL_FORWARD_GRAPH_API)
        self.assertTrue(receipt["native_forward_preserved"])
        self.assertFalse(receipt["math_changed"])
        self.assertFalse(receipt["state_changed"])

    def test_caller_owned_output_and_idempotent_install(self):
        torch_module = _Torch()
        target_cls, layer_cls = _target_types()
        layer = layer_cls()
        context = SimpleNamespace(no_compile_layers={layer.prefix: layer})
        registrations = []

        def register(**kwargs):
            registrations.append(kwargs)
            setattr(torch_module.ops.vllm, kwargs["op_name"], kwargs["op_func"])

        kwargs = dict(
            torch_module=torch_module,
            gdn_cls=target_cls,
            ascend_gdn_cls=_FrozenAscendGdn,
            direct_register_custom_op=register,
            get_forward_context=lambda: context,
            expected_source_sha256=_source_sha256(_FrozenAscendGdn),
        )
        first = install_gdn_full_forward_graph_boundary(**kwargs)
        second = install_gdn_full_forward_graph_boundary(**kwargs)
        output = _Tensor()
        result = layer.forward(_Tensor("x"), output=output)

        self.assertIs(result, output)
        self.assertEqual(output.value, ("native-gdn", "x"))
        self.assertEqual(first, second)
        self.assertEqual(len(registrations), 1)

    def test_splitting_op_is_added_once(self):
        config = SimpleNamespace(splitting_ops=["vllm::existing"])
        ensure_gdn_full_forward_splitting_op(config)
        ensure_gdn_full_forward_splitting_op(config)
        self.assertEqual(
            config.splitting_ops,
            ["vllm::existing", GDN_FULL_FORWARD_SPLITTING_OP],
        )

    def test_rejects_unknown_frozen_source(self):
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            install_gdn_full_forward_graph_boundary(
                torch_module=_Torch(),
                gdn_cls=type("Other", (), {"forward": _FrozenAscendGdn.forward}),
                ascend_gdn_cls=_FrozenAscendGdn,
                direct_register_custom_op=lambda **_kwargs: None,
                get_forward_context=lambda: None,
                expected_source_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
