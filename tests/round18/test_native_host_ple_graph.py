from __future__ import annotations

from types import SimpleNamespace
import unittest

from round18.native_host_ple_graph import (
    HOST_PLE_GRAPH_API,
    HOST_PLE_SPLITTING_OP,
    ensure_host_ple_splitting_op,
    install_graph_safe_host_ple_bridge,
)


class _Tensor:
    def __init__(self, *, shape=(3, 8), device="npu", dtype="bf16"):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.moves = []
        self.copied = None

    def detach(self):
        return self

    def to(self, **kwargs):
        self.moves.append(dict(kwargs))
        return _Tensor(shape=self.shape, device=kwargs.get("device", self.device), dtype=kwargs.get("dtype", self.dtype))

    def copy_(self, value):
        self.copied = value
        return self


class _PleBase:
    def forward(self, *_args, **_kwargs):
        raise AssertionError("original PLE bridge should be replaced")


class _Embedding(_PleBase):
    def __init__(self, *_args, prefix="", **_kwargs):
        self.embedding_dim = 16
        self.ngram_embedding = SimpleNamespace(weight=SimpleNamespace(dtype="fp8"))
        self.prefix_seen = prefix

    def forward_impl(self, _hidden, input_ids, query_start_loc, ngram_context):
        self.forward_impl_args = (input_ids, query_start_loc, ngram_context)
        return _Tensor(shape=(input_ids.shape[0], self.embedding_dim), device="cpu", dtype="fp8")


class _Ops:
    def __init__(self):
        self.calls = []
        self.vllm = SimpleNamespace(round18_host_ple_embedding=self._call)

    def _call(self, *args):
        self.calls.append(args)


class _Torch:
    Tensor = _Tensor

    def __init__(self):
        self.ops = _Ops()
        self.empty_calls = []

    def empty(self, shape, *, device, dtype):
        self.empty_calls.append((tuple(shape), device, dtype))
        return _Tensor(shape=tuple(shape), device=device, dtype=dtype)


class NativeHostPleGraphTests(unittest.TestCase):
    def test_adds_split_to_normalized_config_once(self):
        config = SimpleNamespace(splitting_ops=["vllm::existing"])
        ensure_host_ple_splitting_op(config)
        ensure_host_ple_splitting_op(config)
        self.assertEqual(
            config.splitting_ops,
            ["vllm::existing", HOST_PLE_SPLITTING_OP],
        )

    def test_registers_opaque_split_and_runs_host_lookup(self):
        registered = {}
        torch = _Torch()
        owner = SimpleNamespace(ple_embedding=None)
        context = SimpleNamespace(no_compile_layers={"layers.7.ple": owner})

        def registrar(**kwargs):
            registered.update(kwargs)

        install_graph_safe_host_ple_bridge(
            torch_module=torch,
            ple_offload_cls=_PleBase,
            ple_embedding_cls=_Embedding,
            direct_register_custom_op=registrar,
            get_forward_context=lambda: context,
        )

        embedding = _Embedding(None, 16, 0, 32, 1, "layers.7.ple.ple_embedding")
        owner.ple_embedding = embedding
        self.assertEqual(embedding.round18_ple_owner_prefix, "layers.7.ple")
        self.assertEqual(registered["op_name"], "round18_host_ple_embedding")
        self.assertEqual(registered["mutates_args"], ["output"])

        hidden = _Tensor(shape=(3, 8), device="npu", dtype="bf16")
        input_ids = _Tensor(shape=(3,), device="npu", dtype="int64")
        starts = _Tensor(shape=(2,), device="npu", dtype="int64")
        context_ids = _Tensor(shape=(1, 3), device="npu", dtype="int64")
        output = _PleBase.forward(embedding, hidden, input_ids, starts, context_ids)

        self.assertEqual(torch.empty_calls, [((3, 16), "npu", "fp8")])
        self.assertIs(torch.ops.calls[0][3], output)
        self.assertEqual(torch.ops.calls[0][4], "layers.7.ple")

        registered["op_func"](input_ids, starts, context_ids, output, "layers.7.ple")
        self.assertEqual(input_ids.moves[-1]["device"], "cpu")
        self.assertEqual(starts.moves[-1]["device"], "cpu")
        self.assertEqual(context_ids.moves[-1]["device"], "cpu")
        self.assertIsNotNone(output.copied)
        self.assertEqual(_PleBase.round18_host_ple_graph_api, HOST_PLE_GRAPH_API)
        self.assertEqual(HOST_PLE_SPLITTING_OP, "vllm::round18_host_ple_embedding")

    def test_rejects_malformed_embedding_prefix(self):
        install_graph_safe_host_ple_bridge(
            torch_module=_Torch(),
            ple_offload_cls=_PleBase,
            ple_embedding_cls=_Embedding,
            direct_register_custom_op=lambda **_kwargs: None,
            get_forward_context=lambda: SimpleNamespace(no_compile_layers={}),
        )
        with self.assertRaisesRegex(ValueError, "ple_embedding"):
            _Embedding(None, 16, 0, 32, 1, "layers.7.bad")


if __name__ == "__main__":
    unittest.main()
