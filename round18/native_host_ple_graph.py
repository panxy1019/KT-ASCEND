"""Graph-visible, eager-executed boundary for the frozen Host PLE lookup.

The frozen Qwen image keeps the 95 GiB PLE embedding on Host.  Its original
Round16 bridge is mathematically correct in eager mode, but exposes ordinary
Python tensor inspection to Dynamo.  PIECEWISE compilation requires the Host
lookup to be an opaque custom operation so that the surrounding native NPU
regions can be partitioned without capturing the Host work itself.
"""

from __future__ import annotations

from typing import Any, Callable


HOST_PLE_GRAPH_API = "round18:host-ple-opaque-split:v1"
HOST_PLE_OP_NAME = "round18_host_ple_embedding"
HOST_PLE_SPLITTING_OP = f"vllm::{HOST_PLE_OP_NAME}"
_PLE_SUFFIX = ".ple_embedding"


def _owner_prefix(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix.endswith(_PLE_SUFFIX):
        raise ValueError(
            "Round18 Host PLE prefix must end in '.ple_embedding': "
            f"{prefix!r}"
        )
    return prefix[: -len(_PLE_SUFFIX)]


def ensure_host_ple_splitting_op(compilation_config: Any) -> None:
    """Add the opaque Host lookup to an already normalized PIECEWISE config."""

    splitting_ops = getattr(compilation_config, "splitting_ops", None)
    if splitting_ops is None:
        splitting_ops = []
        compilation_config.splitting_ops = splitting_ops
    if HOST_PLE_SPLITTING_OP not in splitting_ops:
        splitting_ops.append(HOST_PLE_SPLITTING_OP)


def install_graph_safe_host_ple_bridge(
    *,
    torch_module: Any,
    ple_offload_cls: type,
    ple_embedding_cls: type,
    direct_register_custom_op: Callable[..., None],
    get_forward_context: Callable[[], Any],
) -> None:
    """Replace only the Host lookup call with an opaque PIECEWISE split op.

    Weight ownership and ``forward_impl`` remain on the frozen PLE embedding.
    The custom op transfers its small request inputs to Host, runs that exact
    implementation, and publishes the resulting embedding into a caller-owned
    device tensor.  NPU key/value projection, gating, normalization and short
    convolution remain in the native model.
    """

    installed = getattr(ple_offload_cls, "round18_host_ple_graph_api", None)
    if installed not in (None, HOST_PLE_GRAPH_API):
        raise RuntimeError("another graph Host PLE bridge is already installed")
    if installed == HOST_PLE_GRAPH_API:
        return

    original_init = ple_embedding_cls.__init__

    def tagged_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix")
        if prefix is None:
            if len(args) < 6:
                raise ValueError("PLE embedding construction omitted its prefix")
            prefix = args[5]
        self.round18_ple_owner_prefix = _owner_prefix(prefix)
        weight = self.ngram_embedding.weight
        self.round18_ple_output_dtype = weight.dtype

    tagged_init.round18_api_identity = HOST_PLE_GRAPH_API
    ple_embedding_cls.__init__ = tagged_init

    def host_ple_embedding(
        input_ids: Any,
        query_start_loc: Any,
        ngram_context: Any,
        output: Any,
        layer_name: str,
    ) -> None:
        context = get_forward_context()
        layer = context.no_compile_layers[layer_name]
        embedding = layer.ple_embedding
        host_input_ids = input_ids.detach().to(device="cpu", non_blocking=False)
        host_query_start_loc = query_start_loc.detach().to(
            device="cpu", non_blocking=False
        )
        host_ngram_context = ngram_context.detach().to(
            device="cpu", non_blocking=False
        )
        result = embedding.forward_impl(
            host_input_ids,
            host_input_ids,
            host_query_start_loc,
            host_ngram_context,
        )
        output.copy_(
            result.to(
                device=output.device,
                dtype=output.dtype,
                non_blocking=False,
            )
        )

    def host_ple_embedding_fake(
        input_ids: Any,
        query_start_loc: Any,
        ngram_context: Any,
        output: Any,
        layer_name: str,
    ) -> None:
        del input_ids, query_start_loc, ngram_context, output, layer_name

    tensor_type = torch_module.Tensor
    tensor_annotations = {
        "input_ids": tensor_type,
        "query_start_loc": tensor_type,
        "ngram_context": tensor_type,
        "output": tensor_type,
        "layer_name": str,
        "return": None,
    }
    host_ple_embedding.__annotations__ = dict(tensor_annotations)
    host_ple_embedding_fake.__annotations__ = dict(tensor_annotations)

    direct_register_custom_op(
        op_name=HOST_PLE_OP_NAME,
        op_func=host_ple_embedding,
        mutates_args=["output"],
        fake_impl=host_ple_embedding_fake,
    )

    def forward(
        self: Any,
        hidden_states: Any,
        input_ids: Any,
        query_start_loc: Any,
        ngram_context: Any,
        output_buffer: Any | None = None,
    ) -> Any:
        if output_buffer is not None:
            raise RuntimeError(
                "Round18 graph Host PLE owns its fixed-shape output tensor"
            )
        output = torch_module.empty(
            (hidden_states.shape[0], self.embedding_dim),
            device=hidden_states.device,
            dtype=self.round18_ple_output_dtype,
        )
        getattr(torch_module.ops.vllm, HOST_PLE_OP_NAME)(
            input_ids,
            query_start_loc,
            ngram_context,
            output,
            self.round18_ple_owner_prefix,
        )
        return output

    forward.round18_api_identity = HOST_PLE_GRAPH_API
    ple_offload_cls.forward = forward
    ple_offload_cls.round18_host_ple_graph_api = HOST_PLE_GRAPH_API


__all__ = [
    "HOST_PLE_GRAPH_API",
    "HOST_PLE_OP_NAME",
    "HOST_PLE_SPLITTING_OP",
    "ensure_host_ple_splitting_op",
    "install_graph_safe_host_ple_bridge",
]
