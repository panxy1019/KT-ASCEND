"""Read-only exact-source preflight for the frozen vLLM-Ascend image.

This file intentionally depends only on the Python standard library.  It can
be copied into the donor image and run before importing vLLM, torch, or
torch_npu.  The preflight is disabled by default and never mutates the source
tree it inspects.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TARGET_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
AUTO_WEIGHTS_LOADER_CONSUMER_STATUS = (
    "AUTO_WEIGHTS_LOADER_CONSUMER_SOURCE_BOUND"
)
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_CONFIG_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class SourceExpectation:
    name: str
    package: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.package not in ("vllm", "vllm_ascend"):
            raise ValueError("source package must be vllm or vllm_ascend")
        if not self.name or not self.relative_path:
            raise ValueError("source expectation names and paths must be non-empty")
        relative = Path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source expectation path must remain package-relative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("source expectation sha256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class ModelConfigExpectation:
    sha256: str
    architecture: str
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    dtype: str = "bfloat16"
    minimum_context_capacity: int = 256

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("model config sha256 must be lowercase hexadecimal")
        if not self.architecture:
            raise ValueError("model config architecture must be non-empty")
        for value in (
            self.num_hidden_layers,
            self.num_experts,
            self.num_experts_per_tok,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("model config dimensions must be positive integers")
        if type(self.dtype) is not str or not self.dtype:
            raise ValueError("model config dtype must be a non-empty string")
        if (
            type(self.minimum_context_capacity) is not int
            or self.minimum_context_capacity < 256
        ):
            raise ValueError("model context capacity contract must be at least 256")


@dataclass(frozen=True)
class PreflightPaths:
    vllm_package_root: Path
    vllm_ascend_package_root: Path
    model_config: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "vllm_package_root", Path(self.vllm_package_root))
        object.__setattr__(
            self, "vllm_ascend_package_root", Path(self.vllm_ascend_package_root)
        )
        object.__setattr__(self, "model_config", Path(self.model_config))


@dataclass(frozen=True)
class PreflightResult:
    status: str
    passed: bool
    observed_sha256: tuple[tuple[str, str], ...] = ()
    api_checks: tuple[str, ...] = ()
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["observed_sha256"] = dict(self.observed_sha256)
        result["api_checks"] = list(self.api_checks)
        return result

    def verification_token(self) -> "SourcePreflightToken":
        if self.status != "PASS_EXACT_SOURCE_AND_API" or self.passed is not True:
            raise ValueError("only a passing exact-source preflight can issue a token")
        return SourcePreflightToken(
            observed_sha256=self.observed_sha256,
            api_checks=self.api_checks,
            binding_sha256=_preflight_binding_digest(
                self.observed_sha256, self.api_checks
            ),
        )


@dataclass(frozen=True)
class SourcePreflightToken:
    """Immutable digest/API evidence consumed before registry mutation."""

    observed_sha256: tuple[tuple[str, str], ...]
    api_checks: tuple[str, ...]
    binding_sha256: str

    def __post_init__(self) -> None:
        observed = tuple(tuple(item) for item in self.observed_sha256)
        checks = tuple(self.api_checks)
        if any(
            len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in observed
        ):
            raise ValueError("preflight token observed digests are malformed")
        if any(type(name) is not str or not name for name in checks):
            raise ValueError("preflight token API checks are malformed")
        object.__setattr__(self, "observed_sha256", observed)
        object.__setattr__(self, "api_checks", checks)


def _preflight_binding_digest(
    observed_sha256: Sequence[tuple[str, str]], api_checks: Sequence[str]
) -> str:
    payload = json.dumps(
        {
            "observed_sha256": list(observed_sha256),
            "api_checks": list(api_checks),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


DEFAULT_SOURCE_EXPECTATIONS = (
    SourceExpectation(
        "registry",
        "vllm",
        "model_executor/models/registry.py",
        "27f6675a032ba82780c4b6038c05e6aad3988879b44f3582f6d5a273847da2f0",
    ),
    SourceExpectation(
        "layer",
        "vllm",
        "model_executor/layers/fused_moe/layer.py",
        "c888133b47507d6dc0a8e74671ef6e6b2c60c53e615e27098e354ad5b961d725",
    ),
    SourceExpectation(
        "qwen3_next",
        "vllm",
        "model_executor/models/qwen3_next.py",
        "b642c10eb68978ca0df25f92ea866add08b31e654d99e78eaac0195d8bc6c74b",
    ),
    SourceExpectation(
        "model_utils",
        "vllm",
        "model_executor/models/utils.py",
        "f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2",
    ),
    SourceExpectation(
        "moe_runner",
        "vllm",
        "model_executor/layers/fused_moe/runner/moe_runner.py",
        "3c000e17f726c9995a9a950db952415a4b2198d0778e3cedda96355421edfdae",
    ),
    SourceExpectation(
        "routed_experts",
        "vllm",
        "model_executor/layers/fused_moe/routed_experts.py",
        "aa8dd75adc7c2e69f1b1c999aaac9acd4db4a206afa45081e1ffd9c4c4fdb76c",
    ),
    SourceExpectation(
        "expert_map_manager",
        "vllm",
        "model_executor/layers/fused_moe/expert_map_manager.py",
        "308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b",
    ),
    SourceExpectation(
        "unquantized",
        "vllm",
        "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        "76f66e7e37680d7bea541159528d3ca302be73b88beb0f5bf4847deae786dab1",
    ),
    SourceExpectation(
        "weight_utils",
        "vllm",
        "model_executor/model_loader/weight_utils.py",
        "0f7dd5854cb850ee1aede33e11b22f37ea9c53f60d3477c5115efd20d69417d9",
    ),
    SourceExpectation(
        "ascend_patch",
        "vllm_ascend",
        "patch/platform/patch_fused_moe.py",
        "007875c5e3c188e33714cca007bd3e365b8ceb0d5fca8415866c673861267eaf",
    ),
    SourceExpectation(
        "ascend_moe",
        "vllm_ascend",
        "ops/fused_moe/fused_moe.py",
        "2d03cf6963f4e7edd529751383c779c407560011760c86478cf0598661d47f7c",
    ),
    SourceExpectation(
        "moe_comm_method",
        "vllm_ascend",
        "ops/fused_moe/moe_comm_method.py",
        "043f3c28518254347589f7af03793810262d0b29a9a82ca17dfd5fe5f9559f2e",
    ),
    SourceExpectation(
        "prepare_finalize",
        "vllm_ascend",
        "ops/fused_moe/prepare_finalize.py",
        "c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd",
    ),
    SourceExpectation(
        "quant_type",
        "vllm_ascend",
        "quantization/quant_type.py",
        "2faa4432100452f762aac28b17cb6167d8ed62955063abb9c4f56647fb0f8257",
    ),
    SourceExpectation(
        "token_dispatcher",
        "vllm_ascend",
        "ops/fused_moe/token_dispatcher.py",
        "36fa3658532b330f701c441bab279bed320fea067a2ddf75ee4cfa6a703e0d9b",
    ),
    SourceExpectation(
        "moe_runtime_args",
        "vllm_ascend",
        "ops/fused_moe/moe_runtime_args.py",
        "c1d52390f5619ce184207038d131abbe2d6745f527badc82363b0394f80371c2",
    ),
    SourceExpectation(
        "ascend_forward_context",
        "vllm_ascend",
        "ascend_forward_context.py",
        "af4a6e79ffcf6803c2a2a41dd598618398f090ea3a97253642c340ea26295c9f",
    ),
)

DEFAULT_MODEL_CONFIG_EXPECTATION = ModelConfigExpectation(
    sha256="889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
    architecture=TARGET_ARCHITECTURE,
    num_hidden_layers=48,
    num_experts=512,
    num_experts_per_tok=10,
    dtype="bfloat16",
    minimum_context_capacity=256,
)


def validate_exact_source_preflight_token(token: object) -> bool:
    """Require evidence from the complete frozen source/config combination."""

    if not isinstance(token, SourcePreflightToken):
        return False
    expected_observed = tuple(
        (item.name, item.sha256) for item in DEFAULT_SOURCE_EXPECTATIONS
    ) + (("model_config", DEFAULT_MODEL_CONFIG_EXPECTATION.sha256),)
    expected_checks = tuple(
        item.name for item in DEFAULT_SOURCE_EXPECTATIONS
    ) + ("model_config",)
    return (
        token.observed_sha256 == expected_observed
        and token.api_checks == expected_checks
        and token.binding_sha256
        == _preflight_binding_digest(token.observed_sha256, token.api_checks)
    )


class _ApiMismatch(ValueError):
    pass


def _read_bounded(path: Path, limit: int) -> bytes:
    stat = path.stat()
    if not path.is_file():
        raise OSError(f"not a regular file: {path}")
    if stat.st_size > limit:
        raise OSError(f"file exceeds bounded preflight limit: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise OSError(f"file exceeds bounded preflight limit: {path}")
    return data


def _function_args(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    arguments = function.args
    result = {
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    }
    if arguments.vararg is not None:
        result.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        result.add(arguments.kwarg.arg)
    return result


def _functions(tree: ast.AST, name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def _classes(tree: ast.AST, name: str) -> list[ast.ClassDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name]


def _has_call(function: ast.AST, name: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == name:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == name:
            return True
    return False


def _call_line(function: ast.AST, name: str) -> int | None:
    lines = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == name:
            lines.append(node.lineno)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == name:
            lines.append(node.lineno)
    return min(lines) if lines else None


def _require_function(
    tree: ast.AST,
    name: str,
    required_args: Iterable[str],
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    required = set(required_args)
    for function in reversed(_functions(tree, name)):
        if required <= _function_args(function):
            return function
    missing = ", ".join(sorted(required))
    raise _ApiMismatch(f"{name} signature does not include: {missing}")


def _validate_registry(tree: ast.AST) -> None:
    function = _require_function(tree, "register_model", ("self", "model_arch", "model_cls"))
    if not _has_call(function, "issubclass"):
        raise _ApiMismatch("registry no longer validates a model class with issubclass")


def _validate_layer(tree: ast.AST) -> None:
    function = _require_function(
        tree, "FusedMoE", ("routed_experts_cls", "routed_experts_args")
    )
    routed_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "routed_experts_cls"
    ]
    if not routed_calls or not any(
        any(keyword.arg is None for keyword in call.keywords) for call in routed_calls
    ):
        raise _ApiMismatch("FusedMoE no longer constructs routed_experts_cls with forwarded args")


def _validate_qwen(tree: ast.AST) -> None:
    classes = _classes(tree, "Qwen3NextSparseMoeBlock")
    if not classes:
        raise _ApiMismatch("Qwen3NextSparseMoeBlock is unavailable")
    constructors = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    if not constructors or not _has_call(constructors[0], "FusedMoE"):
        raise _ApiMismatch("Qwen3NextSparseMoeBlock no longer calls its FusedMoE global")
    consumers = [
        function
        for function in _functions(tree, "load_weights")
        if _has_call(function, "AutoWeightsLoader")
        and _has_call(function, "load_weights")
    ]
    if not consumers:
        raise _ApiMismatch(
            "Qwen load_weights no longer delegates to AutoWeightsLoader.load_weights"
        )


def _validate_model_utils(tree: ast.AST) -> None:
    classes = _classes(tree, "AutoWeightsLoader")
    if not classes:
        raise _ApiMismatch("AutoWeightsLoader class is unavailable")
    loader = classes[0]
    _require_function(loader, "__init__", ("self", "module"))
    load_param = _require_function(
        loader, "_load_param", ("self", "base_prefix", "param", "weights")
    )
    if not _has_call(load_param, "getattr") or not _has_call(
        load_param, "weight_loader"
    ):
        raise _ApiMismatch(
            "AutoWeightsLoader no longer dispatches parameter weight_loader"
        )
    load_module = _require_function(
        loader, "_load_module", ("self", "base_prefix", "module", "weights")
    )
    load_module_strings = {
        node.value
        for node in ast.walk(load_module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if (
        "load_weights" not in load_module_strings
        or not _has_call(load_module, "module_load_weights")
        or not _has_call(load_module, "_load_param")
    ):
        raise _ApiMismatch(
            "AutoWeightsLoader no longer delegates module and parameter consumers"
        )
    load_weights = _require_function(
        loader, "load_weights", ("self", "weights", "mapper")
    )
    if not _has_call(load_weights, "_load_module"):
        raise _ApiMismatch("AutoWeightsLoader.load_weights no longer walks the module")


def _validate_moe_runner(tree: ast.AST) -> None:
    classes = _classes(tree, "MoERunner")
    if not classes:
        raise _ApiMismatch("MoERunner class is unavailable")
    runner = classes[0]
    _require_function(
        runner,
        "__init__",
        (
            "self",
            "layer_name",
            "moe_config",
            "router",
            "routed_experts",
            "shared_experts",
            "routed_scaling_factor",
        ),
    )
    quant = _require_function(
        runner,
        "_apply_quant_method",
        ("self", "hidden_states", "router_logits", "shared_experts_input"),
    )
    if not _has_call(quant, "select_experts") or not _has_call(
        quant, "forward_modular"
    ):
        raise _ApiMismatch(
            "MoERunner modular path no longer selects once before routed execution"
        )
    implementation = _require_function(
        runner,
        "_forward_impl",
        ("self", "hidden_states", "router_logits", "shared_experts_input"),
    )
    if not _has_call(implementation, "_apply_quant_method"):
        raise _ApiMismatch("MoERunner _forward_impl no longer owns quant execution")
    forward = _require_function(
        runner, "forward", ("self", "hidden_states", "router_logits")
    )
    ordered_calls = (
        "_forward_entry",
        "_maybe_reduce_shared_expert_output",
        "_maybe_apply_routed_scale_to_output",
        "apply_routed_output_transform",
        "_maybe_reduce_final_output",
        "_maybe_add_zero_expert_output",
    )
    lines = tuple(_call_line(forward, name) for name in ordered_calls)
    if any(line is None for line in lines):
        missing = [
            name for name, line in zip(ordered_calls, lines) if line is None
        ]
        detail = ", ".join(missing)
        if "_maybe_apply_routed_scale_to_output" in missing:
            detail = "routed scale owner; " + detail
        raise _ApiMismatch("MoERunner.forward missing owner calls: " + detail)
    numeric_lines = tuple(line for line in lines if line is not None)
    if numeric_lines != tuple(sorted(numeric_lines)):
        raise _ApiMismatch(
            "MoERunner.forward scaling/shared/transform/reduction owner order changed"
        )


def _validate_routed_experts(tree: ast.AST) -> None:
    if not _classes(tree, "RoutedExperts"):
        raise _ApiMismatch("RoutedExperts class is unavailable")
    constructor = _require_function(
        tree,
        "__init__",
        (
            "self",
            "layer_name",
            "params_dtype",
            "moe_config",
            "quant_config",
            "expert_map_manager",
            "ckpt_gate_proj_name",
            "ckpt_down_proj_name",
            "ckpt_up_proj_name",
        ),
    )
    if not _has_call(constructor, "create_weights"):
        raise _ApiMismatch("RoutedExperts no longer owns quant create_weights")
    _require_function(
        tree,
        "weight_loader",
        (
            "self",
            "param",
            "loaded_weight",
            "weight_name",
            "shard_id",
            "expert_id",
            "return_success",
        ),
    )


def _validate_unquantized(tree: ast.AST) -> None:
    if not _classes(tree, "UnquantizedFusedMoEMethod"):
        raise _ApiMismatch("UnquantizedFusedMoEMethod class is unavailable")
    function = _require_function(
        tree,
        "create_weights",
        (
            "self",
            "layer",
            "num_experts",
            "hidden_size",
            "intermediate_size_per_partition",
            "params_dtype",
        ),
    )
    string_values = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if not {"w13_weight", "w2_weight"} <= string_values:
        raise _ApiMismatch("unquantized create_weights no longer registers w13_weight/w2_weight")


def _validate_weight_utils(tree: ast.AST) -> None:
    default = _require_function(
        tree, "default_weight_loader", ("param", "loaded_weight")
    )
    if not _has_call(default, "copy_"):
        raise _ApiMismatch("default weight loader no longer copies into the parameter")
    _require_function(
        tree,
        "maybe_remap_moe_expert_param_name",
        ("name", "params_dict"),
    )
    remap = _require_function(
        tree, "remap_moe_expert_weights", ("weights", "params_dict")
    )
    if not _has_call(remap, "maybe_remap_moe_expert_param_name"):
        raise _ApiMismatch("MoE checkpoint remap no longer consumes the name mapper")


def _validate_ascend_patch(tree: ast.AST) -> None:
    function = _require_function(
        tree, "_ascend_FusedMoE", ("args", "runner_cls", "runner_args", "kwargs")
    )
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_original_FusedMoE"
    ]
    if not calls or not any(
        any(keyword.arg is None for keyword in call.keywords) for call in calls
    ):
        raise _ApiMismatch("_ascend_FusedMoE no longer forwards arbitrary FusedMoE kwargs")


def _validate_ascend_moe(tree: ast.AST) -> None:
    classes = _classes(tree, "AscendUnquantizedFusedMoEMethod")
    if not classes:
        raise _ApiMismatch("AscendUnquantizedFusedMoEMethod is unavailable")
    base_names = {
        base.id for base in classes[0].bases if isinstance(base, ast.Name)
    }
    if "UnquantizedFusedMoEMethod" not in base_names:
        raise _ApiMismatch("Ascend unquantized method no longer inherits upstream create_weights")
    apply = _require_function(
        classes[0],
        "apply",
        (
            "self",
            "layer",
            "x",
            "top_k",
            "router_logits",
            "renormalize",
            "routed_scaling_factor",
            "expert_map",
            "log2phy",
        ),
    )
    if not _has_call(apply, "select_experts"):
        raise _ApiMismatch(
            "Ascend quant apply no longer exposes the reviewed internal routing call"
        )
    runners = _classes(tree, "AscendMoERunner")
    if not runners:
        raise _ApiMismatch("AscendMoERunner class is unavailable")
    runner = runners[0]
    constructor = _require_function(
        runner,
        "__init__",
        (
            "self",
            "layer_name",
            "moe_config",
            "router",
            "routed_experts",
            "routed_scaling_factor",
            "tid2eid",
            "n_shared_experts",
            "runtime_activation",
        ),
    )
    if not _has_call(constructor, "__init__"):
        raise _ApiMismatch("AscendMoERunner no longer invokes its base constructor")
    attributes = {
        node.attr
        for node in ast.walk(constructor)
        if isinstance(node, ast.Attribute)
    }
    if "num_local_experts" not in attributes:
        raise _ApiMismatch(
            "AscendMoERunner no longer consumes local-expert runtime metadata"
        )
    _require_function(
        runner,
        "no_shared_forward_impl",
        ("self", "hidden_states", "router_logits"),
    )
    _require_function(
        runner,
        "shared_forward_impl",
        ("self", "hidden_states", "router_logits"),
    )
    implementation = _require_function(
        runner,
        "_forward_impl",
        ("self", "hidden_states", "router_logits", "shared_experts_input"),
    )
    if not (
        _has_call(implementation, "no_shared_forward_impl")
        or _has_call(implementation, "shared_forward_impl")
    ):
        raise _ApiMismatch(
            "AscendMoERunner _forward_impl no longer delegates its reviewed branch"
        )


def _validate_expert_map(tree: ast.AST) -> None:
    if not _classes(tree, "ExpertMapManager"):
        raise _ApiMismatch("ExpertMapManager class is unavailable")
    _require_function(tree, "map_global_to_local", ("self", "global_id"))


def _validate_moe_comm_method(tree: ast.AST) -> None:
    methods = _classes(tree, "MoECommMethod")
    if not methods:
        raise _ApiMismatch("MoECommMethod class is unavailable")
    owner = methods[0]
    prepare = _require_function(
        owner,
        "prepare",
        ("self", "hidden_states", "router_logits", "quant_type"),
    )
    if not _has_call(prepare, "prepare"):
        raise _ApiMismatch("MoECommMethod.prepare envelope is unavailable")
    finalize = _require_function(
        owner,
        "finalize",
        ("self", "hidden_states", "reduce_results", "padded_hidden_states_shape"),
    )
    if not _has_call(finalize, "finalize"):
        raise _ApiMismatch("MoECommMethod.finalize envelope is unavailable")
    fused = _require_function(
        owner, "fused_experts", ("self", "fused_experts_input")
    )
    required_calls = (
        "build_token_dispatch_input",
        "token_dispatch",
        "build_mlp_compute_input",
        "token_combine",
        "FusedExpertsResult",
    )
    missing = [name for name in required_calls if not _has_call(fused, name)]
    if missing:
        raise _ApiMismatch(
            "MoECommMethod.fused_experts weighted aggregate missing: "
            + ", ".join(missing)
        )
    attributes = {
        node.attr for node in ast.walk(fused) if isinstance(node, ast.Attribute)
    }
    # ``topk_ids`` is read explicitly for optional log2phy remapping.  The
    # full input object (including ``topk_weights``) is passed to
    # build_token_dispatch_input and consumed by the reviewed dispatcher.
    if "topk_ids" not in attributes:
        raise _ApiMismatch(
            "MoECommMethod.fused_experts no longer consumes preselected Top-K"
        )
    if not _classes(tree, "AllGatherCommImpl") or not _classes(
        tree, "FusedMC2CommImpl"
    ):
        raise _ApiMismatch("reviewed MoE communication variants are unavailable")


def _validate_prepare_finalize(tree: ast.AST) -> None:
    classes = _classes(tree, "PrepareAndFinalizeWithAllGather")
    if not classes:
        raise _ApiMismatch("ALLGATHER prepare/finalize implementation is unavailable")
    owner = classes[0]
    _require_function(
        owner,
        "prepare",
        ("self", "hidden_states", "router_logits", "quant_type"),
    )
    _require_function(
        owner,
        "finalize",
        ("self", "hidden_states", "reduce_results", "padded_hidden_states_shape"),
    )
    if not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "QuantType"
        and node.attr == "NONE"
        for node in ast.walk(owner)
    ):
        raise _ApiMismatch("ALLGATHER prepare no longer defaults to QuantType.NONE")


def _validate_quant_type(tree: ast.AST) -> None:
    classes = _classes(tree, "QuantType")
    if not classes:
        raise _ApiMismatch("QuantType enum is unavailable")
    if not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(isinstance(target, ast.Name) and target.id == "NONE" for target in node.targets)
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == "NONE"
        )
        for node in classes[0].body
    ):
        raise _ApiMismatch("QuantType.NONE is unavailable")


def _validate_token_dispatcher(tree: ast.AST) -> None:
    classes = _classes(tree, "TokenDispatcherWithAllGather")
    if not classes:
        raise _ApiMismatch("ALLGATHER token dispatcher is unavailable")
    owner = classes[0]
    dispatch = _require_function(owner, "token_dispatch", ("self", "token_dispatch_input"))
    combine = _require_function(
        owner, "token_combine", ("self", "hidden_states", "combine_metadata")
    )
    attributes = {
        node.attr for node in ast.walk(dispatch) if isinstance(node, ast.Attribute)
    }
    if not {"topk_ids", "topk_weights", "expert_map"} <= attributes:
        raise _ApiMismatch("ALLGATHER dispatch no longer consumes frozen routing")
    if not _has_call(dispatch, "npu_moe_init_routing"):
        raise _ApiMismatch("ALLGATHER dispatch routing call is unavailable")
    if not _has_call(combine, "npu_moe_token_unpermute"):
        raise _ApiMismatch("ALLGATHER weighted token combine is unavailable")
    masks_weights_only = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "topk_weights"
            for target in node.targets
        )
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Mult)
        and any(
            isinstance(term, ast.Name) and term.id == "mask"
            for term in (node.value.left, node.value.right)
        )
        for node in ast.walk(dispatch)
    )
    if not masks_weights_only:
        raise _ApiMismatch(
            "ALLGATHER expert_map path no longer has reviewed weight-only mask semantics"
        )


def _validate_moe_runtime_args(tree: ast.AST) -> None:
    builder = _require_function(
        tree,
        "build_fused_experts_input",
        (
            "hidden_states",
            "topk_weights",
            "topk_ids",
            "w1",
            "w2",
            "quant_type",
            "dynamic_eplb",
            "expert_map",
            "mc2_mask",
            "log2phy",
            "pertoken_scale",
            "apply_router_weight_on_input",
        ),
    )
    for call_name in (
        "MoEFusedExpertsInput",
        "MoEWeights",
        "MoERoutingParams",
        "MoEQuantParams",
    ):
        if not _has_call(builder, call_name):
            raise _ApiMismatch(
                f"build_fused_experts_input no longer constructs {call_name}"
            )


def _validate_ascend_forward_context(tree: ast.AST) -> None:
    functions = _functions(tree, "set_ascend_forward_context")
    if not functions:
        raise _ApiMismatch("Ascend forward-context setter is unavailable")
    setter = functions[0]
    if not _has_call(setter, "select_moe_comm_method") or not _has_call(
        setter, "get_moe_comm_method"
    ):
        raise _ApiMismatch(
            "forward context no longer selects and binds a MoE comm object"
        )
    attributes = {
        node.attr for node in ast.walk(setter) if isinstance(node, ast.Attribute)
    }
    if not {"moe_comm_type", "moe_comm_method"} <= attributes:
        raise _ApiMismatch("forward context no longer stores comm type/object identity")
    selector = _require_function(
        tree, "select_moe_comm_method", ("num_tokens", "vllm_config")
    )
    if not any(
        isinstance(node, ast.Attribute) and node.attr == "ALLGATHER"
        for node in ast.walk(selector)
    ):
        raise _ApiMismatch("forward comm selection no longer exposes ALLGATHER")


_VALIDATORS = {
    "registry": _validate_registry,
    "layer": _validate_layer,
    "qwen3_next": _validate_qwen,
    "model_utils": _validate_model_utils,
    "moe_runner": _validate_moe_runner,
    "routed_experts": _validate_routed_experts,
    "expert_map_manager": _validate_expert_map,
    "unquantized": _validate_unquantized,
    "weight_utils": _validate_weight_utils,
    "ascend_patch": _validate_ascend_patch,
    "ascend_moe": _validate_ascend_moe,
    "moe_comm_method": _validate_moe_comm_method,
    "prepare_finalize": _validate_prepare_finalize,
    "quant_type": _validate_quant_type,
    "token_dispatcher": _validate_token_dispatcher,
    "moe_runtime_args": _validate_moe_runtime_args,
    "ascend_forward_context": _validate_ascend_forward_context,
}


def _validate_config(data: bytes, expectation: ModelConfigExpectation) -> None:
    try:
        config = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _ApiMismatch("model config is not valid UTF-8 JSON") from error
    if not isinstance(config, dict):
        raise _ApiMismatch("model config root must be an object")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise _ApiMismatch("model text_config must be an object")
    observed = (
        config.get("architectures"),
        text_config.get("num_hidden_layers"),
        text_config.get("num_experts"),
        text_config.get("num_experts_per_tok"),
        text_config.get("dtype"),
    )
    expected = (
        [expectation.architecture],
        expectation.num_hidden_layers,
        expectation.num_experts,
        expectation.num_experts_per_tok,
        expectation.dtype,
    )
    if observed != expected:
        raise _ApiMismatch(
            "model config identity mismatch: expected architecture/layers/experts/top-k/dtype "
            f"{expected!r}, observed {observed!r}"
        )
    context_capacity = text_config.get("max_position_embeddings")
    if (
        type(context_capacity) is not int
        or context_capacity < expectation.minimum_context_capacity
    ):
        raise _ApiMismatch(
            "model max_position_embeddings must cover at least the 256-token "
            "input-plus-generation contract"
        )


def run_preflight(
    paths: PreflightPaths,
    *,
    enabled: bool = False,
    source_expectations: Sequence[SourceExpectation] = DEFAULT_SOURCE_EXPECTATIONS,
    model_config_expectation: ModelConfigExpectation = DEFAULT_MODEL_CONFIG_EXPECTATION,
) -> PreflightResult:
    """Hash and AST-check the exact frozen source combination.

    Hash equality is required before any AST result can pass.  This deliberately
    rejects source-compatible but unreviewed versions.
    """

    if type(enabled) is not bool:
        return PreflightResult("INVALID_ENABLE_FLAG", False)
    if not enabled:
        return PreflightResult("DISABLED_DEFAULT_OFF", False)
    if not isinstance(paths, PreflightPaths):
        return PreflightResult("INVALID_PREFLIGHT_PATHS", False)
    expectations = tuple(source_expectations)
    if not expectations or any(
        not isinstance(item, SourceExpectation) for item in expectations
    ):
        return PreflightResult("INVALID_SOURCE_MANIFEST", False)
    names = tuple(item.name for item in expectations)
    if (
        len(set(names)) != len(names)
        or set(names) != set(_VALIDATORS)
    ):
        return PreflightResult("UNKNOWN_SOURCE_COMBINATION", False)
    if not isinstance(model_config_expectation, ModelConfigExpectation):
        return PreflightResult("INVALID_MODEL_CONFIG_EXPECTATION", False)

    roots: Mapping[str, Path] = {
        "vllm": paths.vllm_package_root,
        "vllm_ascend": paths.vllm_ascend_package_root,
    }
    data_by_name: dict[str, bytes] = {}
    observed: list[tuple[str, str]] = []
    source_mismatches: list[str] = []
    try:
        for expectation in expectations:
            data = _read_bounded(
                roots[expectation.package] / expectation.relative_path,
                MAX_SOURCE_BYTES,
            )
            data_by_name[expectation.name] = data
            digest = hashlib.sha256(data).hexdigest()
            observed.append((expectation.name, digest))
            if digest != expectation.sha256:
                source_mismatches.append(expectation.name)
        config_data = _read_bounded(paths.model_config, MAX_CONFIG_BYTES)
        config_digest = hashlib.sha256(config_data).hexdigest()
        observed.append(("model_config", config_digest))
    except (OSError, KeyError) as error:
        return PreflightResult(
            "SOURCE_READ_FAILED", False, tuple(observed), detail=str(error)
        )
    if source_mismatches:
        return PreflightResult(
            "SOURCE_IDENTITY_MISMATCH",
            False,
            tuple(observed),
            detail="sha256 mismatch: " + ", ".join(source_mismatches),
        )
    if config_digest != model_config_expectation.sha256:
        return PreflightResult(
            "MODEL_CONFIG_IDENTITY_MISMATCH",
            False,
            tuple(observed),
            detail="model config sha256 mismatch",
        )

    checked: list[str] = []
    try:
        for expectation in expectations:
            try:
                source = data_by_name[expectation.name].decode("utf-8")
                tree = ast.parse(source, filename=expectation.relative_path)
            except (UnicodeDecodeError, SyntaxError) as error:
                raise _ApiMismatch(
                    f"{expectation.name} is not parseable UTF-8 Python"
                ) from error
            _VALIDATORS[expectation.name](tree)
            checked.append(expectation.name)
    except _ApiMismatch as error:
        return PreflightResult(
            "API_SHAPE_MISMATCH",
            False,
            tuple(observed),
            tuple(checked),
            str(error),
        )
    try:
        _validate_config(config_data, model_config_expectation)
    except _ApiMismatch as error:
        return PreflightResult(
            "MODEL_CONFIG_MISMATCH",
            False,
            tuple(observed),
            tuple(checked),
            str(error),
        )
    checked.append("model_config")
    return PreflightResult(
        "PASS_EXACT_SOURCE_AND_API",
        True,
        tuple(observed),
        tuple(checked),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--vllm-package-root", type=Path, required=True)
    parser.add_argument("--vllm-ascend-package-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_preflight(
        PreflightPaths(
            arguments.vllm_package_root,
            arguments.vllm_ascend_package_root,
            arguments.model_config,
        ),
        enabled=arguments.enable,
    )
    payload = json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AUTO_WEIGHTS_LOADER_CONSUMER_STATUS",
    "DEFAULT_MODEL_CONFIG_EXPECTATION",
    "DEFAULT_SOURCE_EXPECTATIONS",
    "ModelConfigExpectation",
    "PreflightPaths",
    "PreflightResult",
    "SourceExpectation",
    "SourcePreflightToken",
    "main",
    "run_preflight",
    "validate_exact_source_preflight_token",
)
