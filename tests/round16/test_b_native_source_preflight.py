from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


def _module():
    try:
        return importlib.import_module("round16.b.native_source_preflight")
    except ModuleNotFoundError as error:
        raise AssertionError("round16.b.native_source_preflight must exist") from error


_SOURCES = {
    "registry": (
        "vllm",
        "model_executor/models/registry.py",
        """
class nn:
    class Module: pass
class ModelRegistry:
    def register_model(self, model_arch, model_cls):
        return isinstance(model_cls, type) and issubclass(model_cls, nn.Module)
""",
    ),
    "layer": (
        "vllm",
        "model_executor/layers/fused_moe/layer.py",
        """
def FusedMoE(*args, routed_experts_cls=None, routed_experts_args=None, **kwargs):
    return routed_experts_cls(*args, **(routed_experts_args or {}))
""",
    ),
    "qwen3_next": (
        "vllm",
        "model_executor/models/qwen3_next.py",
        """
class Qwen3NextSparseMoeBlock:
    def __init__(self):
        self.experts = FusedMoE(num_experts=512)
class Qwen3NextModel:
    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
""",
    ),
    "model_utils": (
        "vllm",
        "model_executor/models/utils.py",
        """
class AutoWeightsLoader:
    def __init__(self, module, skip_prefixes=None, skip_substrs=None,
                 ignore_unexpected_prefixes=None,
                 ignore_unexpected_suffixes=None):
        self.module = module
    def _load_param(self, base_prefix, param, weights):
        for weight_name, weight_data in weights:
            weight_loader = getattr(param, 'weight_loader', default_weight_loader)
            weight_loader(param, weight_data)
            yield weight_name
    def _load_module(self, base_prefix, module, weights):
        module_load_weights = getattr(module, 'load_weights', None)
        if callable(module_load_weights):
            yield from module_load_weights(weights)
        yield from self._load_param(base_prefix, module, weights)
    def load_weights(self, weights, mapper=None):
        return set(self._load_module('', self.module, weights))
""",
    ),
    "moe_runner": (
        "vllm",
        "model_executor/layers/fused_moe/runner/moe_runner.py",
        """
class MoERunner:
    def __init__(self, layer_name, moe_config, router, routed_experts,
                 enable_dbo=False, gate=None, shared_experts=None,
                 shared_expert_gate=None, routed_input_transform=None,
                 routed_output_transform=None, routed_scaling_factor=1.0):
        self.router = router
        self.routed_experts = routed_experts
    def _apply_quant_method(self, hidden_states, router_logits,
                            shared_experts_input, input_ids=None):
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states, router_logits=router_logits)
        return None, self.routed_experts.forward_modular(
            x=hidden_states, topk_weights=topk_weights, topk_ids=topk_ids)
    def _forward_impl(self, hidden_states, router_logits,
                      shared_experts_input, input_ids=None):
        return self._apply_quant_method(hidden_states, router_logits,
                                        shared_experts_input, input_ids)
    def forward(self, hidden_states, router_logits, input_ids=None):
        result = self._forward_entry(hidden_states, router_logits)
        shared_output, fused_output = result
        shared_output = self._maybe_reduce_shared_expert_output(shared_output)
        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output)
        fused_output = self.apply_routed_output_transform(fused_output)
        result = shared_output + fused_output
        result = self._maybe_reduce_final_output(result, None)
        return self._maybe_add_zero_expert_output(result)
""",
    ),
    "routed_experts": (
        "vllm",
        "model_executor/layers/fused_moe/routed_experts.py",
        """
class RoutedExperts:
    def __init__(self, layer_name, params_dtype, moe_config, quant_config,
                 expert_map_manager, ckpt_gate_proj_name='gate_proj',
                 ckpt_down_proj_name='down_proj', ckpt_up_proj_name='up_proj'):
        self.quant_method.create_weights(layer=self, num_experts=moe_config.num_local_experts,
                                         global_num_experts=moe_config.num_experts,
                                         weight_loader=self.weight_loader)
    def weight_loader(self, param, loaded_weight, weight_name, shard_id,
                      expert_id, return_success=False):
        return self.expert_map_manager.map_global_to_local(expert_id)
""",
    ),
    "expert_map_manager": (
        "vllm",
        "model_executor/layers/fused_moe/expert_map_manager.py",
        """
class ExpertMapManager:
    def map_global_to_local(self, global_id):
        return global_id
""",
    ),
    "unquantized": (
        "vllm",
        "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        """
class UnquantizedFusedMoEMethod:
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        layer.register_parameter('w13_weight', num_experts)
        layer.register_parameter('w2_weight', num_experts)
""",
    ),
    "weight_utils": (
        "vllm",
        "model_executor/model_loader/weight_utils.py",
        """
def default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)
def maybe_remap_moe_expert_param_name(name, params_dict):
    return name
def remap_moe_expert_weights(weights, params_dict):
    for name, weight in weights:
        yield maybe_remap_moe_expert_param_name(name, params_dict), weight
""",
    ),
    "ascend_patch": (
        "vllm_ascend",
        "patch/platform/patch_fused_moe.py",
        """
def _ascend_FusedMoE(*args, runner_cls=None, runner_args=None, **kwargs):
    return _original_FusedMoE(*args, runner_cls=runner_cls,
                              runner_args=runner_args, **kwargs)
""",
    ),
    "ascend_moe": (
        "vllm_ascend",
        "ops/fused_moe/fused_moe.py",
        """
class UnquantizedFusedMoEMethod: pass
class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def apply(self, layer, x, top_k, router_logits, renormalize,
              routed_scaling_factor, expert_map, log2phy):
        select_experts(x, router_logits)
        return x
class AscendMoERunner(MoERunner):
    def __init__(self, layer_name, moe_config, router, routed_experts,
                 enable_dbo=False, gate=None, shared_experts=None,
                 shared_expert_gate=None, routed_input_transform=None,
                 routed_output_transform=None, routed_scaling_factor=1.0,
                 tid2eid=None, n_shared_experts=0, runtime_activation=None):
        super().__init__(layer_name, moe_config, router, routed_experts)
        self.moe_load = [0] * moe_config.num_local_experts
    def no_shared_forward_impl(self, hidden_states, router_logits):
        return self._quant_method.apply(self, hidden_states, 10, router_logits,
                                        True, 1.0, None, None)
    def shared_forward_impl(self, hidden_states, router_logits):
        return self._shared_experts(hidden_states), self.no_shared_forward_impl(
            hidden_states, router_logits)
    def _forward_impl(self, hidden_states, router_logits,
                      shared_experts_input, input_ids=None):
        return self.no_shared_forward_impl(hidden_states, router_logits)
def select_experts(): pass
""",
    ),
    "moe_comm_method": (
        "vllm_ascend",
        "ops/fused_moe/moe_comm_method.py",
        """
class MoECommMethod:
    def prepare(self, hidden_states, router_logits, enable_shared_expert_dp=False,
                replace_allreduce=False, quant_type=None):
        return self.prepare_finalize.prepare(hidden_states, router_logits)
    def finalize(self, hidden_states, reduce_results,
                 padded_hidden_states_shape=None):
        return self.prepare_finalize.finalize(hidden_states, reduce_results)
    def fused_experts(self, fused_experts_input):
        ids = fused_experts_input.topk_ids
        weights = fused_experts_input.topk_weights
        dispatched = build_token_dispatch_input(fused_experts_input, ids)
        dispatch_output = self.token_dispatcher.token_dispatch(dispatched)
        mlp_input = build_mlp_compute_input(fused_experts_input, dispatch_output)
        mlp_output = self._apply_mlp(mlp_input)
        routed_out = self.token_dispatcher.token_combine(mlp_output)
        return FusedExpertsResult(routed_out=routed_out)
class AllGatherCommImpl(MoECommMethod): pass
class FusedMC2CommImpl(MoECommMethod):
    def fused_experts(self, fused_experts_input):
        return super().fused_experts(fused_experts_input)
""",
    ),
    "prepare_finalize": (
        "vllm_ascend",
        "ops/fused_moe/prepare_finalize.py",
        """
class QuantType:
    NONE = 0
class PrepareAndFinalizeWithAllGather:
    def prepare(self, hidden_states, router_logits, quant_type=QuantType.NONE):
        return MoEPrepareOutput(hidden_states=hidden_states,
                                router_logits=router_logits)
    def finalize(self, hidden_states, reduce_results,
                 padded_hidden_states_shape=None):
        return hidden_states
""",
    ),
    "quant_type": (
        "vllm_ascend",
        "quantization/quant_type.py",
        """
from enum import Enum
class QuantType(Enum):
    NONE = 0
    W8A8 = 1
""",
    ),
    "token_dispatcher": (
        "vllm_ascend",
        "ops/fused_moe/token_dispatcher.py",
        """
class TokenDispatcherWithAllGather:
    def token_dispatch(self, token_dispatch_input):
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        if expert_map is not None:
            mask = expert_map[topk_ids] != -1
            topk_weights = topk_weights * mask
        return DeviceOperator.npu_moe_init_routing(
            token_dispatch_input.hidden_states, topk_ids)
    def token_combine(self, hidden_states, combine_metadata, bias=None):
        return DeviceOperator.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            probs=combine_metadata.topk_weights)
class TokenDispatcherWithMC2: pass
class TokenDispatcherWithAll2AllV: pass
""",
    ),
    "moe_runtime_args": (
        "vllm_ascend",
        "ops/fused_moe/moe_runtime_args.py",
        """
class MoEFusedExpertsInput: pass
class MoEWeights: pass
class MoERoutingParams: pass
class MoEQuantParams: pass
def build_fused_experts_input(*, hidden_states, topk_weights, topk_ids,
                              w1, w2, quant_type, dynamic_eplb,
                              expert_map=None, log2phy=None,
                              mc2_mask=None, pertoken_scale=None,
                              apply_router_weight_on_input=False):
    return MoEFusedExpertsInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        weights=MoEWeights(w1=w1, w2=w2),
        routing=MoERoutingParams(expert_map=expert_map, log2phy=log2phy,
                                 mc2_mask=mc2_mask,
                                 pertoken_scale=pertoken_scale),
        quant=MoEQuantParams(quant_type=quant_type),
        dynamic_eplb=dynamic_eplb)
""",
    ),
    "ascend_forward_context": (
        "vllm_ascend",
        "ascend_forward_context.py",
        """
class MoECommType:
    ALLGATHER = 'ALLGATHER'
def set_ascend_forward_context(num_tokens, vllm_config):
    forward_context = get_forward_context()
    moe_comm_type = select_moe_comm_method(num_tokens, vllm_config)
    forward_context.moe_comm_type = moe_comm_type
    forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)
    return forward_context
def select_moe_comm_method(num_tokens, vllm_config):
    return MoECommType.ALLGATHER
""",
    ),
}


class NativeSourcePreflightTests(unittest.TestCase):
    def _fixture(self, root: Path):
        module = _module()
        package_roots = {
            "vllm": root / "vllm",
            "vllm_ascend": root / "vllm_ascend",
        }
        expectations = []
        for name, (package, relative, text) in _SOURCES.items():
            path = package_roots[package] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            expectations.append(
                module.SourceExpectation(
                    name=name,
                    package=package,
                    relative_path=relative,
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        config = {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 48,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "dtype": "bfloat16",
                "max_position_embeddings": 262144,
            },
        }
        config_bytes = (json.dumps(config, sort_keys=True) + "\n").encode()
        config_path = root / "config.json"
        config_path.write_bytes(config_bytes)
        config_expectation = module.ModelConfigExpectation(
            sha256=hashlib.sha256(config_bytes).hexdigest(),
            architecture="Qwen4ExpForConditionalGeneration",
            num_hidden_layers=48,
            num_experts=512,
            num_experts_per_tok=10,
            dtype="bfloat16",
            minimum_context_capacity=256,
        )
        paths = module.PreflightPaths(
            vllm_package_root=package_roots["vllm"],
            vllm_ascend_package_root=package_roots["vllm_ascend"],
            model_config=config_path,
        )
        return paths, tuple(expectations), config_expectation

    def test_default_off_reads_nothing_and_has_no_import_side_effect(self) -> None:
        module = _module()
        missing = Path("/definitely/not/present")
        result = module.run_preflight(
            module.PreflightPaths(missing, missing, missing),
            enabled=False,
        )
        self.assertEqual(result.status, "DISABLED_DEFAULT_OFF")
        self.assertFalse(result.passed)
        self.assertEqual(result.observed_sha256, ())

    def test_exact_hash_api_and_config_fixture_passes(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            paths, expectations, config = self._fixture(Path(directory))
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations,
                model_config_expectation=config,
            )
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "PASS_EXACT_SOURCE_AND_API")
        self.assertEqual(set(result.api_checks), set(_SOURCES) | {"model_config"})
        token = result.verification_token()
        self.assertEqual(token.observed_sha256, result.observed_sha256)
        self.assertEqual(token.api_checks, result.api_checks)

    def test_exact_runner_and_checkpoint_utility_source_identities_are_frozen(self):
        module = _module()
        manifest = {item.name: item for item in module.DEFAULT_SOURCE_EXPECTATIONS}
        self.assertEqual(
            manifest["moe_runner"].sha256,
            "3c000e17f726c9995a9a950db952415a4b2198d0778e3cedda96355421edfdae",
        )
        self.assertEqual(
            manifest["weight_utils"].sha256,
            "0f7dd5854cb850ee1aede33e11b22f37ea9c53f60d3477c5115efd20d69417d9",
        )
        self.assertEqual(
            module.AUTO_WEIGHTS_LOADER_CONSUMER_STATUS,
            "AUTO_WEIGHTS_LOADER_CONSUMER_SOURCE_BOUND",
        )
        self.assertEqual(
            manifest["model_utils"].sha256,
            "f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2",
        )

    def test_hash_mismatch_fails_closed_before_ast_acceptance(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            paths, expectations, config = self._fixture(Path(directory))
            bad = list(expectations)
            bad[0] = module.SourceExpectation(
                name=bad[0].name,
                package=bad[0].package,
                relative_path=bad[0].relative_path,
                sha256="0" * 64,
            )
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=tuple(bad),
                model_config_expectation=config,
            )
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "SOURCE_IDENTITY_MISMATCH")
        self.assertEqual(result.api_checks, ())

        with tempfile.TemporaryDirectory() as directory:
            paths, expectations, config = self._fixture(Path(directory))
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations[:-1],
                model_config_expectation=config,
            )
        self.assertEqual(result.status, "UNKNOWN_SOURCE_COMBINATION")
        self.assertEqual(result.observed_sha256, ())

    def test_unknown_routed_experts_signature_and_config_fail_closed(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, expectations, config = self._fixture(root)
            routed = paths.vllm_package_root / _SOURCES["routed_experts"][1]
            malformed = "class RoutedExperts:\n    def weight_loader(self): pass\n"
            routed.write_text(malformed, encoding="utf-8")
            expectations = tuple(
                module.SourceExpectation(
                    item.name,
                    item.package,
                    item.relative_path,
                    hashlib.sha256(malformed.encode()).hexdigest()
                    if item.name == "routed_experts"
                    else item.sha256,
                )
                for item in expectations
            )
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations,
                model_config_expectation=config,
            )
            self.assertEqual(result.status, "API_SHAPE_MISMATCH")

            paths, expectations, config = self._fixture(root / "second")
            bad_config_data = json.loads(paths.model_config.read_text())
            bad_config_data["text_config"]["num_experts"] = 511
            bad_bytes = (json.dumps(bad_config_data, sort_keys=True) + "\n").encode()
            paths.model_config.write_bytes(bad_bytes)
            bad_config = module.ModelConfigExpectation(
                hashlib.sha256(bad_bytes).hexdigest(),
                config.architecture,
                config.num_hidden_layers,
                config.num_experts,
                config.num_experts_per_tok,
                config.dtype,
                config.minimum_context_capacity,
            )
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations,
                model_config_expectation=bad_config,
            )
            self.assertEqual(result.status, "MODEL_CONFIG_MISMATCH")

    def test_dtype_context_and_qwen_weight_consumer_fail_closed(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in (
                ("dtype", "float16"),
                ("max_position_embeddings", 128),
            ):
                with self.subTest(field=field):
                    paths, expectations, config = self._fixture(root / field)
                    data = json.loads(paths.model_config.read_text())
                    data["text_config"][field] = value
                    payload = (json.dumps(data, sort_keys=True) + "\n").encode()
                    paths.model_config.write_bytes(payload)
                    changed = module.ModelConfigExpectation(
                        hashlib.sha256(payload).hexdigest(),
                        config.architecture,
                        config.num_hidden_layers,
                        config.num_experts,
                        config.num_experts_per_tok,
                        config.dtype,
                        config.minimum_context_capacity,
                    )
                    result = module.run_preflight(
                        paths,
                        enabled=True,
                        source_expectations=expectations,
                        model_config_expectation=changed,
                    )
                    self.assertEqual(result.status, "MODEL_CONFIG_MISMATCH")

            paths, expectations, config = self._fixture(root / "consumer")
            qwen = paths.vllm_package_root / _SOURCES["qwen3_next"][1]
            malformed = _SOURCES["qwen3_next"][2].replace(
                "loader = AutoWeightsLoader(self)", "loader = object()"
            )
            qwen.write_text(malformed, encoding="utf-8")
            expectations = tuple(
                module.SourceExpectation(
                    item.name,
                    item.package,
                    item.relative_path,
                    hashlib.sha256(malformed.encode()).hexdigest()
                    if item.name == "qwen3_next"
                    else item.sha256,
                )
                for item in expectations
            )
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations,
                model_config_expectation=config,
            )
            self.assertEqual(result.status, "API_SHAPE_MISMATCH")
            self.assertIn("AutoWeightsLoader", result.detail)

    def test_runner_owner_order_and_ascend_internal_route_shape_fail_closed(self):
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, expectations, config = self._fixture(root)
            runner = paths.vllm_package_root / _SOURCES["moe_runner"][1]
            malformed = _SOURCES["moe_runner"][2].replace(
                "self._maybe_apply_routed_scale_to_output", "self.unknown_scale_owner"
            )
            runner.write_text(malformed, encoding="utf-8")
            expectations = tuple(
                module.SourceExpectation(
                    item.name,
                    item.package,
                    item.relative_path,
                    hashlib.sha256(malformed.encode()).hexdigest()
                    if item.name == "moe_runner"
                    else item.sha256,
                )
                for item in expectations
            )
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations,
                model_config_expectation=config,
            )
            self.assertEqual(result.status, "API_SHAPE_MISMATCH")
            self.assertIn("routed scale", result.detail)

    def test_loader_consumer_and_default_copy_ast_fail_closed(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (
                    "model_utils",
                    "module_load_weights(weights)",
                    "set()",
                    "delegates module",
                ),
                (
                    "weight_utils",
                    "param.data.copy_(loaded_weight)",
                    "param.data = loaded_weight",
                    "copies into",
                ),
            )
            for name, old, new, message in cases:
                with self.subTest(name=name):
                    paths, expectations, config = self._fixture(root / name)
                    package, relative, original = _SOURCES[name]
                    malformed = original.replace(old, new)
                    path = (
                        paths.vllm_package_root
                        if package == "vllm"
                        else paths.vllm_ascend_package_root
                    ) / relative
                    path.write_text(malformed, encoding="utf-8")
                    expectations = tuple(
                        module.SourceExpectation(
                            item.name,
                            item.package,
                            item.relative_path,
                            hashlib.sha256(malformed.encode()).hexdigest()
                            if item.name == name
                            else item.sha256,
                        )
                        for item in expectations
                    )
                    result = module.run_preflight(
                        paths,
                        enabled=True,
                        source_expectations=expectations,
                        model_config_expectation=config,
                    )
                    self.assertEqual(result.status, "API_SHAPE_MISMATCH")
                    self.assertIn(message, result.detail)

    def test_moe_comm_weighted_aggregate_envelope_fails_closed(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            paths, expectations, config = self._fixture(Path(directory))
            package, relative, original = _SOURCES["moe_comm_method"]
            malformed = original.replace(
                "routed_out = self.token_dispatcher.token_combine(mlp_output)",
                "routed_out = mlp_output",
            )
            path = paths.vllm_ascend_package_root / relative
            path.write_text(malformed, encoding="utf-8")
            expectations = tuple(
                module.SourceExpectation(
                    item.name,
                    item.package,
                    item.relative_path,
                    hashlib.sha256(malformed.encode()).hexdigest()
                    if item.name == "moe_comm_method"
                    else item.sha256,
                )
                for item in expectations
            )
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations,
                model_config_expectation=config,
            )
            self.assertEqual(result.status, "API_SHAPE_MISMATCH")
            self.assertIn("token_combine", result.detail)

    def test_allgather_dispatcher_mask_only_semantics_fail_closed(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            paths, expectations, config = self._fixture(Path(directory))
            _package, relative, original = _SOURCES["token_dispatcher"]
            malformed = original.replace(
                "topk_weights = topk_weights * mask",
                "topk_ids = expert_map[topk_ids]",
            )
            path = paths.vllm_ascend_package_root / relative
            path.write_text(malformed, encoding="utf-8")
            expectations = tuple(
                module.SourceExpectation(
                    item.name,
                    item.package,
                    item.relative_path,
                    hashlib.sha256(malformed.encode()).hexdigest()
                    if item.name == "token_dispatcher"
                    else item.sha256,
                )
                for item in expectations
            )
            result = module.run_preflight(
                paths,
                enabled=True,
                source_expectations=expectations,
                model_config_expectation=config,
            )
            self.assertEqual(result.status, "API_SHAPE_MISMATCH")
            self.assertIn("mask", result.detail)


if __name__ == "__main__":
    unittest.main()
