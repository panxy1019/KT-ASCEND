# Round16-B P0/B1 frozen vLLM-Ascend path map

Status: `OFFLINE_SOURCE_BOUND_DEFAULT_OFF_REAL_MOE_DEVICE_PENDING`

## Runtime and evidence identity

The frozen donor image is
`quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`, image ID
`sha256:7e3ab7e56a54259d3ecec91073346f9ca8b2a4c1f58e15826e3ed6ff67036f00`
and repository digest
`sha256:38e4a59458c82d2972dc08ea3aa209784cc9fb925c417a5559d422f2bbebf830`.
The saved baseline identifies vLLM `0.26.0` at commit
`568afb3a13806beb53bb2e6bd518269357b237c0` and vLLM-Ascend at commit
`550ab66e7feb89f6e2ba49394f6ffb153189ac2b`.

The source archive was read in place at
`/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round15/upstream-image-source/round15-offline-vllm-source-20260928.tar.gz`.
Its saved companion SHA-256 is
`005d731a31cc11854878efbd2c2a68d1840c72afba83b048422678942773eaef`.
Only the named files below were streamed with `tar -xOf`; the archive was not
unpacked or copied.

| Source under `vllm_ascend/` | SHA-256 | Resolved fact |
|---|---|---|
| `models/__init__.py` | `820c9fe4f3a107699f02e5d6e95fbd8312a056dcbdea376a324eb0d31879b9bd` | `ModelRegistry.register_model` maps `Qwen4ExpForConditionalGeneration` to the Ascend class |
| `models/qwen4_exp/__init__.py` | `878eb7ee4b5aec2e766ace3a08ff9d80a023aad2cb32e953ae44b0827b84fa92` | exported Ascend Qwen4Exp classes |
| `models/qwen4_exp/model.py` | `b06a3123073234efb39958c502b53bee23384d1a7eb61f40d12b3d8b55481133` | Ascend wrapper keeps the vendored model and loader authoritative |
| `models/qwen4_exp/nvidia/model.py` | `a38101a1c4a3cb078b0169d3c4b7b6c4b39ba41db8283a29cf8fcc5eb7ac9689` | conditional constructor and `AutoWeightsLoader` path |
| `platform.py` | `c711042261d5283be5aa644c3a5a00fcb34f70c0156f4b866700f38a927a1276` | native platform selects `vllm_ascend.worker.worker.NPUWorker` |
| `worker/worker.py` | `eb0bb11de6dffecbed135979655cf032cccef3f3ea4fe3b93254933d47e0f8c0` | worker constructor, device init and runner-owned model load |
| `worker/model_runner_v1.py` | `de9f108bb1365c7eb5967c0cd098a005510fe627f7a27e0732d120a0f33e2608` | `NPUModelRunner(vllm_config, device)` calls native `get_model` and owns long-lived state |
| `ops/fused_moe/fused_moe.py` | `2d03cf6963f4e7edd529751383c779c407560011760c86478cf0598661d47f7c` | unquantized weight post-processing and routed MoE apply entry |
| `patch/platform/patch_fused_moe.py` | `007875c5e3c188e33714cca007bd3e365b8ceb0d5fca8415866c673861267eaf` | proves `RoutedExperts` allocates before `AscendMoERunner`; global patch rejected |
| `patch/worker/patch_process_weights_after_loading.py` | `85c1fbf7e7ebaa52901e2173728be5dfacdcf796cebeb92f6d0ad9dd930c1d09` | global loader replacement rejected |

The following files are saved from official vLLM commit
`568afb3a13806beb53bb2e6bd518269357b237c0` under
`/home/admin/Documents/ChatGPT/KT/kt-round16-b-vllm/artifacts/round16-b/upstream-vllm-0.26`:

| Official vLLM source | SHA-256 | Resolved fact |
|---|---|---|
| `registry.py` | `27f6675a032ba82780c4b6038c05e6aad3988879b44f3582f6d5a273847da2f0` | `ModelRegistry.register_model` accepts `type[nn.Module]` as well as an import string |
| `layer.py` | `c888133b47507d6dc0a8e74671ef6e6b2c60c53e615e27098e354ad5b961d725` | `FusedMoE` accepts `routed_experts_cls` and `routed_experts_args`, then constructs the owner before weight creation |
| `qwen3_next.py` | `b642c10eb68978ca0df25f92ea866add08b31e654d99e78eaac0195d8bc6c74b` | every `Qwen3NextSparseMoeBlock` calls its module-global `FusedMoE` |
| `model_utils.py` | `f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2` | exact `AutoWeightsLoader`: child custom `load_weights` and parameter `weight_loader` delegation |
| `moe_runner.py` | `3c000e17f726c9995a9a950db952415a4b2198d0778e3cedda96355421edfdae` | route-once modular path and post-forward routed scale/shared/output-transform ownership |
| `routed_experts.py` | `aa8dd75adc7c2e69f1b1c999aaac9acd4db4a206afa45081e1ffd9c4c4fdb76c` | `RoutedExperts` owns expert mapping, loader, quant method and `create_weights` |
| `expert_map_manager.py` | `308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b` | logical, local and physical expert mapping owner |
| `unquantized_fused_moe_method.py` | `76f66e7e37680d7bea541159528d3ca302be73b88beb0f5bf4847deae786dab1` | unquantized `w13_weight`/`w2_weight` allocation shape and post-load path |
| `weight_utils.py` | `0f7dd5854cb850ee1aede33e11b22f37ea9c53f60d3477c5115efd20d69417d9` | `default_weight_loader` copy and MoE expert-name remapping |

The saved raw model-config evidence is
`/home/admin/Documents/ChatGPT/KT/kt-round16-integration/artifacts/round16-b/QWEN_MODEL_CONFIG_REMOTE_RAW.txt`,
SHA-256
`ed9343b42a0e9d7c1bc377ba953d470da079858a8e3abbf3b306deaad4a4cdf9`.
It records config SHA-256
`889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`,
architecture `Qwen4ExpForConditionalGeneration`, 48 hidden/MoE layers, 512
logical experts, Top-10 routing, `dtype=bfloat16`, and
`max_position_embeddings=262144`. The Conditional-versus-Causal selection is
resolved. Live TP1 still must prove template-inclusive input <=128, output <=128,
and `max_model_len` plus state capacity >=256.

## Resolved call chain

```text
vllm.ModelRegistry.register_model
  Qwen4ExpForConditionalGeneration
  -> vllm_ascend.models.qwen4_exp:AscendQwen4ExpForConditionalGeneration

NPUWorker.load_model
  -> NPUModelRunner.load_model
  -> vllm.model_executor.model_loader.get_model(vllm_config=...)
  -> AscendQwen4ExpForConditionalGeneration.__init__(*, vllm_config, prefix="model")
  -> Qwen4ExpForConditionalGeneration.load_weights
  -> AutoWeightsLoader.load_weights

AscendUnquantizedFusedMoEMethod.apply
  -> select_experts
  -> moe_comm_method.fused_experts
```

`NPUWorker` and `NPUModelRunner` remain the sole owners of the native model and
KV state. The Round16 adapter holds the identical model object and delegates to
it. It does not create another runner, model, KV cache, or platform.

## Allocation boundary

The saved donor source explicitly says `RoutedExperts` allocates its parameters
before `AscendMoERunner` is constructed. Therefore Q32 must be installed before
the native model constructor reaches the FusedMoE factory. The official source
now resolves that seam: `FusedMoE` accepts a custom `routed_experts_cls` and
`routed_experts_args`, while `Qwen3NextSparseMoeBlock` looks up its module-global
`FusedMoE` during construction.

`ModelScopedSelectivePreallocationInstaller` takes a construction-wide lock,
verifies that `qwen3_next.FusedMoE` is the expected donor `_ascend_FusedMoE`,
temporarily installs a wrapper only in that Qwen module, and restores the exact
original binding on success or failure. The wrapper injects a routed owner
class plus immutable Q32 arguments. Acknowledgement checks the expected 48
layer count and exact plan/mapping for every constructed layer. This proves the
constructor seam only.

The default AscendMoERunner still receives the factory's original
ExpertMapManager and `moe_config`, whose local capacity was computed before the
custom RoutedExperts instance. A physical-Q32 owner must therefore be paired
with a custom Hybrid runner in the same FusedMoE call. The current routed-only
installer is not registration-eligible and returns
`RUNTIME_RUNNER_BINDING_UNRESOLVED`; it does not mutate the shared config to
pretend that runner metadata is Q32.

The CPU fixture for a fully paired binding registers a dynamic subclass of the
injected native Ascend model class. Its native base constructor runs on the
same `self`, so the design does not create a duplicate native model or KV
owner. Official `registry.py` proves that class value shape is accepted. The
current routed-only installer never reaches this registry mutation.

The donor's process-wide FusedMoE and weight-loader patches remain
`REJECTED_FOR_MODEL_SCOPED_PLUGIN`.

## Provenance distinction

Round12a's successful Native TP8 result is SGLang evidence at commit
`4ccff141dbe992794f9da6c3aa23535b4f72000d`, sourced from
`python/sglang/srt/models/qwen4_exp.py`. It is not vLLM-Ascend evidence.
The successful vLLM-Ascend 0.26 TP8 evidence comes from Round15's saved
`INPUT_QWEN38_A3_DEPLOYMENT_REPORT.md`; no container or device run was repeated.

## Minimum unresolved information before REAL_MOE

1. Run the prepared 17-source/config read-only preflight inside the real image
   and atomically pass its observed-digest/API token into registration.
2. Prove the actual Qwen `self.experts` object receives the paired runner and
   invokes routing once. Bind a native Q32 operation that accepts that frozen
   Top-K without calling Ascend quant `apply`, which reroutes internally.
3. Avoid default `AscendMoERunner.__init__` local512 placement/EPLB/moe-load
   state. Do not mutate shared `moe_config`; use upstream `MoERunner` or an
   equivalent reviewed owner for post-forward semantics.
4. Verify S15 arithmetic ownership on device: KT route-weighted/unscaled partial
   plus native partial, routed scale once, shared/output transform/reduction
   once, and caller-owned residual.
5. Bind the selected 48-layer S15 `q32-physical.json` path/hash and check all 512
   manager mappings plus physical32 BF16 w13/w2 owners in the real loader.
6. Before REQUEST_ASYNC, prove max-running-one submit -> native -> sync, cancel
   drain and slot reuse on the real stream, or add a native work identity.
7. Run TP1 eager with template-inclusive input <=128, output <=128, and live
   `max_model_len`/state capacity >=256. Then perform one bounded REAL_MOE
   numerical comparison.
