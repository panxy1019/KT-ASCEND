"""Construct all48 decoder metadata on meta; no checkpoint weights or inference."""
import argparse
import faulthandler
import json
import struct
from pathlib import Path
import torch
import torch_npu

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
torch.set_num_threads(2)
torch.npu.set_device(0)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--audit-loader", action="store_true")
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.model_loader.loader import _initialize_model
from sglang.srt.configs.load_config import LoadConfig

report = {"status": "RUNNING", "scope": "META_STRUCTURE_ONLY_NO_WEIGHT_LOAD"}
try:
    args = ServerArgs(model_path=a.model, device="npu", tp_size=1,
        enable_multimodal=False, dtype="bfloat16",
        disable_cuda_graph=True, disable_radix_cache=True, max_running_requests=1,
        page_size=4, chunked_prefill_size=32, attention_backend="ascend")
    set_global_server_args_for_scheduler(args)
    config = ModelConfig(a.model, trust_remote_code=False, enable_multimodal=False,
                         dtype="bfloat16")
    print("MODEL_CONFIG_PASS", flush=True)
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29873", backend="gloo", timeout=60)
    initialize_model_parallel(backend="gloo")
    initialize_dp_attention(args, config)
    initialize_moe_config(args)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = _initialize_model(config, LoadConfig())
    assert len(model.model.layers) == 48
    parameters = dict(model.named_parameters())
    decay = [v for k, v in parameters.items() if k.endswith(".A_log")]
    assert len(decay) == 36 and all(v.dtype == torch.float32 for v in decay)
    assert sum(hasattr(layer, "indexer") for layer in model.model.layers) == 12
    assert [i for i, layer in enumerate(model.model.layers) if layer.ple is not None] == [1]
    assert model.model.hyper_connection_mixer is not None
    actual_bytes = sum(v.numel()*v.element_size() for v in parameters.values() if not v.is_meta)
    assert actual_bytes < 128*1024**2, actual_bytes
    report.update(status="PASS_META_STRUCTURE_ONLY", parameter_tensors=len(parameters),
                  non_meta_parameter_bytes=actual_bytes, decay_fp32_layers=len(decay),
                  peak_npu_allocated=torch.npu.max_memory_allocated())
    if a.audit_loader:
        from safetensors import safe_open
        # The loader validates PLE table row counts from the actual hash buffer
        # values, so these tiny buffers must have storage even in a meta audit.
        for decoder in model.model.layers:
            if decoder.ple is not None:
                for name in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
                    value = getattr(decoder.ple, name)
                    setattr(decoder.ple, name, torch.empty(value.shape, dtype=value.dtype, device="cpu"))
        model_dir = Path(a.model)
        weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        headers = {}
        for filename in set(weight_map.values()):
            with (model_dir / filename).open("rb") as f:
                headers[filename] = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        dtype_map = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}
        def metadata_weights():
            for name, filename in weight_map.items():
                entry = headers[filename][name]
                count = 1
                for dim in entry["shape"]:
                    count *= dim
                # Real small parameters cover NPU-resident norm weights and PLE
                # hash buffers; large weights remain metadata with no storage.
                if count <= 65536:
                    with safe_open(str(model_dir / filename), framework="pt", device="cpu") as f:
                        value = f.get_tensor(name).clone()
                else:
                    value = torch.empty(entry["shape"], dtype=dtype_map[entry["dtype"]], device="meta")
                yield name, value
        loaded = model.load_weights(metadata_weights())
        ple = model.model.layers[1].ple
        assert ple._table is not None, "real model loader did not bind original PLE table"
        sample = ple._table.gather_rows(torch.tensor([[0, 320001445]], dtype=torch.long))
        assert sample.shape == (1, 2, 160) and sample.dtype == torch.bfloat16
        assert torch.isfinite(sample).all()
        report["original_ple_table_bound"] = True
        report.update(status="PASS_META_STRUCTURE_AND_LOADER_MAP_ONLY",
                      loaded_destinations=len(loaded), checkpoint_tensors=len(weight_map),
                      ple_shards=len(model.weight_load_audit.ple_shards))
        def invalid_weights(kind):
            for name, value in metadata_weights():
                if kind == "missing_q" and name == "model.language_model.layers.3.self_attn.q_proj.weight":
                    continue
                if kind == "shifted_ple_boundary" and ".ple.ple_embedding.ngram_embedding.shard_" in name:
                    delta = 1 if name.endswith("shard_0.weight") else -1 if name.endswith("shard_1.weight") else 0
                    value = torch.empty((value.shape[0]+delta, value.shape[1]), dtype=value.dtype, device="meta")
                yield name, value
        failures = {}
        for kind, message in (("missing_q", "packed parameter"), ("shifted_ple_boundary", "row boundary mismatch")):
            try:
                model.load_weights(invalid_weights(kind))
            except RuntimeError as exc:
                assert message in str(exc), (kind, str(exc))
                failures[kind] = str(exc)
            else:
                raise AssertionError(f"invalid loader input was accepted: {kind}")
        report["invalid_inputs_rejected"] = failures
except BaseException as exc:
    report.update(status="FAIL", error=repr(exc))
    raise
finally:
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
print(report, flush=True)
