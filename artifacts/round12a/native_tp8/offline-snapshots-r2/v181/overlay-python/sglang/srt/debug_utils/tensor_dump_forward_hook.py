"""
This file provides a function `register_forward_hook_for_model` that registers a forward hook on every operator of the model.
After registration, during model inference, all tensors generated throughout the forward pass will be recorded.

Usage:
Specify the output directory for dumping tensors using the argument `--debug-tensor-dump-output-folder`.
A separate directory will be created for each GPU rank, named in the format `f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{pid}"`.
Each complete forward pass of the model generates a `.pt` file named `f"Pass{pass_num}.pt"`, which can be loaded using `torch.load`.
The file contains a series of key-value pairs, where the keys correspond to operator names in the model
(similar to those in model.safetensors.index.json), and the values are the outputs produced by the respective operators.
"""

import logging
import hashlib
import json
from functools import wraps
from contextvars import ContextVar
import os
from pathlib import Path
from typing import List, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)
_active_capture = ContextVar("round62_active_capture", default=None)


def record_round62_stage(source, layer, stage, tensor):
    capture = _active_capture.get()
    if capture is None or layer not in (7, 8):
        return
    # The pending producer can still be writing this pinned tensor. Optional
    # diagnostics must not read it or introduce an extra CPUInfer drain.
    if source == "cpu" and stage == "incremental_input":
        return
    if capture._r63_compact and capture._forward_pass_id not in capture._r63_full_sequences:
        keep = {"router_logits", "router_topk_ids", "router_weights", "topk_ids",
                "routing_weights", "immediate_ids", "deferred_ids",
                "immediate_ids_after_sync", "deferred_ids_after_sync",
                "routing_weights_after_sync", "cpu_output_after_sync"}
        if stage not in keep:
            return
    name = f"r62.{source}.layer{layer}.{stage}"
    if name in capture._current_tensors:
        raise RuntimeError(f"Round6.2 duplicate stage: {name}")
    capture._current_tensors[name] = tensor.detach().to("cpu").contiguous().clone()


def record_round62_event(event, payload):
    capture = _active_capture.get()
    if capture is not None:
        capture._capture_events.append({"event": event, **payload})


class TensorDumper:
    def __init__(
        self,
        dump_dir: str,
        dump_layers: Optional[List[int]],
        tp_size: int,
        tp_rank: int,
        pp_rank: int,
    ):
        self._dump_layers = dump_layers
        self._forward_pass_id = 0
        self._pid = os.getpid()
        self._current_tensors = {}
        # Debug-only: bind input metadata and logits to the same outer forward.
        self._complete_forward = os.getenv("KT_R62_COMPLETE_FORWARD_DUMP") == "1"
        self._r63_compact = os.getenv("KT_R63_COMPACT_CAPTURE") == "1"
        self._r63_full_sequences = {
            int(i) for i in os.getenv("KT_R63_FULL_DIAGNOSTIC_SEQUENCES", "0,1,32,63,65,130,195").split(",") if i
        }
        if self._r63_compact and not self._complete_forward:
            raise RuntimeError("R63_COMPACT_REQUIRES_COMPLETE_CAPTURE")
        self._capture_active = False
        self._last_committed_forward = -1
        self._capture_events = []
        run_path = os.getenv("KT_R62_RUN_METADATA")
        self._run_metadata = json.loads(Path(run_path).read_text()) if run_path else {}
        self._base_dir = Path(dump_dir)
        rank = tp_size * pp_rank + tp_rank
        self._process_dir = (
            self._base_dir / f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{self._pid}"
        )
        self._process_dir.mkdir(parents=True, exist_ok=True)
        index_path = os.getenv("KT_R63_CAPTURE_INDEX")
        self._r63_index = open(index_path, "x", buffering=1) if index_path else None

    def get_dump_dir(self):
        return str(self._process_dir)

    def add_tensor(self, name, tensor_item):
        if isinstance(tensor_item, (tuple, list)):
            tensors = [t.cpu() for t in tensor_item if t is not None]
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            else:
                self._current_tensors[name] = tensors
        elif isinstance(tensor_item, torch.Tensor):
            self._current_tensors[name] = tensor_item.cpu()
        elif isinstance(tensor_item, LogitsProcessorOutput):
            self._current_tensors[name] = tensor_item.next_token_logits.cpu()
        elif isinstance(tensor_item, ForwardBatch):
            self._current_tensors[name + ".forward_batch_info.input_ids"] = (
                tensor_item.input_ids.cpu()
            )
            self._current_tensors[name + ".forward_batch_info.seq_lens"] = (
                tensor_item.seq_lens.cpu()
            )
            self._current_tensors[name + ".forward_batch_info.positions"] = (
                tensor_item.positions.cpu()
            )
        elif isinstance(tensor_item, PPProxyTensors):
            for tensor_name in tensor_item.tensors.keys():
                self._current_tensors[name + ".pp_proxy_tensors." + tensor_name] = (
                    tensor_item.tensors[tensor_name].cpu()
                )
        else:
            logger.warning(f"Unsupported type: {type(tensor_item)}: {tensor_item}")

    def dump_current_tensors(self):
        if len(self._current_tensors) == 0:
            return
        tensor_file_for_pass = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info(
            f"Dump {self._forward_pass_id:05d}th pass to {tensor_file_for_pass}"
        )
        if self._complete_forward:
            temporary = tensor_file_for_pass.with_suffix(".pt.incomplete")
            if tensor_file_for_pass.exists() or temporary.exists():
                raise RuntimeError("Round6.2 refuses duplicate capture file")
            torch.save(self._current_tensors, str(temporary))
            os.replace(temporary, tensor_file_for_pass)
        else:
            torch.save(self._current_tensors, str(tensor_file_for_pass))
        if self._r63_index is not None:
            tx = self._current_tensors.get("capture.transaction")
            if tx is None:
                raise RuntimeError("R63_INDEX_REQUIRES_TRANSACTION_CAPTURE")
            self._r63_index.write(json.dumps({
                "file": str(tensor_file_for_pass), "pid": self._pid,
                "sequence": self._forward_pass_id, "batch": tx["batch"],
                "tensor_byte_sha256": tx["tensor_byte_sha256"],
            }, sort_keys=True) + "\n")
        self._current_tensors = {}
        self._forward_pass_id += 1

    def _add_hook_recursive(
        self, model, prefix, top_level_module_name, layers_module_name
    ):
        model_top_level_module_matched = False
        layers_prefix = top_level_module_name + "." + layers_module_name
        for name, module in model._modules.items():
            top_level_model = False
            if len(prefix) == 0:
                cur_name = name
                if cur_name == top_level_module_name:
                    model_top_level_module_matched = True
                    top_level_model = True
            else:
                cur_name = prefix + "." + name
            if (
                self._dump_layers is not None
                and name.isdigit()
                and prefix == layers_prefix
            ):
                # If we only need n layers, skip the reset layers.
                # Most models' layout is like model.layers.0.
                cur_layer = int(name)
                if cur_layer not in self._dump_layers:
                    continue
            if module is not None:
                _, sub_count = self._add_hook_recursive(
                    module, cur_name, top_level_module_name, layers_module_name
                )
                if sub_count == 0 or top_level_model:
                    # Avoid duplicated output hooks, e.g. self_attn may contain:
                    # self_attn.qkv_proj, self_attn.attn & self_attn.o_proj.
                    # Therefore, we do not need to add output hooks for self_attn,
                    # since the output of self_attn should be the same to self_attn.o_proj.
                    module.register_forward_hook(
                        self._dump_hook(cur_name, top_level_model)
                    )
        return model_top_level_module_matched, len(model._modules.items())

    def _dump_hook(self, tensor_name, do_dump):
        def inner_dump_hook(module, input, output):
            if self._r63_compact and self._forward_pass_id not in self._r63_full_sequences:
                return
            if do_dump and not self._complete_forward:
                # This is the top-level model, so we will record the input for it.
                for item in input:
                    if isinstance(item, ForwardBatch):
                        self.add_tensor(tensor_name, item)
                self.dump_current_tensors()
            if output is not None:
                self.add_tensor(tensor_name, output)

        return inner_dump_hook

    def begin_complete_forward(self, module, args, kwargs):
        if self._current_tensors or self._capture_active:
            raise RuntimeError("Round6.2 capture contains unflushed forward data")
        if self._forward_pass_id <= self._last_committed_forward:
            raise RuntimeError("Round6.2 duplicate forward sequence")
        found = 0
        batch_metadata = {}
        for item in (*args, *kwargs.values()):
            if isinstance(item, ForwardBatch):
                self.add_tensor("capture", item)
                found += 1
                batch_metadata = {"request_ids": list(getattr(item, "rids", [])),
                                  "forward_mode": str(getattr(item, "forward_mode", "UNKNOWN"))}
        if found != 1:
            raise RuntimeError("Round6.2 requires exactly one ForwardBatch")
        # Preserve metadata even if serving subsequently reuses CPU buffers.
        self._current_tensors = {
            name: value.clone() for name, value in self._current_tensors.items()
        }
        self._capture_active = True
        self._capture_events = []
        self._batch_metadata = batch_metadata
        self._capture_token = _active_capture.set(self)

    def end_complete_forward(self, module, args, kwargs, output):
        if not self._capture_active:
            raise RuntimeError("Round6.2 output without matching entry")
        if not isinstance(output, LogitsProcessorOutput):
            raise RuntimeError("Round6.2 requires an outer LogitsProcessorOutput")
        self.add_tensor("capture.next_token_logits", output)
        hashes = {}
        for name in ("capture.forward_batch_info.input_ids",
                     "capture.forward_batch_info.positions",
                     "capture.next_token_logits"):
            tensor = self._current_tensors[name].contiguous()
            hashes[name] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        self._current_tensors["capture.transaction"] = {
            "capture_version": "round6.2-complete-forward-v2",
            "pid": self._pid,
            "forward_sequence_id": self._forward_pass_id,
            "tensor_byte_sha256": hashes,
            "batch": self._batch_metadata,
            "provenance": self._run_metadata,
            "logits_shape": list(output.next_token_logits.shape),
            "logits_dtype": str(output.next_token_logits.dtype),
            "deferred_events": self._capture_events,
            "r63_observer": "compact" if self._r63_compact else "full",
            "r63_full_diagnostic": not self._r63_compact or self._forward_pass_id in self._r63_full_sequences,
        }
        completed = self._forward_pass_id
        self.dump_current_tensors()
        self._last_committed_forward = completed
        self._capture_active = False
        _active_capture.reset(self._capture_token)

    def reference_logits_only(self, module, args, kwargs, output):
        # Common minimal output observer for OFF/ON parity: no router, buffer,
        # input-D2H or layer hooks. Measurement of raw logits still needs this
        # final output copy in the reference process; disclose that limitation.
        if not isinstance(output, LogitsProcessorOutput):
            raise RuntimeError("Round6.2 reference requires outer logits")
        batches = [item for item in (*args, *kwargs.values()) if isinstance(item, ForwardBatch)]
        if len(batches) != 1 or self._current_tensors:
            raise RuntimeError("Round6.2 invalid reference forward")
        self.add_tensor("capture.next_token_logits", output)
        self._current_tensors["capture.transaction"] = {
            "reference_logits_only": True, "pid": self._pid,
            "forward_sequence_id": self._forward_pass_id,
            "batch": {"request_ids": list(batches[0].rids)},
        }
        self.dump_current_tensors()


def _bind_round62_forward(model, dumper, reference=False):
    # ModelRunner calls model.forward directly, bypassing Module.__call__ hooks.
    # Wrap only the debug-enabled instance, not the production model class.
    original = model.forward

    @wraps(original)
    def captured_forward(*args, **kwargs):
        if not reference:
            dumper.begin_complete_forward(model, args, kwargs)
        output = original(*args, **kwargs)
        if reference:
            dumper.reference_logits_only(model, args, kwargs, output)
        else:
            dumper.end_complete_forward(model, args, kwargs, output)
        return output

    model.forward = captured_forward


def register_forward_hook_for_model(
    model,
    dump_dir: str,
    dump_layers: Optional[List[int]],
    tp_size: int,
    tp_rank: int,
    pp_rank: int,
):
    tensor_dumper = TensorDumper(dump_dir, dump_layers, tp_size, tp_rank, pp_rank)
    if os.getenv("KT_R62_REFERENCE_LOGITS_ONLY") == "1":
        if tensor_dumper._complete_forward:
            raise RuntimeError("Round6.2 reference and full capture are mutually exclusive")
        _bind_round62_forward(model, tensor_dumper, reference=True)
        return tensor_dumper
    # Most models have the layerout like:
    # XxxxForCausalLM
    #     (model): XxxxModel
    #         (layers): ModuleList
    # If the model is not constructed with this layout,
    # environment variable can be used to specify the module names.
    top_level_module_name = os.getenv("TENSOR_DUMP_TOP_LEVEL_MODULE_NAME", "model")
    layers_module_name = os.getenv("TENSOR_DUMP_LAYERS_MODULE_NAME", "layers")
    model_top_level_module_matched, _ = tensor_dumper._add_hook_recursive(
        model, "", top_level_module_name, layers_module_name
    )
    assert (
        model_top_level_module_matched
    ), f"model should have a module named {top_level_module_name}"
    if tensor_dumper._complete_forward:
        _bind_round62_forward(model, tensor_dumper)
    return tensor_dumper
