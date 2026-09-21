"""Real-weight GDN probe for physical NPU 15.

Scope is three original GDN layers with synthetic hidden states. This is a
component probe, not model integration or a full-model output qualification.
"""

import argparse
import ctypes
import hashlib
import json
import math
import pathlib
import sys
import faulthandler

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)

import torch
import torch.nn.functional as F
from safetensors import safe_open


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--identity-receipt", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

base = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(base))
sys.path.insert(0, str(base / "tools"))
from probe_identity import verify_probe_identity  # noqa: E402
from runtime.gdn import GDNConfig, eager_gdn, zero_state  # noqa: E402


revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
identity = verify_probe_identity(
    model=args.model,
    receipt=args.identity_receipt,
    expected_model=pathlib.Path("/models") / revision,
    expected_receipt_sha256=(
        "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
    ),
    revision=revision,
    manifest_sha256=(
        "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
    ),
)
print('GDN_STAGE identity_verified', flush=True)
index = json.loads((args.model / "model.safetensors.index.json").read_text())[
    "weight_map"
]
text_config = json.loads((args.model / "config.json").read_text())["text_config"]
config = GDNConfig(
    num_key_heads=text_config["linear_num_key_heads"],
    num_value_heads=text_config["linear_num_value_heads"],
    key_head_dim=text_config["linear_key_head_dim"],
    value_head_dim=text_config["linear_value_head_dim"],
    conv_kernel_size=text_config["linear_conv_kernel_dim"],
    eps=text_config["rms_norm_eps"],
)
if (
    config.num_key_heads,
    config.num_value_heads,
    config.key_head_dim,
    config.value_head_dim,
    config.conv_kernel_size,
    text_config["output_gate_type"],
    text_config["mamba_ssm_dtype"],
) != (16, 48, 128, 128, 4, "sigmoid", "float32"):
    raise ValueError("frozen GDN configuration mismatch")

layers = (0, 24, 46)
suffixes = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
    "out_proj.weight",
)


def checkpoint_key(layer, suffix):
    return f"model.language_model.layers.{layer}.linear_attn.{suffix}"


def load_layer(layer):
    weights = {}
    for suffix in suffixes:
        key = checkpoint_key(layer, suffix)
        if key not in index:
            raise ValueError(f"missing original GDN tensor: {key}")
        with safe_open(str(args.model / index[key]), framework="pt", device="cpu") as f:
            weights[suffix] = f.get_tensor(key)
    expected_shapes = {
        "in_proj_qkv.weight": (config.conv_dim, text_config["hidden_size"]),
        "in_proj_z.weight": (config.value_dim, text_config["hidden_size"]),
        "in_proj_a.weight": (config.num_value_heads, text_config["hidden_size"]),
        "in_proj_b.weight": (config.num_value_heads, text_config["hidden_size"]),
        "conv1d.weight": (config.conv_dim, 1, config.conv_kernel_size),
        "A_log": (config.num_value_heads,),
        "dt_bias": (config.num_value_heads,),
        "norm.weight": (config.value_head_dim,),
        "out_proj.weight": (text_config["hidden_size"], config.value_dim),
    }
    for name, shape in expected_shapes.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(
                f"original GDN tensor shape mismatch for {name}: "
                f"{tuple(weights[name].shape)} != {shape}"
            )
    return weights


def relative_l2(actual, expected):
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    )


def cpu_fp64_canonical(hidden, weights):
    """Independent literal equations; does not call runtime.gdn."""
    tokens = hidden.shape[0]
    x = hidden.double()
    qkv = F.linear(x, weights["in_proj_qkv.weight"].double())
    z = F.linear(x, weights["in_proj_z.weight"].double()).reshape(
        tokens, config.num_value_heads, config.value_head_dim
    )
    a = F.linear(x, weights["in_proj_a.weight"].double())
    b = F.linear(x, weights["in_proj_b.weight"].double())

    conv_weight = weights["conv1d.weight"].double()[:, 0, :]
    conv_state = torch.zeros(
        config.conv_dim, config.conv_kernel_size - 1, dtype=torch.float64
    )
    convolved = []
    conv_states = []
    for token in range(tokens):
        window = torch.cat((conv_state, qkv[token, :, None]), dim=-1)
        conv_value = (window * conv_weight).sum(-1)
        convolved.append(conv_value * conv_value.sigmoid())
        conv_state = window[:, 1:].clone()
        conv_states.append(conv_state)
    mixed = torch.stack(convolved)
    q_flat, k_flat, v_flat = torch.split(
        mixed, (config.query_dim, config.key_dim, config.value_dim), dim=-1
    )
    q = q_flat.reshape(tokens, config.num_key_heads, config.key_head_dim)
    k = k_flat.reshape(tokens, config.num_key_heads, config.key_head_dim)
    v = v_flat.reshape(tokens, config.num_value_heads, config.value_head_dim)

    recurrent = torch.zeros(
        config.num_value_heads,
        config.key_head_dim,
        config.value_head_dim,
        dtype=torch.float64,
    )
    ratio = config.num_value_heads // config.num_key_heads
    A_log = weights["A_log"].double()
    dt_bias = weights["dt_bias"].double()
    recurrent_outputs = []
    recurrent_states = []
    for token in range(tokens):
        query = q[token] * torch.rsqrt(
            q[token].square().sum(-1, keepdim=True) + config.eps
        )
        key = k[token] * torch.rsqrt(
            k[token].square().sum(-1, keepdim=True) + config.eps
        )
        query = query.repeat_interleave(ratio, dim=0)
        key = key.repeat_interleave(ratio, dim=0)
        log_decay = -A_log.exp() * F.softplus(a[token] + dt_bias)
        recurrent = recurrent * log_decay.exp()[:, None, None]
        error = v[token] - torch.einsum("hk,hkv->hv", key, recurrent)
        recurrent = recurrent + torch.einsum(
            "hk,hv->hkv", key, b[token].sigmoid()[:, None] * error
        )
        recurrent_outputs.append(
            torch.einsum("hk,hkv->hv", query, recurrent)
            / math.sqrt(config.key_head_dim)
        )
        recurrent_states.append(recurrent.clone())
    core = torch.stack(recurrent_outputs)
    normalized = core * torch.rsqrt(
        core.square().mean(-1, keepdim=True) + config.eps
    )
    gated = normalized * weights["norm.weight"].double() * z.sigmoid()
    output = F.linear(
        gated.flatten(1), weights["out_proj.weight"].double()
    )
    return output, conv_states, recurrent_states


def device_forward(hidden, weights, state=None):
    tokens = hidden.shape[0]
    qkv = F.linear(hidden, weights["in_proj_qkv.weight"])
    z = F.linear(hidden, weights["in_proj_z.weight"]).reshape(
        tokens, config.num_value_heads, config.value_head_dim
    )
    a = F.linear(hidden, weights["in_proj_a.weight"])
    b = F.linear(hidden, weights["in_proj_b.weight"])
    core, next_state = eager_gdn(
        qkv,
        a,
        b,
        z,
        weights["conv1d.weight"],
        weights["A_log"],
        weights["dt_bias"],
        weights["norm.weight"],
        config,
        initial_state=state,
    )
    return F.linear(core.flatten(1), weights["out_proj.weight"]), next_state


import torch_npu  # noqa: E402

print('GDN_STAGE torch_npu_imported', flush=True)
torch.set_num_threads(2)
print('GDN_STAGE threads_set', flush=True)
torch.npu.set_device(0)
print('GDN_STAGE device_selected', flush=True)
torch.npu.matmul.allow_hf32 = False
torch.npu.conv.allow_hf32 = False
acl = ctypes.CDLL(
    "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so"
)
physical = ctypes.c_int32(-1)
acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
if acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0:
    raise RuntimeError("failed to resolve physical NPU identity")
if physical.value != 15 or torch.npu.device_count() != 1:
    raise RuntimeError("probe is not isolated to physical NPU 15")
if list(pathlib.Path("/dev").glob("davinci[0-9]*")) != [
    pathlib.Path("/dev/davinci15")
]:
    raise RuntimeError("unexpected physical NPU device exposure")

report = {
    "status": "RUNNING",
    "scope": "GDN_REAL_WEIGHTS_SYNTHETIC_HIDDEN_COMPONENT_ONLY",
    "source_identity": identity,
    "layers": list(layers),
    "gates": {"float32": 1e-4, "bfloat16": 0.03},
    "rows": [],
}
peak_allocated = 0
try:
    with torch.no_grad():
        for layer in layers:
            print(f'GDN_STAGE load_layer {layer}', flush=True)
            cpu_weights = load_layer(layer)
            for dtype in (torch.float32, torch.bfloat16):
                torch.npu.reset_peak_memory_stats()
                device_weights = {}
                for name,tensor in cpu_weights.items():
                    print(f'GDN_STAGE transfer {layer} {dtype} {name}', flush=True)
                    # Bounded private CPU staging decouples mmap-backed source
                    # ownership from accelerator transfer and dtype conversion.
                    staged=tensor.to(dtype=torch.float32 if name == "A_log" else dtype).clone()
                    assert torch.equal(staged,tensor.to(staged.dtype))
                    device_weights[name] = staged.to(device="npu:0")
                    del staged
                if device_weights["A_log"].dtype != torch.float32:
                    raise RuntimeError("A_log was not retained in FP32")
                torch.manual_seed(
                    9400 + layer * 31 + (1 if dtype == torch.bfloat16 else 0)
                )
                hidden_cpu_all = torch.randn(
                    17, text_config["hidden_size"]
                ).to(dtype)
                expected_all, expected_conv_states, expected_recurrent_states = (
                    cpu_fp64_canonical(hidden_cpu_all, cpu_weights)
                )
                print(f'GDN_STAGE canonical_ready {layer} {dtype}', flush=True)
                hidden_all = hidden_cpu_all.to("npu:0")
                for length in (1, 3, 17):
                    expected = expected_all[:length]
                    expected_conv = expected_conv_states[length - 1]
                    expected_recurrent = expected_recurrent_states[length - 1]
                    hidden = hidden_all[:length]

                    hashes = []
                    full_output = None
                    full_state = None
                    for _ in range(3):
                        full_output, full_state = device_forward(
                            hidden, device_weights
                        )
                        torch.npu.synchronize()
                        output_cpu = full_output.cpu()
                        hashes.append(
                            hashlib.sha256(
                                output_cpu.contiguous()
                                .view(torch.uint8)
                                .numpy()
                                .tobytes()
                            ).hexdigest()
                        )

                    pieces = []
                    decode_state = None
                    for token in range(length):
                        piece, decode_state = device_forward(
                            hidden[token : token + 1],
                            device_weights,
                            decode_state,
                        )
                        pieces.append(piece)
                    decode_output = torch.cat(pieces)

                    reset_state = zero_state(
                        config, input_dtype=dtype, device="npu:0"
                    )
                    reset_output, reset_final = device_forward(
                        hidden, device_weights, reset_state
                    )
                    torch.npu.synchronize()

                    output_error = relative_l2(output_cpu, expected)
                    canonical_conv_error = relative_l2(
                        full_state.conv_state.cpu(), expected_conv
                    )
                    canonical_recurrent_error = relative_l2(
                        full_state.recurrent_state.cpu(), expected_recurrent
                    )
                    decode_error = relative_l2(decode_output.cpu(), output_cpu)
                    decode_conv_error = relative_l2(
                        decode_state.conv_state.cpu(), full_state.conv_state.cpu()
                    )
                    decode_recurrent_error = relative_l2(
                        decode_state.recurrent_state.cpu(),
                        full_state.recurrent_state.cpu(),
                    )
                    reset_error = relative_l2(reset_output.cpu(), output_cpu)
                    reset_conv_error = relative_l2(
                        reset_final.conv_state.cpu(), full_state.conv_state.cpu()
                    )
                    reset_recurrent_error = relative_l2(
                        reset_final.recurrent_state.cpu(),
                        full_state.recurrent_state.cpu(),
                    )
                    gate = 1e-4 if dtype == torch.float32 else 0.03
                    errors = {
                        "relative_l2": output_error,
                        "canonical_conv_state_relative_l2": canonical_conv_error,
                        "canonical_recurrent_state_relative_l2": canonical_recurrent_error,
                        "decode_relative_l2": decode_error,
                        "decode_conv_state_relative_l2": decode_conv_error,
                        "decode_recurrent_state_relative_l2": decode_recurrent_error,
                        "reset_relative_l2": reset_error,
                        "reset_conv_state_relative_l2": reset_conv_error,
                        "reset_recurrent_state_relative_l2": reset_recurrent_error,
                    }
                    row = {
                        "layer": layer,
                        "dtype": str(dtype),
                        "tokens": length,
                        **errors,
                        "repeat_hashes": hashes,
                    }
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    if (not all(math.isfinite(value) for value in errors.values())
                            or len(set(hashes)) != 1 or max(errors.values()) > gate):
                        raise AssertionError(row)
                    if full_state.recurrent_state.dtype != torch.float32:
                        raise AssertionError("GDN recurrent state is not FP32")

                    del (
                        expected,
                        expected_conv,
                        expected_recurrent,
                        hidden,
                        full_output,
                        full_state,
                        output_cpu,
                        pieces,
                        decode_output,
                        decode_state,
                        reset_state,
                        reset_output,
                        reset_final,
                        piece,
                    )
                del (
                    hidden_cpu_all,
                    expected_all,
                    expected_conv_states,
                    expected_recurrent_states,
                    hidden_all,
                )
                peak_allocated = max(
                    peak_allocated, int(torch.npu.max_memory_allocated())
                )
                del device_weights
                torch.npu.synchronize()
                torch.npu.empty_cache()
            del cpu_weights

    torch.npu.synchronize()
    torch.npu.empty_cache()
    report.update(
        status="PASS_GDN_COMPONENT_ONLY",
        peak_allocated=peak_allocated,
        after_allocated=int(torch.npu.memory_allocated()),
        after_reserved=int(torch.npu.memory_reserved()),
    )
    if report["after_allocated"] != 0 or report["after_reserved"] != 0:
        raise AssertionError("NPU allocations remain after GDN probe cleanup")
except BaseException as exc:
    report["status"] = "FAIL"
    report["error"] = repr(exc)
    raise
finally:
    args.output.write_text(json.dumps(report, indent=2))

print(
    json.dumps(
        {
            "status": report["status"],
            "rows": len(report["rows"]),
            "peak_allocated": report["peak_allocated"],
        }
    )
)
