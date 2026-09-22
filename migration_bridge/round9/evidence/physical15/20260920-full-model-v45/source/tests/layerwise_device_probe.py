"""Guarded real-NPU gate for the private Round9 layerwise BF16 transport.

The fixture executes original layer-0 experts with synthetic activations.  It
does not qualify scheduler integration, prefill wiring, or a complete model.
"""
import argparse
import ctypes
import hashlib
import json
import pathlib
import sys
import faulthandler
from types import SimpleNamespace


DEVICE_STATUS = "DEVICE_PENDING_LAYERWISE_BF16_TRANSPORT"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
RECEIPT_SHA256 = "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
MANIFEST_SHA256 = "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
HIDDEN = 2560
INTERMEDIATE = 640
EXPERTS = 512
LAYERS = 48
CAPACITY = 32
TOPK = 10
MAX_GROUP_ROWS = 32


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_bound_manifest(path, expected_sha256):
    path = pathlib.Path(path)
    if digest(path) != expected_sha256:
        raise ValueError(f"manifest identity mismatch: {path.name}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"empty identity manifest: {path.name}")
    if any(type(name) is not str or type(value) is not str or len(value) != 64
           for name, value in data.items()):
        raise ValueError(f"invalid identity manifest: {path.name}")
    return data


def verify_tree(root, manifest):
    root = pathlib.Path(root).resolve(strict=True)
    actual_names = {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_names != set(manifest):
        raise ValueError("tree file set differs from identity manifest")
    seen = set()
    for name, expected in manifest.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or name in seen:
            raise ValueError("invalid or duplicate tree entry")
        seen.add(name)
        if digest(path) != expected:
            raise ValueError(f"tree identity changed: {name}")


def tensor_sha256(tensor):
    import torch
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model", "identity-receipt", "output", "admission",
        "frozen-runtime-manifest", "installed-runtime-manifest",
        "sglang-manifest",
    ):
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    for name in (
        "frozen-runtime-manifest-sha256", "installed-runtime-manifest-sha256",
        "sglang-manifest-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workspace-budget-mib", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.workspace_budget_mib <= 1536:
        raise ValueError("workspace budget must be in [1,1536] MiB")

    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    base = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / "tools"))

    admission = json.loads(args.admission.read_text())
    if (admission.get("status") != "ADMITTED_LAYERWISE_BF16_TRANSPORT_PENDING"
            or admission.get("physical") != 15
            or admission.get("capacity") != CAPACITY
            or admission.get("max_group_rows") != MAX_GROUP_ROWS
            or admission.get("workspace_budget_mib") != args.workspace_budget_mib):
        raise ValueError("launcher admission does not match this probe")
    expected_manifest_hashes = {
        "frozen_runtime": args.frozen_runtime_manifest_sha256,
        "installed_runtime": args.installed_runtime_manifest_sha256,
        "sglang": args.sglang_manifest_sha256,
    }
    if admission.get("identity_manifest_sha256") != expected_manifest_hashes:
        raise ValueError("admission identity manifests do not match probe arguments")
    probe_source_sha256 = digest(pathlib.Path(__file__).resolve())
    if admission.get("probe_source_sha256") != probe_source_sha256:
        raise ValueError("executed probe differs from the admitted frozen source")

    frozen_manifest = load_bound_manifest(
        args.frozen_runtime_manifest, args.frozen_runtime_manifest_sha256
    )
    installed_manifest = load_bound_manifest(
        args.installed_runtime_manifest, args.installed_runtime_manifest_sha256
    )
    sglang_manifest = load_bound_manifest(
        args.sglang_manifest, args.sglang_manifest_sha256
    )
    verify_tree(base / "runtime", frozen_manifest)
    required_runtime = ("layerwise.py", "dynamic_adapter.py", "expert_source.py")
    if not set(required_runtime) <= set(frozen_manifest):
        raise ValueError("frozen runtime manifest lacks layerwise dependencies")

    from probe_identity import verify_probe_identity

    source_identity = verify_probe_identity(
        model=args.model,
        receipt=args.identity_receipt,
        expected_model=pathlib.Path("/models") / REVISION,
        expected_receipt_sha256=RECEIPT_SHA256,
        revision=REVISION,
        manifest_sha256=MANIFEST_SHA256,
    )

    import torch
    import torch_npu
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from runtime.expert_source import OriginalExpertSource
    from runtime.framework_config import BufferIdentity
    from runtime.layerwise import (
        AscendBF16TransportBackend,
        LayerwiseTransport,
        group_topk_routes,
        make_reservation,
    )

    torch.set_num_threads(8)
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.04, 0)
    allocator_limit_bytes = int(torch.npu.mem_get_info()[1] * 0.04)
    if allocator_limit_bytes > 43_008 * 1024**2:
        raise RuntimeError("allocator limit exceeds the Round9 42GiB cap")

    acl = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so")
    physical = ctypes.c_int32(-1)
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)
    ]
    if (acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
            or physical.value != 15 or torch.npu.device_count() != 1):
        raise RuntimeError("probe is not isolated on physical NPU 15")
    if list(pathlib.Path("/dev").glob("davinci[0-9]*")) != [pathlib.Path("/dev/davinci15")]:
        raise RuntimeError("unexpected visible physical NPU devices")

    from kt_kernel import kt_kernel_ext
    native_path = pathlib.Path(kt_kernel_ext.__file__).resolve()
    native_root = pathlib.Path("/round9/candidate-runtime-python").resolve(strict=True)
    native_name = str(native_path.relative_to(native_root))
    native_sha256 = digest(native_path)
    if installed_manifest.get(native_name) != native_sha256:
        raise ValueError("loaded native extension differs from installed runtime identity")
    unquant_path = pathlib.Path(sys.modules[UnquantizedFusedMoEMethod.__module__].__file__).resolve()
    sglang_root = pathlib.Path("/round9/candidate-sglang-v8/python").resolve(strict=True)
    unquant_name = str(unquant_path.relative_to(sglang_root))
    if sglang_manifest.get(unquant_name) != digest(unquant_path):
        raise ValueError("loaded SGLang unquantized MoE differs from frozen identity")

    source = OriginalExpertSource(args.model, layers=LAYERS, experts=EXPERTS,
                                  hidden=HIDDEN, intermediate=INTERMEDIATE)

    def original(expert, layer=0):
        def read(name, shape):
            raw = source.read(layer, expert, name)
            return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
        gate = read("gate", (INTERMEDIATE, HIDDEN))
        up = read("up", (INTERMEDIATE, HIDDEN))
        down = read("down", (HIDDEN, INTERMEDIATE))
        return gate, up, down

    residents = tuple(range(0, EXPERTS, EXPERTS // CAPACITY))
    template = torch.nn.Module()
    template.top_k = TOPK
    template.num_experts = CAPACITY
    template.num_local_experts = CAPACITY
    template.num_gpu_experts = CAPACITY
    template.moe_ep_size = 1
    template.moe_ep_rank = 0
    template.moe_tp_size = 1
    template.moe_tp_rank = 0
    template.hidden_size = HIDDEN
    template.intermediate_size_per_partition = INTERMEDIATE
    template.num_fused_shared_experts = 0
    runner = MoeRunnerConfig(
        num_experts=CAPACITY, num_local_experts=CAPACITY,
        hidden_size=HIDDEN, intermediate_size_per_partition=INTERMEDIATE,
        layer_id=0, top_k=TOPK, params_dtype=torch.bfloat16,
    )
    template.moe_runner_config = runner
    template_method = UnquantizedFusedMoEMethod(
        use_triton_kernels=False, use_flashinfer_trtllm_moe=False
    )
    template.quant_method = template_method
    with torch.device("npu:0"):
        template_method.create_weights(
            template, CAPACITY, HIDDEN, INTERMEDIATE, torch.bfloat16
        )
    with torch.no_grad():
        for slot, logical in enumerate(residents):
            gate, up, down = original(logical)
            template.w13_weight[slot].copy_(torch.cat((gate, up), dim=0).to("npu:0"))
            template.w2_weight[slot].copy_(down.to("npu:0"))
    template_method.process_weights_after_loading(template)
    template_method.create_moe_runner(template, runner)
    if {int(torch_npu.get_npu_format(template.w13_weight)),
            int(torch_npu.get_npu_format(template.w2_weight))} != {29}:
        raise RuntimeError("template did not produce NZ29 resident weights")

    resident_mask = torch.zeros(EXPERTS, dtype=torch.bool)
    resident_mask[list(residents)] = True
    resident_mapping = torch.full((EXPERTS,), -1, dtype=torch.int32)
    resident_mapping[list(residents)] = torch.arange(CAPACITY, dtype=torch.int32)
    template_maps = template_method
    template_maps.gpu_experts_mask = resident_mask
    template_maps.gpu_experts_mask_cuda = resident_mask.to("npu:0")
    template_maps.logical_to_gpu_index = resident_mapping
    template_maps.logical_to_gpu_index_cuda = resident_mapping.to("npu:0")
    template_maps.gpu_index_to_logical = torch.tensor(residents, dtype=torch.int32)
    template_maps.wrapper = SimpleNamespace(gpu_experts_mask=resident_mask.clone())
    map_pointers = {
        name: getattr(template_maps, name).data_ptr()
        for name in ("gpu_experts_mask", "gpu_experts_mask_cuda",
                     "logical_to_gpu_index", "logical_to_gpu_index_cuda",
                     "gpu_index_to_logical")
    }
    map_pointers["wrapper.gpu_experts_mask"] = template_maps.wrapper.gpu_experts_mask.data_ptr()

    def assert_original_residents_exact():
        nd13 = torch_npu.npu_format_cast(template.w13_weight, 2).cpu()
        nd2 = torch_npu.npu_format_cast(template.w2_weight, 2).cpu()
        for slot, logical in enumerate(residents):
            gate, up, down = original(logical)
            if not torch.equal(nd13[slot].T, torch.cat((gate, up), dim=0)):
                raise AssertionError(("template_w13_changed", slot, logical))
            if not torch.equal(nd2[slot].T, down):
                raise AssertionError(("template_w2_changed", slot, logical))
        del nd13, nd2
        expected = {
            "gpu_experts_mask": resident_mask,
            "gpu_experts_mask_cuda": resident_mask,
            "logical_to_gpu_index": resident_mapping,
            "logical_to_gpu_index_cuda": resident_mapping,
            "gpu_index_to_logical": torch.tensor(residents, dtype=torch.int32),
            "wrapper.gpu_experts_mask": resident_mask,
        }
        for name, want in expected.items():
            value = (template_maps.wrapper.gpu_experts_mask if name.startswith("wrapper")
                     else getattr(template_maps, name))
            if value.data_ptr() != map_pointers[name] or not torch.equal(value.cpu(), want):
                raise AssertionError(("template_map_changed", name))

    assert_original_residents_exact()
    torch.npu.synchronize()
    base_peak = int(torch.npu.memory_allocated())
    reservation = make_reservation(
        capacity=CAPACITY, hidden=HIDDEN, intermediate=INTERMEDIATE,
        max_group_rows=MAX_GROUP_ROWS,
        allocator_limit_bytes=allocator_limit_bytes,
        observed_base_peak_bytes=base_peak,
        observed_workspace_bytes=args.workspace_budget_mib * 1024**2,
    )
    backend = AscendBF16TransportBackend(
        template, args.model, capacity=CAPACITY, layers=LAYERS, experts=EXPERTS,
        host_staging_limit=64 * 1024**2,
    )
    transport = LayerwiseTransport(
        backend, reservation, experts=EXPERTS, topk=TOPK, layers=1
    )
    if (backend.state != "READY" or backend.method._kt_return_fp32_contribution is not True
            or int(torch_npu.get_npu_format(backend.shadow.w13_weight)) != 29
            or int(torch_npu.get_npu_format(backend.shadow.w2_weight)) != 29):
        raise RuntimeError("private shadow bank is not the admitted FP32-contribution NZ29 path")

    if backend.shadow is template or backend.method is template_method:
        raise AssertionError("private shadow aliases the original resident module")
    for name in ("w13_weight", "w2_weight"):
        if getattr(backend.shadow, name).data_ptr() == getattr(template, name).data_ptr():
            raise AssertionError(("private_shadow_weight_alias", name))
    for forbidden in ("publish", "methods", "layers", "source"):
        if hasattr(backend.adapter, forbidden):
            raise AssertionError(("private_adapter_exposes_placement", forbidden))

    def assert_shadow_exact(logicals, layer=0):
        nd13 = torch_npu.npu_format_cast(backend.shadow.w13_weight, 2).cpu()
        nd2 = torch_npu.npu_format_cast(backend.shadow.w2_weight, 2).cpu()
        for slot, logical in enumerate(logicals):
            gate, up, down = original(logical, layer)
            if not torch.equal(nd13[slot].T, torch.cat((gate, up), dim=0)):
                raise AssertionError(("shadow_w13_identity", slot, logical))
            if not torch.equal(nd2[slot].T, down):
                raise AssertionError(("shadow_w2_identity", slot, logical))
        del nd13, nd2

    def routes(tokens, kind):
        if kind == "one_group":
            row = (7, 1, 31, 3, 29, 5, 23, 11, 19, 13)
            return [row for _ in range(tokens)]
        blocks = []
        for block in range(4):
            values = list(range(block * TOPK, (block + 1) * TOPK))
            blocks.append(tuple(values[3:] + values[:3]))
        return [blocks[row % len(blocks)] for row in range(tokens)]

    weights_cache = {}

    def cpu_reference(hidden, topk_ids, topk_weights):
        expected = torch.zeros(len(topk_ids), HIDDEN, dtype=torch.float32)
        uses = {}
        for row, ids in enumerate(topk_ids):
            for column, logical in enumerate(ids):
                uses.setdefault(logical, []).append((row, column))
        for logical, positions in uses.items():
            if logical not in weights_cache:
                weights_cache[logical] = tuple(value.float() for value in original(logical))
            gate, up, down = weights_cache[logical]
            indices = torch.tensor([row for row, _ in positions], dtype=torch.long)
            selected = hidden.index_select(0, indices).float()
            gate_value = (selected @ gate.T).bfloat16().float()
            up_value = (selected @ up.T).bfloat16().float()
            activated = (torch.nn.functional.silu(gate_value) * up_value).bfloat16().float()
            contribution = (activated @ down.T).bfloat16().float()
            scales = torch.tensor(
                [topk_weights[row, column] for row, column in positions],
                dtype=torch.float32,
            )
            expected.index_add_(0, indices, contribution * scales[:, None])
        return expected.bfloat16()

    report = {
        "schema": 1,
        "status": DEVICE_STATUS,
        "scope": "LAYER0_SYNTHETIC_REAL_NPU_NOT_FULL_MODEL_OR_SCHEDULER",
        "capacity": CAPACITY,
        "resident_capacity": CAPACITY,
        "max_group_rows": MAX_GROUP_ROWS,
        "geometry": {
            "hidden": HIDDEN, "intermediate": INTERMEDIATE,
            "experts": EXPERTS, "layers": LAYERS, "topk": TOPK,
            "dtype": "torch.bfloat16",
            "w13_shape": [CAPACITY, HIDDEN, 2 * INTERMEDIATE],
            "w2_shape": [CAPACITY, INTERMEDIATE, HIDDEN],
            "private_bank_bytes": reservation.private_bank_bytes,
        },
        "storage": "BF16",
        "fixture_layers": [0],
        "allocator_limit_bytes": allocator_limit_bytes,
        "constructor_peak_extra_bytes": backend.measured_preallocation_peak_bytes,
        "execution_peak_extra_bytes": None,
        "observed_workspace_bytes": reservation.observed_workspace_bytes,
        "source_sha256": {name: frozen_manifest[name] for name in required_runtime},
        "layerwise_sha256": frozen_manifest["layerwise.py"],
        "copy_math_sha256": frozen_manifest["dynamic_adapter.py"],
        "overlay_tree_sha256": args.sglang_manifest_sha256,
        "native_sha256": native_sha256,
        "probe_source_sha256": probe_source_sha256,
        "launcher_source_sha256": admission["launcher_source_sha256"],
        "source_identity": source_identity,
        "frozen_runtime_tree_identity": {
            "manifest_sha256": args.frozen_runtime_manifest_sha256,
            "files": len(frozen_manifest),
        },
        "installed_runtime_tree_identity": {
            "manifest_sha256": args.installed_runtime_manifest_sha256,
            "files": len(installed_manifest),
        },
        "sglang_tree_identity": {
            "manifest_sha256": args.sglang_manifest_sha256,
            "files": len(sglang_manifest),
            "loaded_unquantized_moe": unquant_name,
        },
        "source_layer_identity_sweep": [],
        "numeric_cases": [],
        "abort_probe": {
            "inflight_abort_synchronized": False,
            "backend_reusable_after_inflight_abort": False,
            "public_abort_reuse": False,
        },
        "unused_slot_probe": {
            "loaded_unused_slots": 0,
            "unchanged_after_execution": False,
            "numerically_ignored": False,
        },
    }

    try:
        # Exercise a real in-flight abort, its synchronization, and subsequent reuse.
        abort_ids = routes(1, "one_group")
        abort_groups = group_topk_routes(abort_ids, capacity=CAPACITY,
                                         experts=EXPERTS, topk=TOPK)
        abort_hidden = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="npu:0")
        abort_weights = torch.full((1, TOPK), 0.1, dtype=torch.float32, device="npu:0")
        abort_identity = BufferIdentity("direct-abort", 0, 1)
        abandoned = backend.execute_group(
            0, abort_groups[0], abort_hidden, abort_weights, abort_identity
        )
        backend.abort("direct-abort")
        del abandoned, abort_hidden, abort_weights
        if backend._current is not None or backend.state != "READY":
            raise RuntimeError("real in-flight abort did not leave backend reusable")
        report["abort_probe"]["inflight_abort_synchronized"] = True
        report["abort_probe"]["backend_reusable_after_inflight_abort"] = True

        # Resolve every configured logical layer through OriginalExpertSource.
        # This is source-address identity evidence; numerical execution remains
        # explicitly scoped to fixture layer 0.
        source_hashes = set()
        for layer in range(LAYERS):
            logical = (layer * 11 + 7) % EXPERTS
            hashes = {
                projection + "_sha256": hashlib.sha256(
                    source.read(layer, logical, projection)
                ).hexdigest()
                for projection in ("gate", "up", "down")
            }
            combined = hashlib.sha256(
                "".join(hashes.values()).encode("ascii")
            ).hexdigest()
            if combined in source_hashes:
                raise AssertionError(("duplicate_source_layer_identity", layer))
            source_hashes.add(combined)
            report["source_layer_identity_sweep"].append({
                "layer": layer,
                "logical_expert": logical,
                **hashes,
                "combined_sha256": combined,
            })

        # Fill every physical slot, then use only slots 0..9.  Exact hashes of
        # 10..31 must survive, and the numerical reference must still pass.
        unused_preload = tuple(range(100, 100 + CAPACITY))
        backend.adapter.copy_experts(0, tuple(enumerate(unused_preload)))
        backend.adapter.synchronize()
        assert_shadow_exact(unused_preload)

        def unused_slot_hash():
            nd13 = torch_npu.npu_format_cast(backend.shadow.w13_weight, 2).cpu()[TOPK:]
            nd2 = torch_npu.npu_format_cast(backend.shadow.w2_weight, 2).cpu()[TOPK:]
            value = hashlib.sha256()
            value.update(nd13.contiguous().view(torch.uint8).numpy().tobytes())
            value.update(nd2.contiguous().view(torch.uint8).numpy().tobytes())
            return value.hexdigest()

        unused_before = unused_slot_hash()
        report["unused_slot_probe"]["loaded_unused_slots"] = CAPACITY - TOPK

        execution_peaks = []
        request_number = 0
        for tokens in (1, 10, 32):
            for kind in ("one_group", "four_block_multigroup"):
                request_number += 1
                logical_rows = routes(tokens, kind)
                groups = group_topk_routes(
                    logical_rows, capacity=CAPACITY, experts=EXPERTS, topk=TOPK
                )
                if tokens > 1 and kind == "four_block_multigroup" and len(groups) < 2:
                    raise AssertionError("multigroup fixture did not cross capacity")
                torch.manual_seed(19000 + request_number)
                hidden_cpu = torch.randn(tokens, HIDDEN).bfloat16()
                weights_cpu = torch.softmax(torch.randn(tokens, TOPK), dim=-1).float()
                expected = cpu_reference(hidden_cpu, logical_rows, weights_cpu)
                # Start before request inputs reach the NPU: this delta is
                # relative to the persistent template plus private shadow bank.
                torch.npu.synchronize()
                execution_baseline = int(torch.npu.memory_allocated())
                torch.npu.reset_peak_memory_stats()
                hidden = hidden_cpu.to("npu:0")
                ids = torch.tensor(logical_rows, dtype=torch.int64, device="npu:0")
                weights = weights_cpu.to("npu:0")
                request_id = f"layerwise-{request_number}"
                transport.begin(request_id, request_number)
                actual = transport.execute_layer(
                    request_id, request_number, 0, hidden, ids, weights
                )
                transport.finish(request_id, request_number)
                torch.npu.synchronize()
                execution_extra = max(
                    0, int(torch.npu.max_memory_allocated()) - execution_baseline
                )
                execution_peaks.append(execution_extra)
                actual_cpu = actual.cpu()
                if actual_cpu.dtype != torch.bfloat16 or not bool(torch.isfinite(actual_cpu).all()):
                    raise AssertionError("layerwise output is not finite BF16")
                relative_l2 = float(
                    (actual_cpu.float() - expected.float()).double().norm()
                    / expected.float().double().norm().clamp_min(1e-30)
                )
                if relative_l2 > 0.01:
                    raise AssertionError((tokens, kind, relative_l2))
                assert_shadow_exact(groups[-1].logical_experts)
                assert_original_residents_exact()
                if backend._current is not None or backend._synchronized is not None:
                    raise AssertionError("group lease survived synchronized release")
                row = {
                    "tokens": tokens,
                    "kind": kind,
                    "groups": len(groups),
                    "relative_l2": relative_l2,
                    "relative_l2_tolerance": 0.01,
                    "finite": True,
                    "execution_peak_extra_bytes": execution_extra,
                    "output_sha256": tensor_sha256(actual_cpu),
                    "all_top10_routes_executed": True,
                    "last_group_original_weights_exact": True,
                    "private_adapter_copy_only": True,
                    "template_residents_unchanged": True,
                }
                report["numeric_cases"].append(row)
                print(json.dumps(row), flush=True)
                if tokens == 1 and kind == "one_group":
                    if unused_slot_hash() != unused_before:
                        raise AssertionError("unused shadow slots changed")
                    report["unused_slot_probe"]["unchanged_after_execution"] = True
                    report["unused_slot_probe"]["numerically_ignored"] = True
                del actual, actual_cpu, expected, hidden, ids, weights

        # Exercise the public transport abort and request reuse after all kernels.
        transport.begin("public-abort", 1000)
        transport.abort("public-abort", 1000)
        transport.begin("post-abort-reuse", 1001)
        transport.abort("post-abort-reuse", 1001)
        report["abort_probe"]["public_abort_reuse"] = True
        assert_original_residents_exact()
        execution_peak = max(execution_peaks)
        if execution_peak > reservation.observed_workspace_bytes:
            raise RuntimeError("execution workspace exceeded the admitted bound")
        report["execution_peak_extra_bytes"] = execution_peak
        report["copied_original_bf16_bytes"] = backend.adapter.copied_bytes
        report["original_resident_weights_and_maps_exact"] = True
        report["private_shadow_has_no_resident_weight_alias"] = True
        report["status"] = "PASS_LAYERWISE_BF16_TRANSPORT"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
