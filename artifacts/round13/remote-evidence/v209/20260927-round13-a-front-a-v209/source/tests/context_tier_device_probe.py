"""Actual NPU byte-identity probe for the synchronous request-local tier.

This exercises small synthetic BF16 pages on physical15. It does not install a
model allocator hook, release SGLang KV pages, or qualify HiCache.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time


SOURCE_FILES = ("context_page_store.py", "context_tier.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "identity-receipt", "admission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(base), str(base / "tools")]
    source_sha = {
        name: hashlib.sha256((base / "runtime" / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    admission = json.loads(args.admission.read_text())
    if (admission.get("status") != "ADMITTED_CONTEXT_TIER_PENDING"
            or admission.get("physical") != 15
            or admission.get("resource_mode") != "exclusive"
            or admission.get("source_sha256") != source_sha
            or admission.get("probe_sha256")
                != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or not 0 <= time.time() - admission.get("time", 0) < 60):
        raise RuntimeError("fresh source-bound context tier admission required")
    if args.output.exists():
        raise FileExistsError(args.output)

    from probe_identity import verify_probe_identity
    revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    identity = verify_probe_identity(
        model=args.model, receipt=args.identity_receipt,
        expected_model=Path("/models") / revision, revision=revision,
        expected_receipt_sha256="7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239",
        manifest_sha256="62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e",
    )

    import torch
    import torch_npu
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(.01, 0)
    physical = ctypes.c_int32(-1)
    acl = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so")
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    if (acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
            or physical.value != 15 or torch.npu.device_count() != 1
            or set(Path("/dev").glob("davinci[0-9]*")) != {Path("/dev/davinci15")}):
        raise RuntimeError("context tier requires isolated physical15")

    from runtime.context_page_store import (
        PageKey, QSA_LAYER_IDS, RequestLocalPageStore, make_bf16_layer_payload,
    )
    from runtime.context_tier import RequestLocalContextTier

    def tensor_bytes(value):
        return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    baseline = int(torch.npu.memory_allocated())
    slots = {}
    generations = {0: 0, 1: 0}

    class Allocator:
        def allocate(self):
            if 1 in slots:
                raise RuntimeError("bounded spare slot is already allocated")
            generations[1] += 1
            slots[1] = {}
            return 1, generations[1]

        def release(self, physical_page, generation):
            if generations.get(physical_page) != generation or physical_page not in slots:
                raise RuntimeError("stale physical generation release")
            del slots[physical_page]

    class Mapping:
        def __init__(self):
            self.entries = {}

        def publish(self, key, physical_page, generation):
            if key in self.entries:
                raise RuntimeError("duplicate logical page mapping")
            self.entries[key] = (physical_page, generation)

        def clear(self, key, physical_page, generation):
            if self.entries.get(key) != (physical_page, generation):
                raise RuntimeError("stale logical page mapping clear")
            del self.entries[key]

    store = RequestLocalPageStore(max_pages=2, max_host_bytes=1 << 20)
    allocator, mapping = Allocator(), Mapping()

    def exporter(key, physical_page, generation):
        if generations[physical_page] != generation:
            raise RuntimeError("stale export generation")
        return {
            layer: make_bf16_layer_payload(
                layer, 4, tensor_bytes(slots[physical_page][layer][0]),
                tensor_bytes(slots[physical_page][layer][1]))
            for layer in QSA_LAYER_IDS
        }

    def importer(record, physical_page, generation):
        if generations[physical_page] != generation:
            raise RuntimeError("stale import generation")
        for payload in record.layers:
            shape = payload.shape
            key = torch.frombuffer(bytearray(payload.k_bytes), dtype=torch.bfloat16).reshape(shape).npu()
            value = torch.frombuffer(bytearray(payload.v_bytes), dtype=torch.bfloat16).reshape(shape).npu()
            slots[physical_page][payload.layer_id] = (key, value)

    tier = RequestLocalContextTier(
        store, allocator=allocator, mapping=mapping,
        exporter=exporter, importer=importer)
    key = PageKey("context-tier-device", 1, 0)
    slots[0] = {
        layer: (
            torch.randn((4, 2, 256), dtype=torch.float32).to(torch.bfloat16).npu(),
            torch.randn((4, 2, 256), dtype=torch.float32).to(torch.bfloat16).npu(),
        )
        for layer in QSA_LAYER_IDS
    }
    original = {
        layer: tuple(hashlib.sha256(tensor_bytes(value)).hexdigest()
                     for value in slots[0][layer])
        for layer in QSA_LAYER_IDS
    }
    mapping.publish(key, 0, 0)
    tier.register_resident(key, 0, 0)
    spilled = tier.evict(key, current_logical_page=2)
    if spilled.state != "SPILLED" or slots or mapping.entries:
        raise AssertionError("actual NPU page did not leave resident authority")
    restored = tier.prefetch(key)
    if ((restored.physical_page, restored.physical_generation) != (1, 1)
            or mapping.entries.get(key) != (1, 1)):
        raise AssertionError("actual NPU page did not publish a new generation")
    actual = {
        layer: tuple(hashlib.sha256(tensor_bytes(value)).hexdigest()
                     for value in slots[1][layer])
        for layer in QSA_LAYER_IDS
    }
    if actual != original:
        raise AssertionError("HBM to host to HBM page bytes changed")
    other_rejected = False
    try:
        tier.prefetch(PageKey("other-request", 1, 0))
    except ValueError:
        other_rejected = True
    if not other_rejected:
        raise AssertionError("request-local page ownership crossed requests")
    if not tier.invalidate_request(key.request_id, key.allocation_epoch):
        raise AssertionError("request tier did not invalidate")
    torch.npu.synchronize()
    peak = int(torch.npu.max_memory_allocated()) - baseline
    snapshot = tier.snapshot()
    if slots or mapping.entries or snapshot["pages"] or snapshot["host"]["pages"]:
        raise AssertionError("actual tier did not drain all ownership")
    result = {
        "schema": 1,
        "status": "PASS_CONTEXT_TIER_NPU_BYTE_IDENTITY",
        "scope": "SYNTHETIC_12_LAYER_PAGE_NOT_MODEL_ALLOCATOR_OR_HICACHE",
        "physical_device": 15,
        "source_identity": identity,
        "source_sha256": source_sha,
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "qsa_layers": list(QSA_LAYER_IDS),
        "original_physical_generation": [0, 0],
        "restored_physical_generation": [1, 1],
        "all_layer_bytes_exact": actual == original,
        "request_isolation_rejected": other_rejected,
        "final": snapshot,
        "peak_extra_bytes": peak,
        "model_allocator_hook_installed": False,
        "hbm_capacity_savings_qualified": False,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
