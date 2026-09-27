"""Bounded synthetic KV -> CPU records -> BF16 NPU QSA reference probe.

No model hook or allocator page is freed. This does not qualify native FP8,
long contexts, or a working HiCache tier.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time


SOURCE_FILES = ('context_profile.py', 'context_page_store.py',
                'kv_quant_reference.py', 'context_staging.py', 'qsa.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'identity-receipt', 'admission', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(base), str(base / 'tools')]
    source_sha = {name: hashlib.sha256((base/'runtime'/name).read_bytes()).hexdigest()
                  for name in SOURCE_FILES}
    admission = json.loads(args.admission.read_text())
    if (admission.get('status') != 'ADMITTED_CONTEXT_REFERENCE_PENDING' or
            admission.get('physical') != 15 or admission.get('resource_mode') != 'exclusive' or
            admission.get('source_sha256') != source_sha or
            admission.get('probe_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
            not 0 <= time.time() - admission.get('time', 0) < 60):
        raise RuntimeError('fresh source-bound context component admission required')
    if args.output.exists(): raise FileExistsError(args.output)
    from probe_identity import verify_probe_identity
    rev = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
    identity = verify_probe_identity(model=args.model, receipt=args.identity_receipt,
        expected_model=Path('/models')/rev, revision=rev,
        expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',
        manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
    from runtime.context_profile import ContextStateIdentity, derive_qwen4_tp1_geometry
    geometry = derive_qwen4_tp1_geometry(json.loads((args.model/'config.json').read_text()),
                                        config_scope='full_config')
    from runtime.context_page_store import PageKey, RequestLocalPageStore, make_bf16_layer_payload, QSA_LAYER_IDS
    from runtime.kv_quant_reference import encode_kv_page, decode_kv_page
    from runtime.context_staging import stage_selected_rows
    import torch
    import torch_npu
    from runtime.qsa import sparse_gqa
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(.01, 0)
    physical = ctypes.c_int32(-1)
    acl = ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    if (acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0 or physical.value != 15 or
            torch.npu.device_count() != 1 or
            set(Path('/dev').glob('davinci[0-9]*')) != {Path('/dev/davinci15')}):
        raise RuntimeError('context reference requires isolated physical15')
    torch.npu.synchronize(); torch.npu.reset_peak_memory_stats()
    baseline = int(torch.npu.memory_allocated())
    def byte_value(value):
        return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    owner = ContextStateIdentity('context-device-reference', None, 1, 'QSA_KV')
    store = RequestLocalPageStore(max_pages=2, max_host_bytes=1 << 20)
    # Actual NPU -> host exports cover every real QSA layer, using synthetic KV.
    originals = {}
    payloads = [{}, {}]
    for layer in QSA_LAYER_IDS:
        k = torch.randn((8, 2, 256), dtype=torch.float32).to(torch.bfloat16).npu()
        v = torch.randn((8, 2, 256), dtype=torch.float32).to(torch.bfloat16).npu()
        originals[layer] = (k, v)
        for page in range(2):
            payloads[page][layer] = make_bf16_layer_payload(layer, 4,
                byte_value(k[page*4:(page+1)*4]), byte_value(v[page*4:(page+1)*4]))
    for page in range(2): store.publish_export(PageKey(owner.request_id, 1, page), payloads[page])
    selected = torch.tensor([[7, 0, 4, 3, 7, -1]], dtype=torch.int64, device='cpu')
    cases = []
    for layer in QSA_LAYER_IDS:
        leases = []
        def read_record(key, layer_id):
            lease = store.pin(key); leases.append(lease)
            return store.read(lease)
        staged = stage_selected_rows(owner, layer, selected, current_logical_position=7,
                                      record_reader=read_record)
        k, v = originals[layer]
        query = torch.randn((1, 4, 256), dtype=torch.float32).to(torch.bfloat16).npu()
        oracle = sparse_gqa(query, k, v, selected.npu())
        restored_k, restored_v = staged.keys.npu(), staged.values.npu()
        actual = sparse_gqa(query, restored_k, restored_v, staged.remapped.npu())
        torch.npu.synchronize()
        exact = torch.equal(actual.cpu(), oracle.cpu())
        if not exact: raise AssertionError('exact BF16 host staging changed QSA output')
        for lease in leases: store.release(lease)
        records = {page: encode_kv_page(owner, page, layer,
                    k[page*4:(page+1)*4].cpu(), v[page*4:(page+1)*4].cpu()) for page in range(2)}
        quant = stage_selected_rows(owner, layer, selected, current_logical_position=7,
                                    record_reader=lambda key, lid: records[key.logical_page])
        decoded = [decode_kv_page(records[page], owner, logical_page=page, layer_id=layer)
                   for page in range(2)]
        full_k = torch.cat([item[0] for item in decoded]).npu()
        full_v = torch.cat([item[1] for item in decoded]).npu()
        decoded_oracle = sparse_gqa(query, full_k, full_v, selected.npu())
        quant_actual = sparse_gqa(query, quant.keys.npu(), quant.values.npu(), quant.remapped.npu())
        torch.npu.synchronize()
        quant_exact = torch.equal(decoded_oracle.cpu(), quant_actual.cpu())
        if not quant_exact: raise AssertionError('FP8 compact remap differs from decoded full-page oracle')
        rel = float(torch.linalg.vector_norm((quant_actual.float()-oracle.float()).cpu()) /
                    torch.linalg.vector_norm(oracle.float().cpu()).clamp_min(1e-12))
        if not rel <= .10: raise AssertionError('synthetic FP8 QSA fixture exceeds relative-L2 bound')
        cases.append(dict(layer=layer, bf16_output_exact=exact, fp8_decoded_output_exact=quant_exact,
            fp8_vs_original_relative_l2=rel, compact_bytes=staged.staging_bytes,
            output_sha256=hashlib.sha256(byte_value(actual)).hexdigest()))
    # Delay invalidation behind a live reference; it must never become readable
    # under a new request epoch and must reach zero once that reference drains.
    last = store.pin(PageKey(owner.request_id, 1, 0))
    if store.invalidate(owner.request_id, 1): raise AssertionError('live host page freed early')
    store.release(last)
    if store.snapshot()['pages'] or store.snapshot()['host_bytes']:
        raise AssertionError('host page ownership did not drain')
    stale_rejected = False
    try: store.read(last)
    except ValueError: stale_rejected = True
    if not stale_rejected: raise AssertionError('stale host lease accepted')
    peak = int(torch.npu.max_memory_allocated()) - baseline
    result = dict(schema=1, status='PASS_CONTEXT_REFERENCE_QSA_COMPONENT',
        scope='SYNTHETIC_KV_CPU_RECORD_BF16_NPU_QSA_NOT_MODEL_OR_HICACHE',
        physical_device=15, source_identity=identity, source_sha256=source_sha,
        torch_version=str(torch.__version__), torch_npu_version=str(torch_npu.__version__),
        qsa_layers=geometry.qsa_layers, cases=cases, host_store_final=store.snapshot(),
        stale_lease_rejected=stale_rejected, allocator_free_calls=0,
        native_fp8_qualified=False, peak_extra_bytes=peak)
    with args.output.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'], flush=True)


if __name__ == '__main__': main()
