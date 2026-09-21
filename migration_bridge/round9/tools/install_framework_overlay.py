"""Install isolated default-OFF framework hooks over a fresh candidate overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from install_candidate_overlay import install, tree_files, tree_digest


def patch_model(source):
    marker = '    cfg = kt_ep_wrapper.create_kt_config_from_server_args(args, layer_id)\n'
    if source.count(marker) != 1 or 'prepare_masks(args' in source:
        raise ValueError('unexpected or already installed framework model seam')
    source = source.replace(marker,
        '    from sglang.srt.layers.round9.framework_hooks import prepare_masks\n'
        '    round9_resident_capacity = prepare_masks(args, kt_ep_wrapper)\n' + marker)
    validate = '            dynamic_boundary_installed=installed_scheduler_boundary())\n'
    bank = '        bank_identity = bind_verified_banks(candidate_path, args.kt_weight_path, args.model_path)\n'
    if source.count(validate) != 1 or source.count(bank) != 1:
        raise ValueError('candidate model seam missing capacity insertion points')
    validator_call = '        validate_kt_config(cfg,'
    if source.count(validator_call) != 1:
        raise ValueError('candidate validator seam missing')
    source = source.replace(validator_call, '        candidate = validate_kt_config(cfg,')
    source = source.replace(validate,
        '            dynamic_boundary_installed=installed_scheduler_boundary(),\n'
        '            resident_capacity=round9_resident_capacity)\n')
    return source.replace(bank, bank +
        '        from sglang.srt.layers.round9.framework_hooks import validate_q32_model_gate\n'
        '        validate_q32_model_gate(round9_resident_capacity, candidate, bank_identity)\n')


def patch_scheduler(source):
    before = '                scheduler_before(self, batch)\n'
    after = '            scheduler_after(self)\n'
    aborts = re.findall(
        r'^    def abort_request\(self, recv_req(?:: [^)]+)?\):\n',
        source,
        flags=re.MULTILINE,
    )
    abort = aborts[0] if len(aborts) == 1 else None
    if (source.count(before) != 1 or source.count(after) != 1
            or abort is None or 'framework_before(self' in source
            or 'scheduler_abort_received(self' in source):
        raise ValueError('unexpected or already installed framework scheduler seam')
    source = source.replace(before, before +
        '                from sglang.srt.layers.round9.framework_hooks import framework_before\n'
        '                framework_before(self, batch)\n')
    source = source.replace(after,
        '            from sglang.srt.layers.round9.framework_hooks import framework_after\n'
        '            framework_after(self)\n' + after)
    return source.replace(abort, abort +
        '        from sglang.srt.layers.round9.cancel_lifecycle import scheduler_abort_received\n'
        '        scheduler_abort_received(self, recv_req)\n')


def patch_tokenizer_manager(source):
    marker = '    def abort_request(self, rid: str = "", abort_all: bool = False):\n'
    if source.count(marker) != 1 or 'tokenizer_abort_requested(self' in source:
        raise ValueError('unexpected or already installed tokenizer cancel seam')
    return source.replace(marker, marker +
        '        from sglang.srt.layers.round9.cancel_lifecycle import tokenizer_abort_requested\n'
        '        tokenizer_abort_requested(self, rid, abort_all)\n')


def patch_http_server(source):
    marker = ('async def abort_request(obj: AbortReq, request: Request):\n'
              '    """Abort a request."""\n')
    fallback = ('async def abort_request(obj, request):\n'
                '    """Abort a request."""\n')
    selected = marker if source.count(marker) == 1 else fallback
    if source.count(selected) != 1 or 'http_abort_received(obj.rid' in source:
        raise ValueError('unexpected or already installed HTTP cancel seam')
    return source.replace(selected, selected +
        '    from sglang.srt.layers.round9.cancel_lifecycle import http_abort_received\n'
        '    http_abort_received(obj.rid, obj.abort_all)\n')


def patch_model_runner(source):
    marker = '        self.init_memory_pool()\n'
    if source.count(marker) != 1 or 'prefill_model_ready(self)' in source:
        raise ValueError('unexpected or already installed model-runner prefill seam')
    return source.replace(marker,
        '        from sglang.srt.layers.round9.prefill_binding import prefill_model_ready\n'
        '        prefill_model_ready(self)\n'
        '        from sglang.srt.layers.round9.ple_binding import ple_model_ready\n'
        '        ple_model_ready(self)\n'
        '        from sglang.srt.layers.round9.context_reference_binding import context_reference_prepare\n'
        '        context_reference_prepare(self)\n' + marker +
        '        from sglang.srt.layers.round9.context_binding import context_pools_ready\n'
        '        context_pools_ready(self)\n'
        '        from sglang.srt.layers.round9.context_reference_binding import context_reference_ready\n'
        '        context_reference_ready(self)\n'
        '        from sglang.srt.layers.round9.streaming_binding import streaming_model_ready\n'
        '        streaming_model_ready(self)\n')


def patch_ple_prepare(source):
    marker = '        embeddings = self._table.gather_rows(row_ids)\n'
    if source.count(marker) != 1 or 'async_ple.prepare(input_ids' in source:
        raise ValueError('unexpected or already installed PLE prepare seam')
    return source.replace(marker,
        '        async_ple = getattr(self, "_round9_async_ple", None)\n'
        '        embeddings = (async_ple.prepare(input_ids, slot, row_ids, self._table)\n'
        '                      if async_ple is not None else self._table.gather_rows(row_ids))\n')


def patch_context_consumer(source):
    marker = '        attn_output = sparse_gqa(q, keys, values, selected)\n'
    if source.count(marker) != 1 or 'context_consumer.consume(' in source:
        raise ValueError('unexpected or already installed context consumer seam')
    return source.replace(marker,
        '        context_consumer = getattr(self, "_round9_context_consumer", None)\n'
        '        attn_output = (context_consumer.consume(\n'
        '            int(self.attn.layer_id), q, keys, values, selected, positions, sparse_gqa)\n'
        '            if context_consumer is not None\n'
        '            else sparse_gqa(q, keys, values, selected))\n')


def patch_kv_profiler(source):
    marker = ('        if self.mambaish_config is not None:\n'
              '            rest_memory = self.handle_max_mamba_cache(rest_memory)\n')
    if source.count(marker) != 1 or '_round9_prefill_workspace_bytes' in source:
        raise ValueError('unexpected or already installed KV profiler reservation seam')
    return source.replace(marker,
        '        rest_memory -= (getattr(self, "_round9_prefill_workspace_bytes", 0) +\n'
        '                        getattr(self, "_round9_ple_workspace_bytes", 0) +\n'
        '                        getattr(self, "_round9_context_workspace_bytes", 0)) / (1 << 30)\n' + marker)


def install_framework(baseline, target):
    report = install(baseline, target)
    target = Path(target)
    for name, patch in [('sglang/srt/models/qwen4_exp.py', lambda text:patch_context_consumer(patch_ple_prepare(patch_model(text)))),
                        ('sglang/srt/managers/scheduler.py', patch_scheduler),
                        ('sglang/srt/managers/tokenizer_manager.py', patch_tokenizer_manager),
                        ('sglang/srt/entrypoints/http_server.py', patch_http_server),
                        ('sglang/srt/model_executor/model_runner.py', patch_model_runner),
                        ('sglang/srt/model_executor/model_runner_kv_cache_mixin.py', patch_kv_profiler)]:
        path = target / name
        value = patch(path.read_text())
        compile(value, name, 'exec')
        path.write_text(value)
    runtime = Path(__file__).resolve().parents[1] / 'runtime'
    for name in ('framework_config.py', 'route_stats.py', 'frequency.py',
                 'prefill_telemetry.py', 'framework_hooks.py', 'q32_gate.py',
                 'layerwise.py', 'layerwise_gate.py', 'prefill_runtime.py', 'prefill_binding.py',
                 'prefill_end_boundary.py', 'live_request_state.py',
                 'host_budget.py','ple_async.py','ple_rows.py','ple_runtime.py','ple_gate.py','ple_binding.py',
                 'budgeted_migration.py','joint_budget_gate.py',
                 'context_profile.py','context_measurement.py','context_binding.py',
                 'context_page_store.py','kv_quant_reference.py','context_staging.py','qsa.py',
                 'context_reference_gate.py','context_reference_runtime.py',
                 'context_reference_binding.py','streaming_prefill.py','streaming_binding.py',
                 'cancel_lifecycle.py'):
        shutil.copy2(runtime / name, target / 'sglang/srt/layers/round9' / name)
        report['modules'][name] = hashlib.sha256((runtime / name).read_bytes()).hexdigest()
    report['framework_status'] = 'F1_INSTALLED_NOT_DEVICE_QUALIFIED_DEFAULT_OFF'
    report['candidate_model_sha256'] = hashlib.sha256((target / 'sglang/srt/models/qwen4_exp.py').read_bytes()).hexdigest()
    report['candidate_files'] = tree_files(target)
    report['candidate_tree_sha256'] = tree_digest(report['candidate_files'])
    (target / 'round9-candidate-overlay.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(install_framework(a.baseline_python, a.output), indent=2))
