"""Install isolated default-OFF framework hooks over a fresh candidate overlay."""
import argparse
import hashlib
import json
from pathlib import Path
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
    if source.count(before) != 1 or source.count(after) != 1 or 'framework_before(self' in source:
        raise ValueError('unexpected or already installed framework scheduler seam')
    source = source.replace(before, before +
        '                from sglang.srt.layers.round9.framework_hooks import framework_before\n'
        '                framework_before(self, batch)\n')
    return source.replace(after,
        '            from sglang.srt.layers.round9.framework_hooks import framework_after\n'
        '            framework_after(self)\n' + after)


def patch_model_runner(source):
    marker = '        self.init_memory_pool()\n'
    if source.count(marker) != 1 or 'prefill_model_ready(self)' in source:
        raise ValueError('unexpected or already installed model-runner prefill seam')
    return source.replace(marker,
        '        from sglang.srt.layers.round9.prefill_binding import prefill_model_ready\n'
        '        prefill_model_ready(self)\n'
        '        from sglang.srt.layers.round9.ple_binding import ple_model_ready\n'
        '        ple_model_ready(self)\n' + marker)


def patch_ple_prepare(source):
    marker = '        embeddings = self._table.gather_rows(row_ids)\n'
    if source.count(marker) != 1 or 'async_ple.prepare(input_ids' in source:
        raise ValueError('unexpected or already installed PLE prepare seam')
    return source.replace(marker,
        '        async_ple = getattr(self, "_round9_async_ple", None)\n'
        '        embeddings = (async_ple.prepare(input_ids, slot, row_ids, self._table)\n'
        '                      if async_ple is not None else self._table.gather_rows(row_ids))\n')


def patch_kv_profiler(source):
    marker = ('        if self.mambaish_config is not None:\n'
              '            rest_memory = self.handle_max_mamba_cache(rest_memory)\n')
    if source.count(marker) != 1 or '_round9_prefill_workspace_bytes' in source:
        raise ValueError('unexpected or already installed KV profiler reservation seam')
    return source.replace(marker,
        '        rest_memory -= (getattr(self, "_round9_prefill_workspace_bytes", 0) +\n'
        '                        getattr(self, "_round9_ple_workspace_bytes", 0)) / (1 << 30)\n' + marker)


def install_framework(baseline, target):
    report = install(baseline, target)
    target = Path(target)
    for name, patch in [('sglang/srt/models/qwen4_exp.py', lambda text:patch_ple_prepare(patch_model(text))),
                        ('sglang/srt/managers/scheduler.py', patch_scheduler),
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
                 'budgeted_migration.py'):
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
