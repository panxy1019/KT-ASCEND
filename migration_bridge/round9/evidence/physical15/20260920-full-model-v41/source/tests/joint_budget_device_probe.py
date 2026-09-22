"""Physical15 synthetic PLE/expert contention probe; no expert-math claim."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys


def _digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'identity-receipt', 'admission', 'ple-gate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ple-gate-sha256', required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(base), str(base / 'tools')]

    from probe_identity import verify_probe_identity
    from runtime import budgeted_migration, host_budget, joint_budget_gate, ple_async
    from runtime import framework_config, ple_rows, ple_gate

    source_sha = {
        name: hashlib.sha256((base / 'runtime' / name).read_bytes()).hexdigest()
        for name in joint_budget_gate.SOURCE_FILES
    }
    admission = json.loads(args.admission.read_text())
    if (admission.get('status') != 'ADMITTED_JOINT_BUDGET_PENDING' or
            admission.get('physical') != 15 or
            admission.get('source_sha256') != source_sha or
            admission.get('probe_sha256') !=
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
            admission.get('ple_gate_sha256') != args.ple_gate_sha256):
        raise RuntimeError('missing source-bound joint budget admission')
    if args.output.exists():
        raise FileExistsError(args.output)

    revision = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
    identity = verify_probe_identity(
        model=args.model, receipt=args.identity_receipt,
        expected_model=Path('/models') / revision, revision=revision,
        expected_receipt_sha256=
            '7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',
        manifest_sha256=
            '62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')

    import torch
    import torch_npu
    if hashlib.sha256(args.ple_gate.read_bytes()).hexdigest() != args.ple_gate_sha256:
        raise RuntimeError('standalone PLE gate SHA changed')
    ple_sources = {
        module.__name__.rsplit('.', 1)[-1] + '.py':
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (ple_async, ple_rows, host_budget, framework_config)
    }
    ple_gate.require_ple_device_gate(
        args.ple_gate, args.ple_gate_sha256,
        source_identity=identity, source_sha256=ple_sources,
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__))

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(.01, 0)
    physical = ctypes.c_int32(-1)
    acl = ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    if (acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0 or
            physical.value != 15 or torch.npu.device_count() != 1 or
            set(Path('/dev').glob('davinci[0-9]*')) != {Path('/dev/davinci15')}):
        raise RuntimeError('joint budget probe requires isolated physical15')
    checks = {name: False for name in joint_budget_gate.CHECKS}
    checks['physical_mapping'] = True

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    baseline = int(torch.npu.memory_allocated())
    capacity = joint_budget_gate.CONFIGURED_BUDGET['capacity']
    reserve = joint_budget_gate.CONFIGURED_BUDGET['reserve']
    budget = host_budget.HostWorkBudget(
        enabled=True, **capacity,
        reserve_host_bytes=reserve['host_bytes'],
        reserve_h2d_bytes=reserve['h2d_bytes'],
        reserve_pinned_bytes=reserve['pinned_bytes'])
    reader = ple_rows.ShardedBF16Rows(args.model)
    ring = ple_async.AsyncPLERing(
        reader, enabled=True, max_rows=512, budget=budget)
    snapshots = []

    def snapshot(stage):
        value = budget.snapshot()
        item = {name: value[name] for name in
                ('used', 'allocations', 'active_requests', 'in_flight')}
        item['stage'] = stage
        if any(item['used'][name] > capacity[name] for name in capacity):
            raise RuntimeError('joint budget exceeded configured capacity')
        expected = joint_budget_gate.EXPECTED_SNAPSHOTS[stage]
        if any(item[name] != expected[name] for name in
               ('used', 'allocations', 'active_requests', 'in_flight')):
            raise RuntimeError(f'joint budget {stage} ledger differs')
        snapshots.append(item)
        return value

    class SyntheticNPUAdapter:
        expert_bytes = joint_budget_gate.EXPERT_BYTES

        def __init__(self):
            self.destination = torch.empty(
                (joint_budget_gate.MIGRATION_PROBE_EXPERTS,
                 self.expert_bytes // 2),
                dtype=torch.bfloat16, device='npu')
            self.copied = 0

        def copy_experts(self, layer, moves):
            if layer != 0 or tuple(slot for slot, _ in moves) != (0, 1):
                raise ValueError('synthetic migration requires exact layer0/two-slot case')
            snapshot('migration_live')
            for slot, logical in moves:
                payload = torch.full(
                    (self.expert_bytes // 2,), float(logical + 1),
                    dtype=torch.bfloat16, device='cpu')
                self.destination[slot].copy_(payload, non_blocking=False)
                self.copied += self.expert_bytes

        @staticmethod
        def synchronize():
            torch.npu.synchronize()

    adapter = SyntheticNPUAdapter()
    shared_requests = set()
    migration = budgeted_migration.BudgetedMigration(
        adapter, budget, lambda: ('joint-request', 0),
        requests=shared_requests)

    flat = [0, reader.total_rows - 1]
    for end in reader.ends[:-1]:
        flat.extend((end - 1, end))
    flat = (flat * (512 // len(flat) + 1))[:512]
    rows = [flat[index:index + 16] for index in range(0, 512, 16)]
    expected_ple = reader.gather_bytes(flat)
    current = ring.submit_exact(
        'joint-request', 0, rows, phase=ple_async.PREFILL_INPUT,
        priority=host_budget.Priority.PLE_CURRENT)
    if current is None:
        raise RuntimeError('required PLE current ticket was not admitted')
    consumer = torch.npu.Stream()
    with torch.npu.stream(consumer):
        device_rows = ring.wait(current, consumer_stream=consumer)
        copied_ple = device_rows.clone()
    snapshot('ple_current_live')
    checks['ple_current_overlap'] = True

    prefetch = ring.submit_exact(
        'joint-request', 0, [rows[0]], phase=ple_async.SAMPLED_DECODE,
        priority=host_budget.Priority.PLE_PREFETCH)
    if prefetch is None:
        raise RuntimeError('reserved budget failed to admit bounded prefetch case')
    snapshot('prefetch_live')

    moves = ((0, 7), (1, 11))
    migration.copy_experts(0, moves)
    if adapter.copied != len(moves) * joint_budget_gate.EXPERT_BYTES:
        raise RuntimeError('synthetic migration byte count differs')
    if budget.snapshot()['allocations'] != 4:
        raise RuntimeError('required migration did not release while prefetch remained live')
    checks['required_progress_with_prefetch'] = True
    snapshot('after_migration')

    ring.release(current)
    actual_ple = copied_ple.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    if actual_ple != expected_ple:
        raise RuntimeError('PLE bytes changed under migration contention')
    ring.cancel_and_drain(prefetch)
    snapshot('after_cancel')
    checks['prefetch_cancel_reclaim'] = True
    budget.retire_request('joint-request')
    shared_requests.remove('joint-request')
    if budget.snapshot()['active_requests'] != 2 or shared_requests:
        raise RuntimeError('joint request did not retire after credit reclaim')
    checks['request_retirement'] = True

    adapter.synchronize()
    migration_cpu = adapter.destination.cpu()
    expected_migration = torch.empty_like(migration_cpu)
    expected_migration[0].fill_(8.0)
    expected_migration[1].fill_(12.0)
    if not torch.equal(migration_cpu, expected_migration):
        raise RuntimeError('synthetic migration destination content differs')
    migration_bytes = migration_cpu.contiguous().view(
        torch.uint8).numpy().tobytes()
    if (_digest_bytes(migration_bytes) !=
            joint_budget_gate.EXPECTED_MIGRATION_SHA256):
        raise RuntimeError('synthetic migration destination digest differs')
    checks['migration_h2d_exact'] = True
    ring.close()
    reader.close()
    snapshot('final')
    final = budget.snapshot()
    if (final['allocations'] or final['active_requests'] or final['in_flight'] or
            any(final['used'].values())):
        raise RuntimeError('joint budget credits did not return to zero')
    checks['zero_credits_after_close'] = True
    checks['no_capacity_exceeded'] = True
    if (joint_budget_gate.CONFIGURED_BUDGET['reserve']['h2d_bytes'] !=
            joint_budget_gate.RESIDENT_CAPACITY * joint_budget_gate.EXPERT_BYTES):
        raise RuntimeError('full Q32 H2D reserve arithmetic differs')
    checks['full_q32_budget_math'] = True

    peak = int(torch.npu.max_memory_allocated()) - baseline
    document = dict(
        schema=1, status='PASS_JOINT_BUDGET_SYNTHETIC_CONTENTION',
        scope='SYNTHETIC_Q32_TRANSPORT_CREDITS_NOT_EXPERT_MATH_OR_MODEL',
        physical_device=15, source_identity=identity,
        source_sha256=source_sha, ple_gate_sha256=args.ple_gate_sha256,
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),
        expert_bytes=joint_budget_gate.EXPERT_BYTES,
        resident_capacity=joint_budget_gate.RESIDENT_CAPACITY,
        migration_probe_experts=joint_budget_gate.MIGRATION_PROBE_EXPERTS,
        configured_budget=joint_budget_gate.CONFIGURED_BUDGET,
        checks=checks, snapshots=snapshots,
        case=dict(
            ple_rows=512, ple_row_width=160,
            ple_payload_bytes=joint_budget_gate.PLE_SLOT_BYTES,
            migration_experts=len(moves),
            actual_migration_h2d_bytes=adapter.copied,
            full_q32_h2d_bytes=(joint_budget_gate.RESIDENT_CAPACITY *
                                joint_budget_gate.EXPERT_BYTES),
            ple_sha256=_digest_bytes(actual_ple),
            migration_sha256=joint_budget_gate.EXPECTED_MIGRATION_SHA256),
        peak_extra_bytes=peak)
    raw = (json.dumps(document, sort_keys=True, indent=2) + '\n').encode()
    temporary = args.output.with_suffix('.pending.json')
    temporary.write_bytes(raw)
    joint_budget_gate.require_joint_budget_device_gate(
        temporary, hashlib.sha256(raw).hexdigest(),
        source_identity=identity, source_sha256=source_sha,
        ple_gate_sha256=args.ple_gate_sha256,
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__))
    temporary.replace(args.output)
    print(document['status'], flush=True)


if __name__ == '__main__':
    main()
