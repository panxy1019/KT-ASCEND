import hashlib
import json
import copy
from pathlib import Path
import tempfile
import unittest

from migration_bridge.round9.runtime import joint_budget_gate as gate


class JointBudgetGateTests(unittest.TestCase):
    def document(self):
        snapshots = []
        for stage in gate.SNAPSHOT_STAGES:
            snapshots.append(dict(stage=stage, **copy.deepcopy(
                gate.EXPECTED_SNAPSHOTS[stage])))
        return dict(
            schema=1,
            status='PASS_JOINT_BUDGET_SYNTHETIC_CONTENTION',
            scope='SYNTHETIC_Q32_TRANSPORT_CREDITS_NOT_EXPERT_MATH_OR_MODEL',
            physical_device=15,
            source_identity={'model': 'frozen'},
            source_sha256={name: 'a' * 64 for name in gate.SOURCE_FILES},
            ple_gate_sha256='b' * 64,
            torch_version='2.test', torch_npu_version='2.test',
            expert_bytes=9830400, resident_capacity=32,
            migration_probe_experts=2,
            configured_budget=copy.deepcopy(gate.CONFIGURED_BUDGET),
            checks={name: True for name in gate.CHECKS},
            snapshots=snapshots,
            case=dict(ple_rows=512, ple_row_width=160,
                      ple_payload_bytes=163840, migration_experts=2,
                      actual_migration_h2d_bytes=19660800,
                      full_q32_h2d_bytes=314572800,
                      ple_sha256='c' * 64,
                      migration_sha256=gate.EXPECTED_MIGRATION_SHA256),
            peak_extra_bytes=gate.MIN_PEAK_EXTRA_BYTES,
        )

    def call(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gate.json'
            raw = (json.dumps(document, sort_keys=True) + '\n').encode()
            path.write_bytes(raw)
            return gate.require_joint_budget_device_gate(
                path, hashlib.sha256(raw).hexdigest(),
                source_identity={'model': 'frozen'},
                source_sha256={name: 'a' * 64 for name in gate.SOURCE_FILES},
                ple_gate_sha256='b' * 64,
                torch_version='2.test', torch_npu_version='2.test')

    def test_exact_source_geometry_budget_and_case_are_admitted(self):
        result = self.call(self.document())
        self.assertEqual(result['expert_bytes'], 2560 * 640 * 3 * 2)
        self.assertEqual(result['configured_budget']['capacity']['h2d_bytes'],
                         32 * result['expert_bytes'] + 2 * 512 * 160 * 2)
        self.assertEqual(result['case']['actual_migration_h2d_bytes'],
                         2 * result['expert_bytes'])

    def test_identity_checks_budget_snapshots_and_case_fail_closed(self):
        mutations = []
        changed = self.document(); changed['source_sha256']['host_budget.py'] = 'e' * 64
        mutations.append(changed)
        changed = self.document(); changed['checks']['prefetch_cancel_reclaim'] = False
        mutations.append(changed)
        changed = self.document(); changed['configured_budget']['reserve']['h2d_bytes'] -= 1
        mutations.append(changed)
        changed = self.document(); changed['case']['full_q32_h2d_bytes'] -= 1
        mutations.append(changed)
        changed = self.document(); changed['case']['migration_sha256'] = 'd' * 64
        mutations.append(changed)
        changed = self.document(); changed['snapshots'][0]['used']['host_bytes'] = \
            gate.CONFIGURED_BUDGET['capacity']['host_bytes'] + 1
        mutations.append(changed)
        changed = self.document()
        next(item for item in changed['snapshots']
             if item['stage'] == 'migration_live')['used']['h2d_bytes'] -= 1
        mutations.append(changed)
        changed = self.document()
        next(item for item in changed['snapshots'] if item['stage'] == 'final')['allocations'] = 1
        mutations.append(changed)
        for index, document in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                self.call(document)

    def test_hash_and_caller_contract_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gate.json'
            path.write_text(json.dumps(self.document()))
            kwargs = dict(source_identity={'model': 'frozen'},
                          source_sha256={name: 'a' * 64 for name in gate.SOURCE_FILES},
                          ple_gate_sha256='b' * 64,
                          torch_version='2.test', torch_npu_version='2.test')
            with self.assertRaises(RuntimeError):
                gate.require_joint_budget_device_gate(path, '0' * 64, **kwargs)
            with self.assertRaises(RuntimeError):
                gate.require_joint_budget_device_gate(
                    path, hashlib.sha256(path.read_bytes()).hexdigest(),
                    **(kwargs | {'ple_gate_sha256': None}))

    def test_probe_source_binds_real_component_paths_and_no_blanket_pass(self):
        source = (Path(__file__).with_name('joint_budget_device_probe.py')
                  .read_text())
        for required in (
                'budgeted_migration.BudgetedMigration(',
                'ple_async.AsyncPLERing(',
                'ring.wait(current, consumer_stream=consumer)',
                'self.destination[slot].copy_(payload, non_blocking=False)',
                "ring.cancel_and_drain(prefetch)",
                "snapshot('migration_live')",
                "snapshot('final')",
                'joint_budget_gate.require_joint_budget_device_gate('):
            self.assertIn(required, source)
        self.assertNotIn(
            '{name: True for name in joint_budget_gate.CHECKS}', source)


if __name__ == '__main__': unittest.main()
