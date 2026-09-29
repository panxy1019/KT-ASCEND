from __future__ import annotations

import unittest

from round16.common.resource_target import (
    NpuTarget,
    parse_device_claims,
    parse_npu_smi,
    select_safe_target,
)


NPU_SMI = """
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip  Phy-ID              | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
| 5     Ascend910           | OK            | 274.2                59                      0    / 0                |
| 0     10                  | 0000:89:00.0  | 32                   0    / 0                9406 / 65536            |
| 5     Ascend910           | OK            | -                    59                      0    / 0                |
| 1     11                  | 0000:8B:00.0  | 60                   0    / 0                63331/ 65536            |
| 6     Ascend910           | OK            | 275.2                59                      0    / 0                |
| 0     12                  | 0000:85:00.0  | 29                   0    / 0                9409 / 65536            |
| 6     Ascend910           | OK            | -                    61                      0    / 0                |
| 1     13                  | 0000:87:00.0  | 56                   0    / 0                63330/ 65536            |
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
| 5       0                 | 2980234       | python3.10         | 6124                  | NA                      |
| 5       1                 | 2980235       | python3.10         | 6124                  | NA                      |
| 5       1                 | 3476439       | VLLMEngineCor      | 54415                 | NA                      |
| 6       0                 | 2980236       | python3.10         | 6123                  | NA                      |
| 6       1                 | 2980237       | python3.10         | 6123                  | NA                      |
| 6       1                 | 3476477       | VLLMEngineCor      | 54414                 | NA                      |
"""


class ResourceTargetTests(unittest.TestCase):
    def test_parse_npu_smi_preserves_physical_card_chip_bus_and_processes(self) -> None:
        devices = parse_npu_smi(NPU_SMI)

        self.assertEqual(tuple(devices), (10, 11, 12, 13))
        self.assertEqual(devices[12].card_id, 6)
        self.assertEqual(devices[12].chip_id, 0)
        self.assertEqual(devices[12].pci_bdf, "0000:85:00.0")
        self.assertEqual(devices[12].hbm_used_mib, 9409)
        self.assertEqual(devices[12].hbm_total_mib, 65536)
        self.assertEqual(devices[12].processes[0].pid, 2980236)
        self.assertEqual(devices[13].processes[1].name, "VLLMEngineCor")

    def test_parse_claims_rejects_duplicate_or_out_of_range_identity(self) -> None:
        claims = parse_device_claims(
            "/qwen38-vllm-12|ASCEND_RT_VISIBLE_DEVICES=12\n"
            "/qwen38-vllm-13|ASCEND_RT_VISIBLE_DEVICES=13\n"
        )
        self.assertEqual(claims, {12: "/qwen38-vllm-12", 13: "/qwen38-vllm-13"})

        with self.assertRaisesRegex(ValueError, "duplicate device claim"):
            parse_device_claims(
                "/first|ASCEND_RT_VISIBLE_DEVICES=12\n"
                "/second|ASCEND_RT_VISIBLE_DEVICES=12\n"
            )
        with self.assertRaisesRegex(ValueError, "physical device"):
            parse_device_claims("/bad|ASCEND_RT_VISIBLE_DEVICES=16\n")

    def test_select_target_fails_closed_for_container_claim_even_without_engine(self) -> None:
        devices = parse_npu_smi(NPU_SMI)
        decision = select_safe_target(
            devices,
            claims={12: "/restarting-external"},
            allowed_training_pids={2980236},
            candidates=(
                NpuTarget(12, 6, 0, "0000:85:00.0", (2, 3, 4, 7)),
            ),
        )

        self.assertFalse(decision.ready)
        self.assertIsNone(decision.target)
        self.assertEqual(decision.rejections[12], ("EXTERNAL_CONTAINER_CLAIM",))

    def test_selects_first_explicit_candidate_with_only_registered_training(self) -> None:
        devices = parse_npu_smi(NPU_SMI)
        decision = select_safe_target(
            devices,
            claims={},
            allowed_training_pids={2980234, 2980236},
            candidates=(
                NpuTarget(12, 6, 0, "0000:85:00.0", (2, 3, 4, 7)),
                NpuTarget(10, 5, 0, "0000:89:00.0", (0, 1, 2, 3)),
            ),
        )

        self.assertTrue(decision.ready)
        self.assertEqual(decision.target.physical_id, 12)
        self.assertEqual(decision.target.numa_nodes, (2, 3, 4, 7))
        self.assertEqual(decision.rejections, {})

    def test_unknown_process_or_identity_drift_rejects_candidate(self) -> None:
        devices = parse_npu_smi(NPU_SMI)
        decision = select_safe_target(
            devices,
            claims={},
            allowed_training_pids=set(),
            candidates=(
                NpuTarget(13, 6, 0, "0000:87:00.0", (2, 3, 4, 7)),
                NpuTarget(11, 5, 1, "0000:8B:00.0", (0, 1, 2, 3)),
            ),
        )

        self.assertFalse(decision.ready)
        self.assertEqual(
            decision.rejections[13],
            ("CARD_CHIP_OR_BUS_IDENTITY_MISMATCH", "UNKNOWN_DEVICE_PROCESS"),
        )
        self.assertEqual(decision.rejections[11], ("UNKNOWN_DEVICE_PROCESS",))

    def test_target_contract_rejects_implicit_or_malformed_topology(self) -> None:
        with self.assertRaisesRegex(ValueError, "four distinct NUMA"):
            NpuTarget(12, 6, 0, "0000:85:00.0", (2, 2, 4, 7))
        with self.assertRaisesRegex(ValueError, "PCI BDF"):
            NpuTarget(12, 6, 0, "85:00.0", (2, 3, 4, 7))
        with self.assertRaisesRegex(ValueError, "physical device"):
            NpuTarget(16, 8, 0, "0000:81:00.0", (2, 3, 4, 7))


if __name__ == "__main__":
    unittest.main()
