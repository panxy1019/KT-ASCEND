from __future__ import annotations

import unittest

from round16.a.provider_manifest import (
    OPERATION_GDN,
    OPERATION_MOE,
    OPERATION_QSA,
    dependency_tiers,
    manifest_payload,
)
from round16.a.sglang_adapter import KTAscendProvider
from round16.common.contracts import CONTRACT_VERSION, Phase, ProviderSpec, TensorSpec


class _Backend:
    def run(self, operation, inputs, metadata, state_lease):  # pragma: no cover
        raise AssertionError("support checks must not execute the backend")

    def close(self) -> None:
        pass


def _spec(
    operation: str,
    *,
    phase: Phase = Phase.DECODE,
    dtype: str = "bfloat16",
    device_type: str = "npu",
    state_format: str | None = None,
    name: str = "hidden_states",
    shape: tuple[int, ...] = (1, 2048),
    layout: str = "ND",
    stride: tuple[int, ...] | None = None,
) -> ProviderSpec:
    formats = {
        OPERATION_QSA: "qsa-index-v1",
        OPERATION_GDN: "gdn-state-v1",
        OPERATION_MOE: "resident-moe-v1",
    }
    return ProviderSpec(
        operation=operation,
        phase=phase,
        inputs=(TensorSpec(name, shape, dtype, layout, stride=stride, device_type=device_type),),
        state_format=state_format or formats[operation],
        stream_kind="ascend-current-stream",
        metadata_version=CONTRACT_VERSION,
    )


class ProviderManifestTests(unittest.TestCase):
    def test_manifest_declares_three_provider_boundaries_and_dependency_tiers(self) -> None:
        payload = manifest_payload()

        self.assertEqual(payload["contract_version"], CONTRACT_VERSION)
        self.assertEqual(
            {entry["operation"] for entry in payload["operations"]},
            {OPERATION_QSA, OPERATION_GDN, OPERATION_MOE},
        )
        self.assertEqual(set(dependency_tiers()), {"P0", "P1", "P2"})
        self.assertEqual(payload["device_validation"], "DEVICE_REQUIRED_NOT_RUN")
        self.assertEqual(payload["performance_validation"], "NOT_RUN")
        self.assertTrue(all("fallback" in entry for entry in payload["operations"]))
        self.assertTrue(all("graph_capability" in entry for entry in payload["operations"]))

    def test_supports_rejects_wrong_contract_without_touching_backend(self) -> None:
        provider = KTAscendProvider(_Backend())

        decision = provider.supports(_spec(OPERATION_QSA, dtype="float32"))

        self.assertFalse(decision.supported)
        self.assertEqual(decision.reason_code, "UNSUPPORTED_DTYPE")
        self.assertTrue(decision.fallback_allowed)

    def test_supports_reports_device_pending_when_no_backend_is_injected(self) -> None:
        provider = KTAscendProvider()

        decision = provider.supports(_spec(OPERATION_GDN, phase=Phase.PREFILL))

        self.assertFalse(decision.supported)
        self.assertEqual(decision.reason_code, "DEVICE_REQUIRED_NOT_RUN")
        self.assertTrue(decision.requires_device)

    def test_injected_backend_accepts_declared_boundaries_but_qsa_is_decode_only(self) -> None:
        provider = KTAscendProvider(_Backend())

        self.assertTrue(provider.supports(_spec(OPERATION_QSA)).supported)
        self.assertTrue(
            provider.supports(_spec(OPERATION_GDN, phase=Phase.PREFILL)).supported
        )
        self.assertTrue(provider.supports(_spec(OPERATION_MOE)).supported)
        rejected = provider.supports(_spec(OPERATION_QSA, phase=Phase.PREFILL))
        self.assertEqual(rejected.reason_code, "UNSUPPORTED_PHASE")

    def test_supports_accepts_only_the_resolved_hidden_state_shape_envelope(self) -> None:
        provider = KTAscendProvider(_Backend())
        cases = (
            ("zero rows", {"shape": (0, 2048)}, "UNSUPPORTED_SHAPE"),
            ("too many rows", {"shape": (129, 2048)}, "UNSUPPORTED_SHAPE"),
            ("width drift", {"shape": (1, 4096)}, "UNSUPPORTED_SHAPE"),
            ("wrong rank", {"shape": (1, 1, 2048)}, "UNSUPPORTED_RANK"),
            ("wrong stride", {"stride": (1, 128)}, "UNSUPPORTED_STRIDE"),
            ("unresolved layout", {"layout": "TND"}, "UNSUPPORTED_LAYOUT"),
            ("wrong input", {"name": "query"}, "UNSUPPORTED_INPUT_NAME"),
        )

        self.assertTrue(
            provider.supports(
                _spec(OPERATION_GDN, stride=(2048, 1), phase=Phase.PREFILL)
            ).supported
        )
        for label, changes, reason in cases:
            with self.subTest(label=label):
                decision = provider.supports(_spec(OPERATION_GDN, **changes))
                self.assertFalse(decision.supported)
                self.assertEqual(decision.reason_code, reason)
                self.assertTrue(decision.fallback_allowed)


if __name__ == "__main__":
    unittest.main()
