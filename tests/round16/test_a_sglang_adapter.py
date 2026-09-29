from __future__ import annotations

from dataclasses import dataclass
import unittest

from round16.a.provider_manifest import OPERATION_GDN, OPERATION_QSA
from round16.a.sglang_adapter import (
    BackendResult,
    ImmediateCompletion,
    KTAscendProvider,
    MetadataContractError,
    adapt_metadata,
)
from round16.common.contracts import (
    CONTRACT_VERSION,
    Phase,
    ProviderResult,
    ProviderSpec,
    StateLease,
    SupportDecision,
    TensorSpec,
)


def _lease(*, writable_state=("gdn.recurrent",)) -> StateLease:
    return StateLease(
        request_id="request-7",
        request_epoch=3,
        allocation_generation=4,
        batch_generation=5,
        layer_id=6,
        token_range=(8, 10),
        completion_id="completion-9",
        writable_state=writable_state,
    )


def _spec(operation: str = OPERATION_GDN) -> ProviderSpec:
    return ProviderSpec(
        operation=operation,
        phase=Phase.DECODE,
        inputs=(
            TensorSpec(
                "hidden_states",
                (2, 2048),
                "bfloat16",
                "ND",
                stride=(2048, 1),
                device_type="mock",
            ),
        ),
        state_format="gdn-state-v1" if operation == OPERATION_GDN else "qsa-index-v1",
        stream_kind="mock-current-stream",
        metadata_version=CONTRACT_VERSION,
    )


def _spec_with_tensor(tensor: TensorSpec) -> ProviderSpec:
    base = _spec()
    return ProviderSpec(
        operation=base.operation,
        phase=base.phase,
        inputs=(tensor,),
        state_format=base.state_format,
        stream_kind=base.stream_kind,
        metadata_version=base.metadata_version,
    )


def _metadata(spec: ProviderSpec | None = None) -> dict[str, object]:
    lease = _lease()
    return {
        "provider_spec": spec or _spec(),
        "runtime_stream_kind": (spec or _spec()).stream_kind,
        "request_id": lease.request_id,
        "request_epoch": lease.request_epoch,
        "allocation_generation": lease.allocation_generation,
        "batch_generation": lease.batch_generation,
        "layer_id": lease.layer_id,
        "token_range": lease.token_range,
        "completion_id": lease.completion_id,
    }


@dataclass(frozen=True)
class FakeTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    stride: tuple[int, ...] | None
    device_type: str


@dataclass(frozen=True)
class _Device:
    type: str


class TorchLikeTensor:
    """No logical name or device_type attribute, like a standard tensor."""

    shape = (2, 2048)
    dtype = "torch.bfloat16"
    layout = "torch.strided"
    device = _Device("mock")

    def stride(self) -> tuple[int, int]:
        return (2048, 1)


def _fake_tensor(spec: ProviderSpec | None = None, **changes: object) -> FakeTensor:
    tensor = (spec or _spec()).inputs[0]
    values: dict[str, object] = {
        "name": tensor.name,
        "shape": tensor.shape,
        "dtype": tensor.dtype,
        "layout": tensor.layout,
        "stride": tensor.stride,
        "device_type": tensor.device_type,
    }
    values.update(changes)
    return FakeTensor(**values)  # type: ignore[arg-type]


class _Backend:
    def __init__(
        self,
        events: list[str],
        *,
        aliases_input: bool = False,
        return_input: bool = False,
        return_member: bool = False,
        pop_member: bool = False,
    ) -> None:
        self.events = events
        self.aliases_input = aliases_input
        self.return_input = return_input
        self.return_member = return_member
        self.pop_member = pop_member
        self.close_calls = 0

    def run(self, operation, inputs, metadata, state_lease):
        self.events.append("backend")
        if self.pop_member:
            if isinstance(inputs, dict):
                output = inputs.pop("hidden_states")
            else:
                output = inputs.pop(0)
        elif self.return_member:
            output = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs[0]
        else:
            output = inputs if self.return_input else {"kind": operation}
        return BackendResult(
            output=output,
            completion=ImmediateCompletion(state_lease.completion_id),
            state_writes=("gdn.recurrent",),
            aliases_input=self.aliases_input,
        )

    def close(self) -> None:
        self.close_calls += 1


class _Fallback:
    def __init__(
        self,
        events: list[str],
        *,
        return_input: bool = False,
        return_member: bool = False,
        pop_member: bool = False,
    ) -> None:
        self.events = events
        self.return_input = return_input
        self.return_member = return_member
        self.pop_member = pop_member
        self.close_calls = 0

    def supports(self, spec: ProviderSpec) -> SupportDecision:
        return SupportDecision.accept()

    def run(self, inputs, metadata, state_lease):
        self.events.append("fallback")
        if self.pop_member:
            if isinstance(inputs, dict):
                output = inputs.pop("hidden_states")
            else:
                output = inputs.pop(0)
        elif self.return_member:
            output = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs[0]
        else:
            output = inputs if self.return_input else "eager"
        return ProviderResult(
            output=output,
            completion=ImmediateCompletion(state_lease.completion_id),
            lease=state_lease,
        )

    def close(self) -> None:
        self.close_calls += 1


class _FailOnceCloseBackend(_Backend):
    def close(self) -> None:
        self.close_calls += 1
        self.events.append(f"backend-close-{self.close_calls}")
        if self.close_calls == 1:
            raise ValueError("backend close failure")


class _CloseRecordingFallback(_Fallback):
    def close(self) -> None:
        self.close_calls += 1
        self.events.append("fallback-close")


@dataclass
class _ObjectMetadata:
    provider_spec: ProviderSpec
    runtime_stream_kind: str
    request_id: str
    request_epoch: int
    allocation_generation: int
    batch_generation: int
    layer_id: int
    token_range: tuple[int, int]
    completion_id: str


class SGLangAdapterTests(unittest.TestCase):
    def test_metadata_adapter_accepts_mapping_or_attributes_and_freezes_lease_identity(self) -> None:
        lease = _lease()
        runtime_input = _fake_tensor()
        mapping_view = adapt_metadata(_metadata(), lease, runtime_input)
        object_view = adapt_metadata(
            _ObjectMetadata(**_metadata()), lease, runtime_input
        )

        self.assertEqual(mapping_view.lease_identity, lease.identity)
        self.assertEqual(object_view.lease_identity, lease.identity)
        self.assertEqual(mapping_view.provider_spec, _spec())
        self.assertEqual(mapping_view.observed_inputs, _spec().inputs)
        self.assertEqual(mapping_view.runtime_stream_kind, "mock-current-stream")

    def test_metadata_adapter_freezes_mapping_shape_and_stride_lists(self) -> None:
        shape = [2, 2048]
        stride = [2048, 1]
        spec_mapping = {
            "operation": OPERATION_GDN,
            "phase": "decode",
            "inputs": [
                {
                    "name": "hidden_states",
                    "shape": shape,
                    "dtype": "bfloat16",
                    "layout": "ND",
                    "stride": stride,
                    "device_type": "mock",
                }
            ],
            "state_format": "gdn-state-v1",
            "stream_kind": "mock-current-stream",
            "metadata_version": CONTRACT_VERSION,
        }
        metadata = _metadata()
        metadata["provider_spec"] = spec_mapping

        adapted = adapt_metadata(metadata, _lease(), _fake_tensor())
        shape[1] = 4096
        stride[0] = 1

        self.assertEqual(adapted.provider_spec.inputs[0].shape, (2, 2048))
        self.assertEqual(adapted.provider_spec.inputs[0].stride, (2048, 1))

    def test_metadata_mismatch_fails_before_backend_or_fallback(self) -> None:
        events: list[str] = []
        provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))
        metadata = _metadata()
        metadata["batch_generation"] = 99

        with self.assertRaisesRegex(MetadataContractError, "batch_generation"):
            provider.run(_fake_tensor(), metadata, _lease())

        self.assertEqual(events, [])

    def test_unsupported_spec_falls_back_before_any_backend_state_write(self) -> None:
        events: list[str] = []
        provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))
        unsupported = ProviderSpec(
            operation=OPERATION_QSA,
            phase=Phase.PREFILL,
            inputs=_spec(OPERATION_QSA).inputs,
            state_format="qsa-index-v1",
            stream_kind="mock-current-stream",
            metadata_version=CONTRACT_VERSION,
        )

        result = provider.run(
            _fake_tensor(unsupported), _metadata(unsupported), _lease()
        )

        self.assertEqual(result.output, "eager")
        self.assertEqual(events, ["fallback"])
        self.assertEqual(result.lease, _lease())

    def test_unqualified_shapes_and_stride_fallback_without_calling_backend(self) -> None:
        invalid_tensors = (
            TensorSpec("hidden_states", (0, 2048), "bfloat16", "ND", device_type="mock"),
            TensorSpec("hidden_states", (129, 2048), "bfloat16", "ND", device_type="mock"),
            TensorSpec("hidden_states", (1, 4096), "bfloat16", "ND", device_type="mock"),
            TensorSpec("hidden_states", (1, 1, 2048), "bfloat16", "ND", device_type="mock"),
            TensorSpec(
                "hidden_states",
                (1, 2048),
                "bfloat16",
                "ND",
                stride=(1, 128),
                device_type="mock",
            ),
        )
        for tensor in invalid_tensors:
            events: list[str] = []
            provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))
            with self.subTest(shape=tensor.shape, stride=tensor.stride):
                result = provider.run(
                    _fake_tensor(_spec_with_tensor(tensor)),
                    _metadata(_spec_with_tensor(tensor)),
                    _lease(),
                )
                self.assertEqual(result.output, "eager")
                self.assertEqual(events, ["fallback"])

    def test_backend_result_preserves_completion_lease_and_state_write_contract(self) -> None:
        events: list[str] = []
        lease = _lease()
        provider = KTAscendProvider(_Backend(events))

        result = provider.run(_fake_tensor(), _metadata(), lease)

        self.assertEqual(events, ["backend"])
        self.assertEqual(result.lease.identity, lease.identity)
        self.assertEqual(result.completion.completion_id, lease.completion_id)
        self.assertTrue(result.completion.done)
        self.assertEqual(result.state_writes, ("gdn.recurrent",))

    def test_backend_cannot_alias_output_to_input(self) -> None:
        provider = KTAscendProvider(_Backend([], aliases_input=True))

        with self.assertRaisesRegex(RuntimeError, "alias"):
            provider.run(_fake_tensor(), _metadata(), _lease())

    def test_provider_detects_identity_alias_even_if_backend_does_not_declare_it(self) -> None:
        shared_input = _fake_tensor()
        provider = KTAscendProvider(_Backend([], return_input=True))

        with self.assertRaisesRegex(RuntimeError, "alias"):
            provider.run(shared_input, _metadata(), _lease())

    def test_provider_rejects_fallback_output_alias(self) -> None:
        fallback = _Fallback([], return_input=True)
        provider = KTAscendProvider(_Backend([]), fallback=fallback)
        unsupported = ProviderSpec(
            operation=OPERATION_QSA,
            phase=Phase.PREFILL,
            inputs=_spec(OPERATION_QSA).inputs,
            state_format="qsa-index-v1",
            stream_kind="mock-current-stream",
            metadata_version=CONTRACT_VERSION,
        )
        shared_input = _fake_tensor(unsupported)

        with self.assertRaisesRegex(RuntimeError, "alias"):
            provider.run(shared_input, _metadata(unsupported), _lease())

    def test_backend_rejects_mapping_and_sequence_member_aliases(self) -> None:
        for label, runtime_inputs in (
            ("mapping", {"hidden_states": _fake_tensor()}),
            ("sequence", [_fake_tensor()]),
        ):
            events: list[str] = []
            provider = KTAscendProvider(_Backend(events, return_member=True))
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "alias"):
                    provider.run(runtime_inputs, _metadata(), _lease())
                self.assertEqual(events, ["backend"])

    def test_fallback_rejects_mapping_and_sequence_member_aliases(self) -> None:
        unsupported = ProviderSpec(
            operation=OPERATION_QSA,
            phase=Phase.PREFILL,
            inputs=_spec(OPERATION_QSA).inputs,
            state_format="qsa-index-v1",
            stream_kind="mock-current-stream",
            metadata_version=CONTRACT_VERSION,
        )
        for label, runtime_inputs in (
            ("mapping", {"hidden_states": _fake_tensor(unsupported)}),
            ("sequence", [_fake_tensor(unsupported)]),
        ):
            events: list[str] = []
            provider = KTAscendProvider(
                _Backend(events), fallback=_Fallback(events, return_member=True)
            )
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "alias"):
                    provider.run(runtime_inputs, _metadata(unsupported), _lease())
                self.assertEqual(events, ["fallback"])

    def test_backend_alias_check_uses_pre_execution_input_snapshot(self) -> None:
        for label, runtime_inputs in (
            ("mapping", {"hidden_states": _fake_tensor()}),
            ("sequence", [_fake_tensor()]),
        ):
            events: list[str] = []
            provider = KTAscendProvider(_Backend(events, pop_member=True))
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "alias"):
                    provider.run(runtime_inputs, _metadata(), _lease())
                self.assertEqual(events, ["backend"])
                self.assertEqual(len(runtime_inputs), 0)

    def test_fallback_alias_check_uses_pre_execution_input_snapshot(self) -> None:
        unsupported = ProviderSpec(
            operation=OPERATION_QSA,
            phase=Phase.PREFILL,
            inputs=_spec(OPERATION_QSA).inputs,
            state_format="qsa-index-v1",
            stream_kind="mock-current-stream",
            metadata_version=CONTRACT_VERSION,
        )
        for label, runtime_inputs in (
            ("mapping", {"hidden_states": _fake_tensor(unsupported)}),
            ("sequence", [_fake_tensor(unsupported)]),
        ):
            events: list[str] = []
            provider = KTAscendProvider(
                _Backend(events), fallback=_Fallback(events, pop_member=True)
            )
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "alias"):
                    provider.run(runtime_inputs, _metadata(unsupported), _lease())
                self.assertEqual(events, ["fallback"])
                self.assertEqual(len(runtime_inputs), 0)

    def test_forged_qualified_spec_cannot_hide_bad_runtime_inputs(self) -> None:
        cases = (
            ("input count", (), "input count"),
            ("name", {"query": _fake_tensor()}, "name='query'"),
            ("shape", _fake_tensor(shape=(1, 4096)), "shape"),
            ("dtype", _fake_tensor(dtype="float32"), "dtype"),
            ("layout", _fake_tensor(layout="TND"), "layout"),
            ("stride", _fake_tensor(stride=(1, 2048)), "stride"),
            ("device_type", _fake_tensor(device_type="npu"), "device_type"),
        )
        for label, runtime_inputs, field in cases:
            events: list[str] = []
            provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))
            with self.subTest(label=label):
                with self.assertRaisesRegex(MetadataContractError, field):
                    provider.run(runtime_inputs, _metadata(), _lease())
                self.assertEqual(events, [])

    def test_runtime_dimensions_require_exact_ints_before_backend_or_fallback(self) -> None:
        one_row_spec = _spec_with_tensor(
            TensorSpec(
                "hidden_states",
                (1, 2048),
                "bfloat16",
                "ND",
                stride=(2048, 1),
                device_type="mock",
            )
        )
        cases = (
            ("shape float", _spec(), _fake_tensor(shape=(2.9, 2048)), "shape"),
            ("stride float", _spec(), _fake_tensor(stride=(2048.9, 1)), "stride"),
            (
                "bool dimension",
                one_row_spec,
                _fake_tensor(one_row_spec, shape=(True, 2048)),
                "shape",
            ),
        )
        for label, spec, runtime_input, field in cases:
            events: list[str] = []
            provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))
            with self.subTest(label=label):
                with self.assertRaisesRegex(MetadataContractError, field):
                    provider.run(runtime_input, _metadata(spec), _lease())
                self.assertEqual(events, [])

    def test_mapping_key_names_torch_like_tensor_without_name(self) -> None:
        events: list[str] = []
        provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))

        result = provider.run(
            {"hidden_states": TorchLikeTensor()},
            _metadata(),
            _lease(),
        )

        self.assertEqual(result.output, {"kind": OPERATION_GDN})
        self.assertEqual(events, ["backend"])

    def test_wrong_mapping_key_rejects_before_backend_or_fallback(self) -> None:
        events: list[str] = []
        provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))

        with self.assertRaisesRegex(MetadataContractError, "name='query'"):
            provider.run({"query": TorchLikeTensor()}, _metadata(), _lease())

        self.assertEqual(events, [])

    def test_sequence_is_positional_for_torch_like_tensor_without_name(self) -> None:
        events: list[str] = []
        provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))

        result = provider.run([TorchLikeTensor()], _metadata(), _lease())

        self.assertEqual(result.output, {"kind": OPERATION_GDN})
        self.assertEqual(events, ["backend"])

    def test_runtime_stream_mismatch_rejects_before_backend_or_fallback(self) -> None:
        events: list[str] = []
        provider = KTAscendProvider(_Backend(events), fallback=_Fallback(events))
        metadata = _metadata()
        metadata["runtime_stream_kind"] = "forged-stream"

        with self.assertRaisesRegex(MetadataContractError, "runtime_stream_kind"):
            provider.run(_fake_tensor(), metadata, _lease())

        self.assertEqual(events, [])

    def test_close_is_idempotent_and_closed_provider_rejects_new_work(self) -> None:
        backend = _Backend([])
        fallback = _Fallback([])
        provider = KTAscendProvider(backend, fallback=fallback)

        provider.close()
        provider.close()

        self.assertEqual(backend.close_calls, 1)
        self.assertEqual(fallback.close_calls, 1)
        self.assertEqual(provider.supports(_spec()).reason_code, "PROVIDER_CLOSED")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            provider.run(_fake_tensor(), _metadata(), _lease())

    def test_close_continues_after_failure_and_retries_only_failed_resource(self) -> None:
        events: list[str] = []
        backend = _FailOnceCloseBackend(events)
        fallback = _CloseRecordingFallback(events)
        provider = KTAscendProvider(backend, fallback=fallback)

        with self.assertRaisesRegex(RuntimeError, "failed to close") as raised:
            provider.close()

        self.assertEqual(events, ["backend-close-1", "fallback-close"])
        self.assertEqual(len(raised.exception.errors), 1)  # type: ignore[attr-defined]
        self.assertEqual(provider.supports(_spec()).reason_code, "PROVIDER_CLOSED")

        provider.close()
        provider.close()

        self.assertEqual(events, ["backend-close-1", "fallback-close", "backend-close-2"])
        self.assertEqual(backend.close_calls, 2)
        self.assertEqual(fallback.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
