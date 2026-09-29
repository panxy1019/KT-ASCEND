from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from round16.common.contracts import (
    CONTRACT_VERSION,
    Phase,
    ProviderResult,
    ProviderSpec,
    RouteBatch,
    StateLease,
    SupportDecision,
    TensorSpec,
)


class _Completion:
    def __init__(self, completion_id: str, *, done: bool = True) -> None:
        self.completion_id = completion_id
        self._done = done

    @property
    def done(self) -> bool:
        return self._done

    def wait(self) -> None:
        self._done = True


def _lease() -> StateLease:
    return StateLease(
        request_id="request-7",
        request_epoch=3,
        allocation_generation=4,
        batch_generation=5,
        layer_id=6,
        token_range=(8, 10),
        completion_id="completion-9",
        writable_state=("gdn.recurrent",),
    )


class CommonContractTests(unittest.TestCase):
    def test_contract_identity_and_specs_are_immutable(self) -> None:
        self.assertEqual(CONTRACT_VERSION, "round16-contract-v2")
        tensor = TensorSpec(
        name="hidden_states",
        shape=(2, 4096),
        dtype="bfloat16",
        layout="ND",
        stride=(4096, 1),
        device_type="npu",
    )
        spec = ProviderSpec(
        operation="qsa.prefill",
        phase=Phase.PREFILL,
        inputs=(tensor,),
        state_format="qsa-v1",
        stream_kind="ascend-current-stream",
        metadata_version="round14a-s0",
    )

        self.assertEqual(tensor.rank, 2)
        self.assertIs(spec.phase, Phase.PREFILL)
        with self.assertRaises(FrozenInstanceError):
            tensor.dtype = "float32"  # type: ignore[misc]


    def test_tensor_spec_rejects_invalid_shape_or_stride(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TensorSpec("x", (-1, 2), "bfloat16", "ND")
        with self.assertRaisesRegex(ValueError, "stride rank"):
            TensorSpec("x", (1, 2), "bfloat16", "ND", stride=(2,))

    def test_specs_deep_freeze_caller_owned_sequences(self) -> None:
        shape = [2, 2048]
        stride = [2048, 1]
        tensor = TensorSpec(
            "hidden_states", shape, "bfloat16", "ND", stride=stride
        )
        inputs = [tensor]
        spec = ProviderSpec(
            "qsa.prefill",
            Phase.PREFILL,
            inputs,
            "qsa-v1",
            "ascend-current-stream",
            "round14a-s0",
        )

        shape[0] = 999
        stride[0] = 1
        inputs.clear()

        self.assertEqual(tensor.shape, (2, 2048))
        self.assertEqual(tensor.stride, (2048, 1))
        self.assertEqual(spec.inputs, (tensor,))

    def test_specs_reject_bool_or_non_integer_dimensions(self) -> None:
        with self.assertRaisesRegex(TypeError, "shape dimensions"):
            TensorSpec("x", (True, 2), "bfloat16", "ND")
        with self.assertRaisesRegex(TypeError, "stride entries"):
            TensorSpec("x", (1, 2), "bfloat16", "ND", stride=(2, 1.5))

    def test_specs_require_string_identity_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "tensor name"):
            TensorSpec(["x"], (1,), "bfloat16", "ND")
        with self.assertRaisesRegex(TypeError, "dtype, layout, and device_type"):
            TensorSpec("x", (1,), ["bfloat16"], "ND")
        tensor = TensorSpec("x", (1,), "bfloat16", "ND")
        with self.assertRaisesRegex(TypeError, "provider operation"):
            ProviderSpec(
                ["qsa.prefill"],
                Phase.PREFILL,
                (tensor,),
                "qsa-v1",
                "stream",
                "v2",
            )


    def test_state_lease_freezes_full_generation_identity(self) -> None:
        lease = _lease()

        self.assertEqual(lease.identity, (
        "request-7",
        3,
        4,
        5,
        6,
        (8, 10),
        "completion-9",
        ))
        with self.assertRaisesRegex(ValueError, "token_range"):
            StateLease("r", 0, 0, 0, 0, (4, 3), "c")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            StateLease("r", 0, 0, 0, 0, (0, 1), "c", ("kv", "kv"))

    def test_state_lease_deep_freezes_ranges_and_permissions(self) -> None:
        token_range = [2, 4]
        writable_state = ["gdn.recurrent"]
        lease = StateLease(
            "request",
            1,
            2,
            3,
            4,
            token_range,
            "completion",
            writable_state,
        )

        token_range[0] = 99
        writable_state.append("injected")

        self.assertEqual(lease.token_range, (2, 4))
        self.assertEqual(lease.writable_state, ("gdn.recurrent",))
        with self.assertRaisesRegex(TypeError, "non-negative integers"):
            StateLease("r", True, 0, 0, 0, (0, 1), "c")

    def test_state_lease_requires_hashable_string_identity(self) -> None:
        with self.assertRaisesRegex(TypeError, "request_id and completion_id"):
            StateLease(["request"], 0, 0, 0, 0, (0, 1), "completion")
        with self.assertRaisesRegex(TypeError, "request_id and completion_id"):
            StateLease("request", 0, 0, 0, 0, (0, 1), ["completion"])


    def test_route_batch_copies_inputs_and_rejects_shape_drift(self) -> None:
        ids = [[1, 3], [2, -1]]
        weights = [[0.75, 0.25], [1.0, 0.0]]
        batch = RouteBatch.freeze(ids, weights, valid_rows=[True, False])
        ids[0][0] = 99

        self.assertEqual(batch.topk_ids, ((1, 3), (2, -1)))
        self.assertEqual(batch.topk_weights, ((0.75, 0.25), (1.0, 0.0)))
        self.assertEqual(batch.token_count, 2)
        self.assertEqual(batch.top_k, 2)
        with self.assertRaisesRegex(ValueError, "same row count"):
            RouteBatch.freeze([[1]], [[1.0], [1.0]])
        with self.assertRaisesRegex(ValueError, "consistent top-k"):
            RouteBatch.freeze([[1, 2], [3]], [[0.5, 0.5], [1.0]])

    def test_route_batch_direct_constructor_deep_freezes_inputs(self) -> None:
        ids = [[1, 2]]
        weights = [[0.6, 0.4]]
        valid = [True]
        batch = RouteBatch(ids, weights, valid)

        ids[0][0] = 99
        weights[0][0] = float("nan")
        valid[0] = False

        self.assertEqual(batch.topk_ids, ((1, 2),))
        self.assertEqual(batch.topk_weights, ((0.6, 0.4),))
        self.assertEqual(batch.valid_rows, (True,))

        with self.assertRaisesRegex(TypeError, "expert ids"):
            RouteBatch(((True,),), ((1.0,),), (True,))
        with self.assertRaisesRegex(TypeError, "valid_rows"):
            RouteBatch(((1,),), ((1.0,),), (1,))


    def test_support_decision_makes_device_pending_explicit(self) -> None:
        decision = SupportDecision.device_required("torch_npu import needs device image")

        self.assertFalse(decision.supported)
        self.assertTrue(decision.requires_device)
        self.assertTrue(decision.fallback_allowed)
        self.assertEqual(decision.reason_code, "DEVICE_REQUIRED_NOT_RUN")


    def test_provider_result_rejects_wrong_completion_or_state_write(self) -> None:
        lease = _lease()
        completion = _Completion(lease.completion_id)
        result = ProviderResult(
        output="result",
        completion=completion,
        lease=lease,
        state_writes=("gdn.recurrent",),
    )

        self.assertTrue(result.completion.done)
        with self.assertRaisesRegex(ValueError, "completion_id"):
            ProviderResult("x", _Completion("late"), lease)
        with self.assertRaisesRegex(ValueError, "not leased"):
            ProviderResult("x", completion, lease, state_writes=("qsa.index",))

    def test_provider_result_freezes_state_write_sequence(self) -> None:
        lease = _lease()
        writes = ["gdn.recurrent"]
        result = ProviderResult("x", _Completion(lease.completion_id), lease, writes)

        writes.append("injected")

        self.assertEqual(result.state_writes, ("gdn.recurrent",))

    def test_provider_result_requires_complete_completion_protocol(self) -> None:
        class _IncompleteCompletion:
            completion_id = "completion-9"

        with self.assertRaisesRegex(TypeError, "CompletionBoundary"):
            ProviderResult("x", _IncompleteCompletion(), _lease())
        with self.assertRaisesRegex(TypeError, "StateLease"):
            ProviderResult("x", _Completion("completion-9"), object())


if __name__ == "__main__":
    unittest.main()
