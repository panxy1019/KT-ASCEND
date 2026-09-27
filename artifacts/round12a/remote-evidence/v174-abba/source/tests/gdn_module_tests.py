"""Opt-in CPU tests for the stateless eager GDN reference."""

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from runtime.gdn import (  # noqa: E402
    GDNConfig,
    GDNState,
    eager_gdn,
    gated_delta_recurrence,
    sigmoid_gated_rms_norm,
    zero_state,
)


class GDNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(922)
        self.config = GDNConfig(
            num_key_heads=2,
            num_value_heads=6,
            key_head_dim=3,
            value_head_dim=2,
            conv_kernel_size=4,
            eps=1e-6,
        )

    def make_inputs(self, length=9, dtype=torch.float32):
        cfg = self.config
        conv_dim = (
            2 * cfg.num_key_heads * cfg.key_head_dim
            + cfg.num_value_heads * cfg.value_head_dim
        )
        return {
            "mixed_qkv": torch.randn(length, conv_dim, dtype=dtype),
            "a": torch.randn(length, cfg.num_value_heads, dtype=dtype),
            "b": torch.randn(length, cfg.num_value_heads, dtype=dtype),
            "z": torch.randn(
                length, cfg.num_value_heads, cfg.value_head_dim, dtype=dtype
            ),
            "conv_weight": torch.randn(
                conv_dim, cfg.conv_kernel_size, dtype=dtype
            ),
            "A_log": torch.randn(cfg.num_value_heads, dtype=torch.float32),
            "dt_bias": torch.randn(cfg.num_value_heads, dtype=torch.float32),
            "norm_weight": torch.randn(cfg.value_head_dim, dtype=dtype),
        }

    def run_gdn(self, tensors, state=None):
        return eager_gdn(
            tensors["mixed_qkv"],
            tensors["a"],
            tensors["b"],
            tensors["z"],
            tensors["conv_weight"],
            tensors["A_log"],
            tensors["dt_bias"],
            tensors["norm_weight"],
            self.config,
            initial_state=state,
        )

    @staticmethod
    def slice_inputs(tensors, start, end):
        sliced = {}
        for name, value in tensors.items():
            sliced[name] = value[start:end] if name in {"mixed_qkv", "a", "b", "z"} else value
        return sliced

    def test_recurrence_matches_independent_fp64_equations(self):
        cfg = self.config
        length = 4
        q = torch.randn(length, cfg.num_key_heads, cfg.key_head_dim)
        k = torch.randn_like(q)
        v = torch.randn(length, cfg.num_value_heads, cfg.value_head_dim)
        a = torch.randn(length, cfg.num_value_heads)
        b = torch.randn(length, cfg.num_value_heads)
        A_log = torch.randn(cfg.num_value_heads)
        dt_bias = torch.randn(cfg.num_value_heads)
        initial = torch.randn(
            cfg.num_value_heads,
            cfg.key_head_dim,
            cfg.value_head_dim,
            dtype=torch.float32,
        )

        state = initial.double()
        expected = []
        ratio = cfg.num_value_heads // cfg.num_key_heads
        for token in range(length):
            qhat = q[token].double()
            qhat = qhat * torch.rsqrt(qhat.square().sum(-1, keepdim=True) + cfg.eps)
            khat = k[token].double()
            khat = khat * torch.rsqrt(khat.square().sum(-1, keepdim=True) + cfg.eps)
            q_expanded = qhat.repeat_interleave(ratio, dim=0)
            k_expanded = khat.repeat_interleave(ratio, dim=0)
            gate = -A_log.double().exp() * torch.nn.functional.softplus(
                a[token].double() + dt_bias.double()
            )
            decayed = state * gate.exp()[:, None, None]
            error = v[token].double() - torch.einsum("hk,hkv->hv", k_expanded, decayed)
            delta = b[token].double().sigmoid()[:, None] * error
            state = decayed + torch.einsum("hk,hv->hkv", k_expanded, delta)
            expected.append(
                torch.einsum("hk,hkv->hv", q_expanded, state)
                / (cfg.key_head_dim**0.5)
            )

        output, final_state = gated_delta_recurrence(
            q, k, v, a, b, A_log, dt_bias, initial_state=initial, eps=cfg.eps
        )
        torch.testing.assert_close(
            output.double(), torch.stack(expected), rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(final_state.double(), state, rtol=2e-5, atol=2e-6)
        self.assertEqual(final_state.dtype, torch.float32)

    def test_value_heads_expand_each_key_head_contiguously_three_times(self):
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        k = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        v = torch.ones(1, 6, 1)
        a = torch.full((1, 6), -20.0)
        b = torch.full((1, 6), 20.0)
        output, _ = gated_delta_recurrence(
            q,
            k,
            v,
            a,
            b,
            torch.zeros(6),
            torch.zeros(6),
            eps=1e-6,
        )
        self.assertTrue(torch.all(output[0, :3, 0] > 0.7))
        torch.testing.assert_close(output[0, 3:, 0], torch.zeros(3), rtol=0, atol=0)

    def test_output_norm_uses_sigmoid_gate_instead_of_swish(self):
        x = torch.tensor([[[2.0, -1.0], [1.5, 0.5]]])
        z = torch.tensor([[[-3.0, -1.0], [-2.0, -4.0]]])
        weight = torch.tensor([1.25, -0.75])
        rms = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
        expected = rms * weight.double() * z.double().sigmoid()
        swish = rms * weight.double() * (z.double() * z.double().sigmoid())

        actual = sigmoid_gated_rms_norm(x, z, weight, eps=1e-6)
        torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
        self.assertGreater((actual.double() - swish).abs().max().item(), 0.1)

    def test_chunking_matches_full_run_and_does_not_mutate_input_state(self):
        tensors = self.make_inputs(length=9)
        full_output, full_state = self.run_gdn(tensors)

        state = zero_state(self.config, input_dtype=torch.float32, device="cpu")
        pieces = []
        for start, end in ((0, 2), (2, 3), (3, 7), (7, 9)):
            old_conv = state.conv_state.clone()
            old_recurrent = state.recurrent_state.clone()
            piece, next_state = self.run_gdn(
                self.slice_inputs(tensors, start, end), state=state
            )
            torch.testing.assert_close(state.conv_state, old_conv, rtol=0, atol=0)
            torch.testing.assert_close(
                state.recurrent_state, old_recurrent, rtol=0, atol=0
            )
            pieces.append(piece)
            state = next_state

        torch.testing.assert_close(
            torch.cat(pieces), full_output, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            state.conv_state, full_state.conv_state, rtol=0, atol=0
        )
        torch.testing.assert_close(
            state.recurrent_state,
            full_state.recurrent_state,
            rtol=2e-5,
            atol=2e-6,
        )

    def test_token_decode_matches_prefill_and_reset_is_fresh(self):
        tensors = self.make_inputs(length=7, dtype=torch.bfloat16)
        full_output, full_state = self.run_gdn(tensors)

        state = None
        decoded = []
        for token in range(7):
            output, state = self.run_gdn(
                self.slice_inputs(tensors, token, token + 1), state=state
            )
            decoded.append(output)

        torch.testing.assert_close(
            torch.cat(decoded).float(), full_output.float(), rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(state.conv_state, full_state.conv_state, rtol=0, atol=0)
        torch.testing.assert_close(
            state.recurrent_state, full_state.recurrent_state, rtol=2e-5, atol=2e-6
        )
        self.assertEqual(state.recurrent_state.dtype, torch.float32)
        self.assertEqual(state.conv_state.dtype, torch.bfloat16)

        reset = zero_state(self.config, input_dtype=torch.bfloat16, device="cpu")
        self.assertIsInstance(reset, GDNState)
        self.assertTrue(torch.equal(reset.conv_state, torch.zeros_like(reset.conv_state)))
        self.assertTrue(
            torch.equal(reset.recurrent_state, torch.zeros_like(reset.recurrent_state))
        )
        reset_output, reset_final = self.run_gdn(tensors, state=reset)
        torch.testing.assert_close(reset_output, full_output, rtol=0, atol=0)
        torch.testing.assert_close(reset_final.conv_state, full_state.conv_state, rtol=0, atol=0)
        torch.testing.assert_close(
            reset_final.recurrent_state,
            full_state.recurrent_state,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
