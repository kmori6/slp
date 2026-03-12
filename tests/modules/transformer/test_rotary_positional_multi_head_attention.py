import pytest
import torch

from slp.modules.transformer.positional_encoding import sinusoidal_positional_encoding
from slp.modules.transformer.rotary_positional_multi_head_attention import (
    RotaryPositionalMultiHeadAttention,
    apply_rotary_embedding,
)


def _manual_rope_rotation(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply RoPE with the explicit 2D rotation formula from RoFormer."""
    sin = p[:, None, :, 0::2]
    cos = p[:, None, :, 1::2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(3, -1)


def test_apply_rotary_embedding():
    b, h, t, s, d_k = 2, 4, 5, 6, 8
    q = torch.randn(b, h, t, d_k)
    k = torch.randn(b, h, s, d_k)
    p_q = sinusoidal_positional_encoding(d_k, t)[None, :, :]
    p_k = sinusoidal_positional_encoding(d_k, s)[None, :, :]

    q_out, k_out = apply_rotary_embedding(q, k, p_q, p_k)
    q_expected = _manual_rope_rotation(q, p_q)
    k_expected = _manual_rope_rotation(k, p_k)

    assert torch.allclose(q_out, q_expected)
    assert torch.allclose(k_out, k_expected)


@pytest.fixture
def module() -> RotaryPositionalMultiHeadAttention:
    return RotaryPositionalMultiHeadAttention(hidden_size=64, d_k=16, num_heads=4, num_kv_heads=4, dropout_rate=0.0)


def test_output_shape(module: RotaryPositionalMultiHeadAttention):
    batch_size, target_seq_len, source_seq_len = 4, 10, 12
    q = torch.randn(batch_size, target_seq_len, 64)
    k = torch.randn(batch_size, source_seq_len, 64)
    v = torch.randn(batch_size, source_seq_len, 64)
    mask = torch.ones(batch_size, target_seq_len, source_seq_len, dtype=torch.bool)
    p_q = sinusoidal_positional_encoding(16, target_seq_len)[None, :, :]
    p_k = sinusoidal_positional_encoding(16, source_seq_len)[None, :, :]

    output = module(q, k, v, mask, p_q, p_k)

    assert output.shape == (batch_size, target_seq_len, 64)


def test_gradient_flow(module: RotaryPositionalMultiHeadAttention):
    batch_size, target_seq_len, source_seq_len = 2, 10, 12
    q = torch.randn(batch_size, target_seq_len, 64, requires_grad=True)
    k = torch.randn(batch_size, source_seq_len, 64, requires_grad=True)
    v = torch.randn(batch_size, source_seq_len, 64, requires_grad=True)
    mask = torch.ones(batch_size, target_seq_len, source_seq_len, dtype=torch.bool)
    p_q = sinusoidal_positional_encoding(16, target_seq_len)[None, :, :]
    p_k = sinusoidal_positional_encoding(16, source_seq_len)[None, :, :]

    output = module(q, k, v, mask, p_q, p_k)
    loss = output.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
