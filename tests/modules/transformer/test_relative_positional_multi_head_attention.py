import pytest
import torch

from slp.modules.transformer.relative_positional_multi_head_attention import (
    RelativePositionalMultiHeadAttention,
    left_shift,
)


def test_left_shift():
    # expected[i, j] = x[i, 2L - 1 - i + j] for j <= i (causal part), else 0.
    seq_len = 5
    x = torch.arange(2 * seq_len, dtype=torch.long).repeat(1, seq_len, 1)
    expected = torch.zeros(1, seq_len, seq_len, dtype=torch.long)
    for i in range(seq_len):
        for j in range(i + 1):
            col = 2 * seq_len - 1 - i + j
            expected[0, i, j] = x[0, i, col]

    result = left_shift(x)

    assert torch.equal(result, expected)


@pytest.fixture
def module() -> RelativePositionalMultiHeadAttention:
    return RelativePositionalMultiHeadAttention(hidden_size=64, d_k=16, num_heads=4, dropout_rate=0.0)


def test_output_shape(module: RelativePositionalMultiHeadAttention):
    x = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, 10, dtype=torch.bool)

    output = module(x, x, x, mask)

    assert output.shape == (4, 10, 64)


def test_gradient_flow(module: RelativePositionalMultiHeadAttention):
    q = torch.randn(2, 10, 64, requires_grad=True)
    k = torch.randn(2, 10, 64, requires_grad=True)
    v = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(q, k, v, mask)
    loss = output.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
