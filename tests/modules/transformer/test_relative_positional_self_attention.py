import pytest
import torch

from slp.modules.transformer.relative_positional_self_attention import (
    RelativePositionalSelfAttention,
)


@pytest.fixture
def module() -> RelativePositionalSelfAttention:
    return RelativePositionalSelfAttention(hidden_size=64, d_k=32, num_heads=2, dropout_rate=0.1, window_size=4)


def test_output_shape(module: RelativePositionalSelfAttention):
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, x, x, mask)

    assert output.shape == (2, 10, 64)


def test_gradient_flow(module: RelativePositionalSelfAttention):
    x = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, x, x, mask)
    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
