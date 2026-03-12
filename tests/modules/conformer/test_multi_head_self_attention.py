import pytest
import torch

from slp.modules.conformer.multi_head_self_attention import MultiHeadSelfAttentionModule


@pytest.fixture
def module() -> MultiHeadSelfAttentionModule:
    return MultiHeadSelfAttentionModule(input_size=64, num_heads=4, dropout_rate=0.1)


def test_output_shape(module: MultiHeadSelfAttentionModule):
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    assert output.shape == (2, 10, 64)


def test_gradient_flow(module: MultiHeadSelfAttentionModule):
    x = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
