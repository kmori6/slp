import pytest
import torch

from slp.modules.conformer.feed_forward import FeedForwardModule


@pytest.fixture
def module() -> FeedForwardModule:
    return FeedForwardModule(input_size=64, hidden_size=256, dropout_rate=0.1)


def test_output_shape(module: FeedForwardModule):
    x = torch.randn(2, 10, 64)

    output = module(x)

    assert output.shape == (2, 10, 64)


def test_gradient_flow(module: FeedForwardModule):
    x = torch.randn(2, 10, 64, requires_grad=True)

    output = module(x)

    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
