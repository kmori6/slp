import pytest
import torch

from slp.modules.transformer.swiglu_feed_forward import SwiGLUFeedForward


@pytest.fixture
def module() -> SwiGLUFeedForward:
    return SwiGLUFeedForward(input_size=64, hidden_size=256, dropout_rate=0.1)


def test_output_shape(module: SwiGLUFeedForward):
    x = torch.randn(4, 10, 64)

    output = module(x)

    assert output.shape == (4, 10, 64)


def test_gradient_flow(module: SwiGLUFeedForward):
    x = torch.randn(2, 10, 64, requires_grad=True)

    output = module(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
